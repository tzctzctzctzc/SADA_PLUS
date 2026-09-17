#!/usr/bin/env python3
"""Visually audit the frozen P041 bundle contract with positive and swapped controls."""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P041_selective_dependence"
CONFIG = json.loads((ROOT / "configs/P041_selective_dependence_pilot.json").read_text())
AUDIT = CONFIG["visual_audit"]
EXPECTED_GPU = CONFIG["gpu_uuid"]

assert Path.cwd().resolve() == ROOT
assert os.environ.get("CUDA_VISIBLE_DEVICES") == EXPECTED_GPU
assert torch.cuda.device_count() == 1
assert "A100" in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(20260917)
torch.backends.cuda.matmul.allow_tf32 = False

model_path = ROOT / AUDIT["judge_model"]
source_manifest = json.loads((model_path / "source_manifest.json").read_text())
assert source_manifest["revision"] == AUDIT["judge_revision"]
processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True,
    min_pixels=AUDIT["min_pixels"],
    max_pixels=AUDIT["max_pixels"],
)
processor.tokenizer.padding_side = "left"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
).eval()
assert next(model.parameters()).device.type == "cuda"
label_ids = [
    processor.tokenizer.encode(label, add_special_tokens=False)
    for label in ("True", "False")
]
assert all(len(ids) == 1 for ids in label_ids)
label_ids = [ids[0] for ids in label_ids]

bundle_by_id = {
    row["bundle_id"]: row
    for row in (
        json.loads(line)
        for line in (RUN_DIR / "bundles.jsonl").open()
        if line.strip()
    )
}
contract = json.loads((RUN_DIR / "data_contract_audit.json").read_text())
audit_rows = contract["rows"]
assert len(audit_rows) == AUDIT["bundles"]
assert all(row["passed"] for row in audit_rows)
bundles = [bundle_by_id[row["bundle_id"]] for row in audit_rows]

image_manifest = json.loads((RUN_DIR / "image_manifest_audit.json").read_text())
assert image_manifest["complete"]
images = {row["idx"]: row for row in image_manifest["entries"]}


@functools.lru_cache(maxsize=700)
def load_image(product_id: str) -> Image.Image:
    entry = images[product_id]
    data = (ROOT / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def condition_lines(bundle: dict, swapped: bool) -> tuple[list[str], list[str]]:
    changed = bundle["intervened_source"]
    fixed = bundle["fixed_source"]
    before_value = changed["active_value_after"] if swapped else changed["active_value_before"]
    after_value = changed["active_value_before"] if swapped else changed["active_value_after"]
    source = [
        f"Image S-before must visibly have {changed['active_key']} = {before_value} and {changed['null_key']} = {changed['null_value']}.",
        f"Image S-after must visibly have {changed['active_key']} = {after_value} and {changed['null_key']} = {changed['null_value']}.",
    ]
    targets = [
        f"Image T-before must visibly have {changed['active_key']} = {before_value} and {fixed['key']} = {fixed['value']}.",
        f"Image T-after must visibly have {changed['active_key']} = {after_value} and {fixed['key']} = {fixed['value']}.",
        f"Image T-null must visibly have {changed['null_key']} = {changed['null_value']} and {fixed['key']} = {fixed['value']}.",
    ]
    return source, targets


def make_item(bundle: dict, kind: str) -> dict:
    swapped = "swapped" in kind
    source_lines, target_lines = condition_lines(bundle, swapped)
    if kind.startswith("source"):
        product_ids = [
            bundle["intervened_source"]["before"],
            bundle["intervened_source"]["after"],
        ]
        labels = ["S-before", "S-after"]
        requirements = source_lines
    else:
        product_ids = [
            bundle["active"]["target_before"],
            bundle["active"]["target_after"],
            bundle["null"]["target"],
        ]
        labels = ["T-before", "T-after", "T-null"]
        requirements = target_lines
    content = [{
        "type": "text",
        "text": (
            "Inspect the product images and test every stated visual condition. "
            "Use visible evidence only. If an attribute is not discernible, answer False. "
            "Answer True only when every condition is clearly satisfied.\n\n"
            + "\n".join(requirements)
            + "\n"
        ),
    }]
    for label in labels:
        content.extend([
            {"type": "text", "text": f"\nImage {label}:\n"},
            {"type": "image"},
        ])
    content.append({
        "type": "text",
        "text": "\nAre all stated conditions visibly correct? Answer exactly one word: True or False",
    })
    messages = [
        {"role": "system", "content": "You are a strict visual data auditor."},
        {"role": "user", "content": content},
    ]
    return {
        "bundle_id": bundle["bundle_id"],
        "kind": kind,
        "expected": "False" if swapped else "True",
        "product_ids": product_ids,
        "requirements": requirements,
        "messages": messages,
    }


def prepare(items: list[dict]) -> dict:
    texts = []
    flat_images = []
    for item in items:
        texts.append(
            processor.apply_chat_template(
                item["messages"], tokenize=False, add_generation_prompt=True
            )
        )
        flat_images.extend(load_image(product_id) for product_id in item["product_ids"])
    batch = processor(
        text=texts,
        images=flat_images,
        padding=True,
        return_tensors="pt",
    )
    assert len(batch["image_grid_thw"]) == sum(len(item["product_ids"]) for item in items)
    for index, item in enumerate(items):
        vision_starts = int(
            (batch["input_ids"][index] == model.config.vision_start_token_id).sum()
        )
        assert vision_starts == len(item["product_ids"])
        assert int(batch["attention_mask"][index, -1]) == 1
    return {key: value.to(model.device) for key, value in batch.items()}


kinds = AUDIT["required_checks"]
items = [make_item(bundle, kind) for bundle in bundles for kind in kinds]
output_path = RUN_DIR / "visual_audit_results.jsonl"
output_path.write_text("")
results = []
batch_size = AUDIT["batch_size"]
torch.cuda.reset_peak_memory_stats()
for start in range(0, len(items), batch_size):
    current = items[start:start + batch_size]
    batch = prepare(current)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, label_ids].float().cpu()
    probabilities = logits.softmax(dim=-1)
    with output_path.open("a") as handle:
        for item, pair, probability in zip(current, logits.tolist(), probabilities.tolist()):
            margin = pair[0] - pair[1]
            if item["expected"] == "True":
                passed = margin > AUDIT["positive_margin_min"]
            else:
                passed = margin < AUDIT["negative_margin_max"]
            row = {
                "bundle_id": item["bundle_id"],
                "kind": item["kind"],
                "expected": item["expected"],
                "pass": passed,
                "true_false_logits": pair,
                "true_false_probabilities": probability,
                "margin_true_minus_false": margin,
                "product_ids": item["product_ids"],
                "requirements": item["requirements"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(item["messages"], sort_keys=True).encode()
                ).hexdigest(),
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    del batch, logits, probabilities
    if (start // batch_size + 1) % 10 == 0:
        print(f"completed {min(start + batch_size, len(items))}/{len(items)}", flush=True)

by_bundle: dict[str, list[dict]] = {}
for row in results:
    by_bundle.setdefault(row["bundle_id"], []).append(row)
bundle_pass = {
    bundle_id: len(rows) == len(kinds) and all(row["pass"] for row in rows)
    for bundle_id, rows in by_bundle.items()
}
pass_count = sum(bundle_pass.values())
pass_rate = pass_count / len(bundles)
per_check = {}
for kind in kinds:
    selected = [row for row in results if row["kind"] == kind]
    per_check[kind] = {
        "passed": sum(row["pass"] for row in selected),
        "total": len(selected),
        "pass_rate": sum(row["pass"] for row in selected) / len(selected),
        "mean_margin_true_minus_false": sum(
            row["margin_true_minus_false"] for row in selected
        ) / len(selected),
    }
failure_keys = Counter()
for bundle in bundles:
    if not bundle_pass[bundle["bundle_id"]]:
        for row in by_bundle[bundle["bundle_id"]]:
            if not row["pass"]:
                failure_keys[
                    f"{bundle['intervened_source']['active_key']}::{row['kind']}"
                ] += 1
summary = {
    "status": "pass" if pass_rate >= AUDIT["bundle_pass_rate_min"] else "fail",
    "gpu_uuid": EXPECTED_GPU,
    "judge_model": AUDIT["judge_model"],
    "judge_revision": AUDIT["judge_revision"],
    "bundles": len(bundles),
    "checks": len(results),
    "bundle_passed": pass_count,
    "bundle_pass_rate": pass_rate,
    "required_bundle_pass_rate": AUDIT["bundle_pass_rate_min"],
    "per_check": per_check,
    "top_failure_keys": failure_keys.most_common(20),
    "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
}
(RUN_DIR / "visual_audit_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if summary["status"] != "pass":
    raise SystemExit(2)
