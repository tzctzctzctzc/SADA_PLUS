#!/usr/bin/env python3
"""Visually falsify the frozen P042 role-conditioned transform contract."""

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
RUN_DIR = ROOT / "runs/P042_role_conditioned_equivariance"
CONFIG = json.loads((ROOT / "configs/P042_role_conditioned_equivariance.json").read_text())
AUDIT = CONFIG["visual_audit"]
EXPECTED_GPU = CONFIG["gpu_uuid"]

assert Path.cwd().resolve() == ROOT
assert os.environ.get("CUDA_VISIBLE_DEVICES") == EXPECTED_GPU
assert torch.cuda.device_count() == 1
assert "A100" in torch.cuda.get_device_name(0)
torch.set_num_threads(8)
torch.manual_seed(20260918)
torch.backends.cuda.matmul.allow_tf32 = False

bundles = [
    json.loads(line)
    for line in (RUN_DIR / "audit_bundles.jsonl").open()
    if line.strip()
]
assert len(bundles) == CONFIG["data"]["audit_bundles"]
all_products = [product_id for row in bundles for product_id in row["product_ids"]]
assert len(all_products) == len(set(all_products)) == 3 * len(bundles)

raw_manifest = json.loads((RUN_DIR / "image_manifest_raw.json").read_text())
assert raw_manifest["complete"]
raw_images = {row["idx"]: row for row in raw_manifest["entries"]}

model_path = ROOT / AUDIT["judge_model"]
model_manifest = json.loads((model_path / "source_manifest.json").read_text())
assert model_manifest["revision"] == AUDIT["judge_revision"]
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
assert all(len(value) == 1 for value in label_ids)
label_ids = [value[0] for value in label_ids]


def raw_spec(product_id: str) -> dict:
    entry = raw_images[product_id]
    return {"path": entry["path"], "sha256": entry["sha256"]}


def transformed_spec(bundle: dict, role: str) -> dict:
    metrics = bundle["transform_metrics"][role]
    return {"path": metrics["path"], "sha256": metrics["sha256"]}


@functools.lru_cache(maxsize=700)
def image_for(path_text: str, expected_sha: str) -> Image.Image:
    path = ROOT / path_text
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_sha
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def make_item(bundle: dict, kind: str) -> dict:
    color_raw = raw_spec(bundle["color_source_id"])
    other_raw = raw_spec(bundle["other_source_id"])
    target_raw = raw_spec(bundle["positive_target_id"])
    color_cf = transformed_spec(bundle, "color_source")
    other_cf = transformed_spec(bundle, "other_source")
    target_cf = transformed_spec(bundle, "target")
    if kind == "base_alignment_positive":
        specs = [color_raw, other_raw, target_raw]
        labels = ["Color reference", f"{bundle['other_key']} reference", "Target"]
        question = (
            f"The Target must visibly share the dominant product color of the Color reference "
            f"and the {bundle['other_key']} of the {bundle['other_key']} reference. "
            "Ignore the image background. Are both relations clearly correct?"
        )
        expected = "True"
    elif kind == "active_equivariance_positive":
        specs = [color_raw, color_cf, target_raw, target_cf]
        labels = ["Color reference before", "Color reference after", "Target before", "Target after"]
        question = (
            "The two 'after' images were produced by the same color intervention. "
            "The product color must visibly change in both before/after pairs, product identity "
            "and geometry must remain the same, and the Target after must retain the same-color "
            "relation to the Color reference after. Are all conditions clearly correct?"
        )
        expected = "True"
    elif kind == "null_invariance_positive":
        specs = [other_raw, other_cf]
        labels = [f"{bundle['other_key']} reference before", f"{bundle['other_key']} reference after"]
        question = (
            "The product color must visibly change between the images, while product identity, "
            f"geometry, and the requested attribute '{bundle['other_key']}' remain the same. "
            "Are all conditions clearly correct?"
        )
        expected = "True"
    elif kind == "cross_effect_negative":
        specs = [color_raw, target_cf]
        labels = ["Original Color reference", "Color-intervened Target"]
        question = (
            "Ignoring the background, do these two products still clearly have the same dominant "
            "product color?"
        )
        expected = "False"
    else:
        assert kind == "identity_change_negative"
        specs = [color_raw, color_raw]
        labels = ["Image one", "Image two"]
        question = (
            "Ignoring the background, did the dominant product color visibly change from Image one "
            "to Image two?"
        )
        expected = "False"
    content = [{
        "type": "text",
        "text": (
            "Inspect only visible evidence in the product images. If a required relation is "
            "ambiguous, answer False.\n"
        ),
    }]
    for label in labels:
        content.extend([
            {"type": "text", "text": f"\n{label}:\n"},
            {"type": "image"},
        ])
    content.append({
        "type": "text",
        "text": f"\n{question} Answer exactly one word: True or False",
    })
    messages = [
        {"role": "system", "content": "You are a strict visual data auditor."},
        {"role": "user", "content": content},
    ]
    return {
        "bundle_id": bundle["bundle_id"],
        "other_key": bundle["other_key"],
        "kind": kind,
        "expected": expected,
        "specs": specs,
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
        flat_images.extend(
            image_for(spec["path"], spec["sha256"]) for spec in item["specs"]
        )
    batch = processor(text=texts, images=flat_images, padding=True, return_tensors="pt")
    assert len(batch["image_grid_thw"]) == sum(len(item["specs"]) for item in items)
    for index, item in enumerate(items):
        starts = int(
            (batch["input_ids"][index] == model.config.vision_start_token_id).sum()
        )
        assert starts == len(item["specs"])
        assert int(batch["attention_mask"][index, -1]) == 1
    return {key: value.to(model.device) for key, value in batch.items()}


kinds = AUDIT["required_checks"]
items = [make_item(bundle, kind) for bundle in bundles for kind in kinds]
result_path = RUN_DIR / "visual_audit_results.jsonl"
result_path.write_text("")
results = []
torch.cuda.reset_peak_memory_stats()
for start in range(0, len(items), AUDIT["batch_size"]):
    current = items[start:start + AUDIT["batch_size"]]
    batch = prepare(current)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, label_ids].float().cpu()
    probabilities = logits.softmax(dim=-1)
    with result_path.open("a") as handle:
        for item, pair, probability in zip(current, logits.tolist(), probabilities.tolist()):
            margin = pair[0] - pair[1]
            passed = (
                margin > AUDIT["positive_margin_min"]
                if item["expected"] == "True"
                else margin < AUDIT["negative_margin_max"]
            )
            row = {
                "bundle_id": item["bundle_id"],
                "other_key": item["other_key"],
                "kind": item["kind"],
                "expected": item["expected"],
                "pass": passed,
                "true_false_logits": pair,
                "true_false_probabilities": probability,
                "margin_true_minus_false": margin,
                "image_paths": [spec["path"] for spec in item["specs"]],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(item["messages"], sort_keys=True).encode()
                ).hexdigest(),
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    del batch, logits, probabilities
    if (start // AUDIT["batch_size"] + 1) % 10 == 0:
        print(f"completed {min(start + AUDIT['batch_size'], len(items))}/{len(items)}", flush=True)

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
    count = sum(row["pass"] for row in selected)
    per_check[kind] = {
        "passed": count,
        "total": len(selected),
        "pass_rate": count / len(selected),
        "mean_margin_true_minus_false": sum(
            row["margin_true_minus_false"] for row in selected
        ) / len(selected),
    }
negative_ok = all(
    per_check[kind]["pass_rate"] >= 0.95
    for kind in ("cross_effect_negative", "identity_change_negative")
)
failure_keys = Counter(
    f"{row['other_key']}::{row['kind']}"
    for row in results
    if not row["pass"]
)
status = (
    "pass"
    if pass_rate >= AUDIT["bundle_pass_rate_min"] and negative_ok
    else "fail"
)
summary = {
    "status": status,
    "gpu_uuid": EXPECTED_GPU,
    "judge_model": AUDIT["judge_model"],
    "judge_revision": AUDIT["judge_revision"],
    "bundles": len(bundles),
    "checks": len(results),
    "bundle_passed": pass_count,
    "bundle_pass_rate": pass_rate,
    "required_bundle_pass_rate": AUDIT["bundle_pass_rate_min"],
    "negative_controls_pass": negative_ok,
    "per_check": per_check,
    "top_failure_keys": failure_keys.most_common(20),
    "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
}
(RUN_DIR / "visual_audit_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if status != "pass":
    raise SystemExit(2)
