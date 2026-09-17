#!/usr/bin/env python3
"""Visually falsify the frozen P044 reference-subset noise decomposition screen."""

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
RUN_DIR = ROOT / "runs/P044_reference_subset_noise_decomposition_screen"
RUN_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = json.loads((ROOT / "configs/P044_reference_subset_noise_decomposition_screen.json").read_text())
SOURCE_RUN_DIR = ROOT / CONFIG["source"]["run_dir"]
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
    for line in (SOURCE_RUN_DIR / CONFIG["source"]["bundles"]).open()
    if line.strip()
]
assert len(bundles) == CONFIG["source"]["bundles_count"]
all_products = [product_id for row in bundles for product_id in row["product_ids"]]
assert len(all_products) == len(set(all_products)) == 3 * len(bundles)

raw_manifest = json.loads((SOURCE_RUN_DIR / CONFIG["source"]["raw_manifest"]).read_text())
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
    target_cf = transformed_spec(bundle, "target")
    if kind == "color_relation_positive":
        specs = [color_raw, target_raw]
        labels = ["Color reference", "Official target"]
        question = "Ignoring the background, do these products clearly share the same dominant product color?"
        expected = "True"
    elif kind == "other_relation_positive":
        specs = [other_raw, target_raw]
        labels = [f"{bundle['other_key']} reference", "Official target"]
        question = f"Ignoring the background, do these products clearly share the same visible '{bundle['other_key']}' attribute?"
        expected = "True"
    elif kind == "color_identity_positive":
        specs = [color_raw, color_raw]
        labels = ["Image one", "Image two"]
        question = "Ignoring the background, do these products clearly share the same dominant product color?"
        expected = "True"
    elif kind == "other_identity_positive":
        specs = [other_raw, other_raw]
        labels = ["Image one", "Image two"]
        question = f"Ignoring the background, do these products clearly share the same visible '{bundle['other_key']}' attribute?"
        expected = "True"
    elif kind == "color_cross_negative":
        specs = [color_raw, target_cf]
        labels = ["Original color reference", "Color-intervened target"]
        question = "Ignoring the background, do these products still clearly share the same dominant product color?"
        expected = "False"
    else:
        assert kind == "identity_change_negative"
        specs = [color_raw, color_raw]
        labels = ["Image one", "Image two"]
        question = "Ignoring the background, did the dominant product color visibly change from Image one to Image two?"
        expected = "False"
    content = [{"type": "text", "text": "Inspect only visible evidence in the product images. If a required relation is ambiguous, answer False.\n"}]
    for label in labels:
        content.extend([{"type": "text", "text": f"\n{label}:\n"}, {"type": "image"}])
    content.append({"type": "text", "text": f"\n{question} Answer exactly one word: True or False"})
    messages = [{"role": "system", "content": "You are a strict visual data auditor."}, {"role": "user", "content": content}]
    return {"bundle_id": bundle["bundle_id"], "other_key": bundle["other_key"], "kind": kind, "expected": expected, "specs": specs, "messages": messages}

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

by_bundle: dict[str, dict[str, dict]] = {}
for row in results:
    by_bundle.setdefault(row["bundle_id"], {})[row["kind"]] = row
assert len(results) == len(bundles) * len(kinds)
assert all(set(rows) == set(kinds) for rows in by_bundle.values())

route_counts = Counter()
for rows in by_bundle.values():
    color_ok = rows["color_relation_positive"]["pass"]
    other_ok = rows["other_relation_positive"]["pass"]
    route = "clean_both" if color_ok and other_ok else "color_only" if color_ok else "other_only" if other_ok else "neither"
    route_counts[route] += 1

per_check = {}
for kind in kinds:
    selected = [row for row in results if row["kind"] == kind]
    count = sum(row["pass"] for row in selected)
    per_check[kind] = {"passed": count, "total": len(selected), "pass_rate": count / len(selected), "mean_margin_true_minus_false": sum(row["margin_true_minus_false"] for row in selected) / len(selected)}

n = len(bundles)
clean_rate = route_counts["clean_both"] / n
partial_rate = (route_counts["color_only"] + route_counts["other_only"]) / n
routeable_rate = 1 - route_counts["neither"] / n
calibration_ok = all(per_check[k]["pass_rate"] >= AUDIT["calibration_pass_rate_min"] for k in AUDIT["calibration_checks"])
gates = {
    "calibration": calibration_ok,
    "routeable_rate": routeable_rate >= AUDIT["routeable_rate_min"],
    "partial_rate": partial_rate >= AUDIT["partial_rate_min"],
    "clean_rate": clean_rate >= AUDIT["clean_rate_min"],
    "color_only_rate": route_counts["color_only"] / n >= AUDIT["each_partial_branch_rate_min"],
    "other_only_rate": route_counts["other_only"] / n >= AUDIT["each_partial_branch_rate_min"],
}
status = "pass" if all(gates.values()) else "fail"
summary = {
    "status": status, "gpu_uuid": EXPECTED_GPU, "judge_model": AUDIT["judge_model"], "judge_revision": AUDIT["judge_revision"],
    "bundles": n, "checks": len(results), "route_counts": dict(route_counts), "routeable_rate": routeable_rate, "partial_rate": partial_rate, "clean_rate": clean_rate,
    "gates": gates, "all_gates_pass": all(gates.values()), "per_check": per_check, "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
}
(RUN_DIR / "visual_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
if status != "pass":
    raise SystemExit(2)
