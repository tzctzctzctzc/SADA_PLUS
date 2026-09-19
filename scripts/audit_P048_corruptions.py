#!/usr/bin/env python3
"""Visually confirm P048 cross-role negatives without judging absolute target correctness."""

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
RUN_DIR = ROOT / "runs/P048_visible_role_contrast"
CONFIG = json.loads((ROOT / "configs/P048_visible_role_contrast.json").read_text())
AUDIT = CONFIG["visual_confirmation"]

assert Path.cwd().resolve() == ROOT
assert json.loads((RUN_DIR / "implementation_audit_static.json").read_text())["classification"] == "faithful"
assert os.environ.get("CUDA_VISIBLE_DEVICES") == CONFIG["gpu_uuid"]
assert torch.cuda.device_count() == 1
assert "GPU-" + str(torch.cuda.get_device_properties(0).uuid) == CONFIG["gpu_uuid"]
torch.manual_seed(CONFIG["training"]["seed"])
torch.backends.cuda.matmul.allow_tf32 = False

bundles = [
    json.loads(line)
    for line in (RUN_DIR / "bundles.jsonl").open()
    if line.strip() and json.loads(line)["split"] == "confirmation"
]
assert len(bundles) == CONFIG["data"]["counts"]["confirmation"]
manifest = json.loads((RUN_DIR / "image_manifest.json").read_text())
assert manifest["complete"]
images = {row["idx"]: row for row in manifest["entries"]}

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
label_ids = [processor.tokenizer.encode(label, add_special_tokens=False) for label in ("True", "False")]
assert all(len(value) == 1 for value in label_ids)
label_ids = [value[0] for value in label_ids]


@functools.lru_cache(maxsize=256)
def image_for(product_id: str) -> Image.Image:
    entry = images[product_id]
    data = (ROOT / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def make_item(bundle: dict, kind: str) -> dict:
    key = bundle["active_role_key"]
    wrong_id = bundle["active_wrong_source_id"]
    if kind == "identity_positive":
        product_ids, expected = [wrong_id, wrong_id], "True"
    else:
        assert kind == "cross_role_negative"
        product_ids, expected = [wrong_id, bundle["positive_target_id"]], "False"
    messages = [
        {"role": "system", "content": "You are a strict visual data auditor."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect only visible evidence in the product images. If the relation is ambiguous, answer False.\nImage one:\n"},
                {"type": "image"},
                {"type": "text", "text": "\nImage two:\n"},
                {"type": "image"},
                {"type": "text", "text": f"\nIgnoring the background, do these two products clearly share the same visible '{key}' attribute? Answer exactly one word: True or False"},
            ],
        },
    ]
    return {
        "bundle_id": bundle["bundle_id"],
        "kind": kind,
        "role_key": key,
        "expected": expected,
        "product_ids": product_ids,
        "messages": messages,
    }


def prepare(items: list[dict]) -> dict[str, torch.Tensor]:
    texts = [processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True) for item in items]
    flat_images = [image_for(product_id) for item in items for product_id in item["product_ids"]]
    batch = processor(text=texts, images=flat_images, padding=True, return_tensors="pt")
    assert len(batch["image_grid_thw"]) == 2 * len(items)
    return {key: value.to(model.device) for key, value in batch.items()}


items = [make_item(bundle, kind) for bundle in bundles for kind in AUDIT["checks"]]
results = []
torch.cuda.reset_peak_memory_stats()
for start in range(0, len(items), AUDIT["batch_size"]):
    current = items[start : start + AUDIT["batch_size"]]
    batch = prepare(current)
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, label_ids].float().cpu()
    for item, pair in zip(current, logits.tolist()):
        margin = pair[0] - pair[1]
        passed = margin > 0 if item["expected"] == "True" else margin < 0
        results.append(
            {
                "bundle_id": item["bundle_id"],
                "kind": item["kind"],
                "role_key": item["role_key"],
                "expected": item["expected"],
                "pass": passed,
                "true_false_logits": pair,
                "margin_true_minus_false": margin,
                "product_ids": item["product_ids"],
                "prompt_sha256": hashlib.sha256(json.dumps(item["messages"], sort_keys=True).encode()).hexdigest(),
            }
        )
    print(f"completed {min(start + AUDIT['batch_size'], len(items))}/{len(items)}", flush=True)

with (RUN_DIR / "corruption_audit_results.jsonl").open("w") as handle:
    for row in results:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
per_check = {}
for kind in AUDIT["checks"]:
    selected = [row for row in results if row["kind"] == kind]
    passed = sum(row["pass"] for row in selected)
    per_check[kind] = {"passed": passed, "total": len(selected), "pass_rate": passed / len(selected)}
gates = {
    "identity_calibration": per_check["identity_positive"]["pass_rate"] >= AUDIT["identity_pass_rate_min"],
    "cross_role_negative": per_check["cross_role_negative"]["pass_rate"] >= AUDIT["cross_role_negative_pass_rate_min"],
}
summary = {
    "status": "pass" if all(gates.values()) else "fail",
    "gpu_uuid": CONFIG["gpu_uuid"],
    "judge_model": AUDIT["judge_model"],
    "bundles": len(bundles),
    "checks": len(results),
    "role_counts": Counter(row["role_key"] for row in results if row["kind"] == "cross_role_negative"),
    "per_check": per_check,
    "gates": gates,
    "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
}
(RUN_DIR / "corruption_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["status"] != "pass":
    raise SystemExit(2)
