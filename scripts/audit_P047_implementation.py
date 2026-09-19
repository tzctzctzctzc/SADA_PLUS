#!/usr/bin/env python3
"""Audit P047 data semantics, loss direction, and implementation completeness before GPU use."""

from __future__ import annotations

import hashlib
import json
import py_compile
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P047_position_controlled_role_contrast.json"
RUN_DIR = ROOT / "runs/P047_position_controlled_role_contrast"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: object) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


assert Path.cwd().resolve() == ROOT
config = json.loads(CONFIG_PATH.read_text())
paths = [
    ROOT / "scripts/prepare_P047.py",
    ROOT / "scripts/extract_P047_images.py",
    ROOT / "scripts/audit_P047_corruptions.py",
    ROOT / "scripts/audit_P047_implementation.py",
    ROOT / "scripts/analyze_P047.py",
    ROOT / "src/run_P047.py",
]
for path in paths:
    py_compile.compile(str(path), doraise=True)

checks: dict[str, bool] = {}
details: dict[str, object] = {}
checks["frozen_config_and_hashes"] = (
    config["status"] == "preregistered_pending_data_and_implementation_audit"
    and config["gpu_uuid"] == "GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f"
    and sha256_file(ROOT / config["data"]["query_path"]) == config["data"]["query_sha256"]
    and config["training"]["arms"]
    == ["ordinary_retrieval", "paraphrase_invariance", "image_role_swap", "text_role_swap"]
)

bundles = [json.loads(line) for line in (RUN_DIR / "bundles.jsonl").open() if line.strip()]
split_counts = Counter(row["split"] for row in bundles)
checks["split_counts"] = split_counts == Counter(config["data"]["counts"])
products = [product_id for row in bundles for product_id in row["product_ids"]]
checks["within_pilot_global_product_disjointness"] = len(products) == len(set(products))
checks["balanced_role_positions"] = all(
    Counter(row["active_role"] for row in bundles if row["split"] == split)
    == Counter({0: count // 2, 1: count // 2})
    for split, count in config["data"]["counts"].items()
)

metadata = {
    row["idx"]: row
    for row in (
        json.loads(line)
        for line in (ROOT / config["data"]["metadata_path"]).open()
        if line.strip()
    )
}
same_images = []
balanced_pointers = []
cross_mismatch = []
paraphrase_preserves = []
no_value_leak = []
for row in bundles:
    original_sources = list(row["source_ids"])
    text_swap_sources = list(row["source_ids"])
    same_images.append(original_sources == text_swap_sources)
    original = row["instructions"]["original"]
    swapped = row["instructions"]["text_role_swap"]
    balanced_pointers.append(
        original.count("Product 1") == original.count("Product 2") == 1
        and swapped.count("Product 1") == swapped.count("Product 2") == 1
        and original != swapped
    )
    active = row["active_role"]
    key = row["active_role_key"]
    target = metadata[row["positive_target_id"]]
    correct = metadata[row["source_ids"][active]]
    wrong = metadata[row["source_ids"][1 - active]]
    cross_mismatch.append(
        row["role_keys"][active] == key
        and key in correct["attribute"]
        and key in wrong["attribute"]
        and canonical(correct["attribute"][key]) == canonical(target["attribute"][key])
        and canonical(wrong["attribute"][key]) != canonical(target["attribute"][key])
    )
    paraphrase = row["instructions"]["paraphrase"]
    paraphrase_preserves.append(
        row["role_keys"][0] in paraphrase.lower()
        and row["role_keys"][1] in paraphrase.lower()
        and paraphrase.count("Product 1") == paraphrase.count("Product 2") == 1
    )
    values = [canonical(target["attribute"][key]) for key in row["role_keys"]]
    no_value_leak.append(
        not any(len(value) >= 3 and any(value in text.lower() for text in row["instructions"].values()) for value in values)
    )
checks["same_images_same_order_for_text_swap"] = all(same_images)
checks["balanced_product_pointer_swap"] = all(balanced_pointers)
checks["metadata_verified_cross_role_mismatch"] = all(cross_mismatch)
checks["paraphrase_role_preservation"] = all(paraphrase_preserves)
checks["no_attribute_value_leak"] = all(no_value_leak)

temperature = config["training"]["temperature"]
margin = config["training"]["role_margin"]
good = temperature * F.softplus(torch.tensor((margin - (0.8 - 0.2)) / temperature))
bad = temperature * F.softplus(torch.tensor((margin - (0.2 - 0.8)) / temperature))
probe = torch.tensor([0.4], requires_grad=True)
probe_loss = temperature * F.softplus((margin - (probe - torch.tensor([0.1]))) / temperature)
probe_loss.backward()
checks["loss_direction_unit_test"] = float(good) < float(bad) and float(probe.grad) < 0
details.update({"loss_good": float(good), "loss_bad": float(bad), "positive_score_gradient": float(probe.grad)})

source = (ROOT / "src/run_P047.py").read_text()
checks["full_method_not_head_only"] = all(
    token in source
    for token in [
        '"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"',
        'if ".visual.merger." in name',
        'query_variants = ["original", "paraphrase", "image_role_swap", "text_role_swap"]',
        'elif variant == "text_role_swap"',
        'if variant == "image_role_swap"',
        'queries.retain_grad()',
        'candidates.retain_grad()',
        'candidate_eval_batch_size',
    ]
)
checks["matched_arms_and_budget"] = (
    config["training"]["steps"] == config["data"]["counts"]["train"]
    and config["training"]["seed"] == 20260919
)

if all(checks.values()):
    classification = "faithful"
elif not checks["loss_direction_unit_test"] or not checks["metadata_verified_cross_role_mismatch"]:
    classification = "incorrect"
elif not checks["full_method_not_head_only"]:
    classification = "minimal"
else:
    classification = "conservative"

summary = {
    "status": "pass" if classification == "faithful" else "fail",
    "classification": classification,
    "scope": "static code, frozen data, loss semantics, and method completeness; visual validity and runtime gradient coverage are separate required gates",
    "checks": checks,
    "details": details,
    "split_counts": split_counts,
    "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths + [CONFIG_PATH]},
}
(RUN_DIR / "implementation_audit_static.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["status"] != "pass":
    raise SystemExit(2)
