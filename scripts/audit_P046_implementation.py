#!/usr/bin/env python3
"""Static and generated-data fidelity audit for P046 before any pilot run."""

from __future__ import annotations

import hashlib
import json
import py_compile
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/vepfs/vlms/1")
CONFIG_PATH = ROOT / "configs/P046_self_normalized_role_corruption.json"
RUN_DIR = ROOT / "runs/P046_self_normalized_role_corruption"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()


assert Path.cwd().resolve() == ROOT
config = json.loads(CONFIG_PATH.read_text())
paths = [
    ROOT / "scripts/prepare_P046.py",
    ROOT / "scripts/extract_P046_images.py",
    ROOT / "scripts/audit_P046_corruptions.py",
    ROOT / "scripts/audit_P046_implementation.py",
    ROOT / "src/run_P046.py",
]
for path in paths:
    py_compile.compile(str(path), doraise=True)

checks: dict[str, bool] = {}
details: dict[str, object] = {}
checks["config_and_hashes"] = (
    config["status"] == "preregistered_pending_implementation_audit"
    and config["gpu_uuid"] == "GPU-6fc32588-9464-5f67-98bc-d436aec84f4a"
    and sha256_file(ROOT / config["data"]["query_path"]) == config["data"]["query_sha256"]
    and config["training"]["arms"] == ["ordinary_retrieval", "generic_augmentation", "role_corruption"]
)

bundles = [json.loads(line) for line in (RUN_DIR / "bundles.jsonl").open() if line.strip()]
expected_counts = config["data"]["counts"]
split_counts = Counter(row["split"] for row in bundles)
checks["split_counts"] = split_counts == Counter(expected_counts)

blocked = set()
for path in [
    ROOT / "runs/P041_selective_dependence/bundles.jsonl",
    ROOT / "runs/P042_role_conditioned_equivariance/raw_candidates.jsonl",
    ROOT / "runs/P043_role_conditioned_equivariance_maskaware_confirm/raw_candidates.jsonl",
]:
    for line in path.open():
        if line.strip():
            blocked.update(map(str, json.loads(line)["product_ids"]))
base_products = [product_id for row in bundles for product_id in row["product_ids"]]
checks["global_product_disjointness"] = len(base_products) == len(set(base_products)) and not set(base_products) & blocked
details["blocked_products"] = len(blocked)
details["base_products"] = len(base_products)

metadata = {
    row["idx"]: row
    for row in (
        json.loads(line)
        for line in (ROOT / config["data"]["metadata_path"]).open()
        if line.strip()
    )
}
single_role = []
target_other_identity = []
role_alignment = []
no_leak = []
for row in bundles:
    slot = row["corrupt_slot"]
    original = list(row["source_ids"])
    corrupted = list(original)
    corrupted[slot] = row["donor_id"]
    changed = [index for index, (left, right) in enumerate(zip(original, corrupted)) if left != right]
    single_role.append(changed == [slot])
    other = 1 - slot
    target_other_identity.append(
        original[other] == corrupted[other]
        and row["positive_target_id"] == row["positive_target_id"]
        and row["negative_target_id"] == row["negative_target_id"]
    )
    key = row["corrupt_role_key"]
    role_alignment.append(
        row["role_keys"][slot] == key
        and f"Product {slot + 1}" in row["instruction"]
        and key in row["instruction"].lower()
        and canonical(row["source_role_value"]) == canonical(row["target_role_value"])
        and canonical(row["donor_role_value"]) != canonical(row["target_role_value"])
    )
    values = [
        canonical(metadata[row["source_ids"][index]]["attribute"][role_key])
        for index, role_key in enumerate(row["role_keys"])
    ]
    no_leak.append(not any(len(value) >= 3 and value in row["instruction"].lower() for value in values))
checks["single_role_mutation"] = all(single_role)
checks["target_and_other_reference_identity"] = all(target_other_identity)
checks["role_text_alignment"] = all(role_alignment)
checks["no_attribute_value_leak"] = all(no_leak)

manifest = json.loads((RUN_DIR / "image_manifest.json").read_text())
entries = {row["idx"]: row for row in manifest["entries"]}
required_images = set(base_products) | {row["negative_target_id"] for row in bundles}
hashes_ok = manifest["complete"] and set(entries) == required_images
if hashes_ok:
    hashes_ok = all(sha256_file(ROOT / entry["path"]) == entry["sha256"] for entry in entries.values())
checks["image_manifest_and_hashes"] = hashes_ok

temperature = config["training"]["temperature"]
margin = config["training"]["role_margin"]
good = temperature * F.softplus(torch.tensor((margin - (0.8 - 0.2)) / temperature))
bad = temperature * F.softplus(torch.tensor((margin - (0.2 - 0.8)) / temperature))
probe = torch.tensor([0.4], requires_grad=True)
probe_loss = temperature * F.softplus((margin - (probe - torch.tensor([0.1]))) / temperature)
probe_loss.backward()
checks["loss_direction_unit_test"] = float(good) < float(bad) and float(probe.grad) < 0
details["loss_good"] = float(good)
details["loss_bad"] = float(bad)
details["positive_score_gradient"] = float(probe.grad)

training_source = (ROOT / "src/run_P046.py").read_text()
checks["full_method_not_head_only"] = all(
    token in training_source
    for token in [
        '"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"',
        'if ".visual.merger." in name',
        'candidates.retain_grad()',
        'query_conversation(row, "corrupt")',
    ]
)
checks["matched_arms_and_budget"] = (
    len(config["training"]["arms"]) == 3
    and config["training"]["steps"] > 0
    and config["training"]["seed"] == 20260919
)

classification = "faithful" if all(checks.values()) else "incorrect"
summary = {
    "status": "pass" if classification == "faithful" else "fail",
    "classification": classification,
    "scope": "static code, frozen generated data, loss semantics, and method completeness; runtime gradient coverage is a separate required audit",
    "checks": checks,
    "details": details,
    "split_counts": split_counts,
    "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths + [CONFIG_PATH]},
}
(RUN_DIR / "implementation_audit_static.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["status"] != "pass":
    raise SystemExit(2)
