#!/usr/bin/env python3
"""Static semantic audit for P049 before any verifier training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P049_counterfactual_role_verifier"
CONFIG_PATH = ROOT / "configs/P049_counterfactual_role_verifier.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def margin_loss(positive: torch.Tensor, negative: torch.Tensor, margin: float, temperature: float) -> torch.Tensor:
    return temperature * F.softplus((margin - (positive - negative)) / temperature)


assert Path.cwd().resolve() == ROOT
config = json.loads(CONFIG_PATH.read_text())
bundles = load_jsonl(RUN_DIR / "bundles.jsonl")
p048 = load_jsonl(ROOT / config["data"]["p048_bundles"])
checks = {}
details = {}
checks["frozen_config_and_hashes"] = (
    sha256_file(ROOT / config["data"]["query_path"]) == config["data"]["query_sha256"]
    and sha256_file(ROOT / config["data"]["p048_bundles"]) == config["data"]["p048_bundles_sha256"]
)
checks["split_counts"] = Counter(row["split"] for row in bundles) == Counter(config["data"]["counts"])

def heldout_signature(rows: list[dict]) -> list[tuple]:
    return sorted((
        row["bundle_id"], row["split"], tuple(row["source_ids"]), row["positive_target_id"], row["negative_target_id"],
        row["instructions"]["original"], row["instructions"]["paraphrase"], row["instructions"]["text_role_swap"],
    ) for row in rows if row["split"] in {"validation", "confirmation"})

checks["heldout_identical_to_P048"] = heldout_signature(bundles) == heldout_signature(p048)
train = [row for row in bundles if row["split"] == "train"]
heldout = [row for row in bundles if row["split"] != "train"]
train_products = [product_id for row in train for product_id in row["product_ids"]]
heldout_products = [product_id for row in heldout for product_id in row["product_ids"]]
checks["train_heldout_product_disjointness"] = not (set(train_products) & set(heldout_products)) and len(train_products) == len(set(train_products)) and len(heldout_products) == len(set(heldout_products))
checks["same_images_same_order_for_text_swap"] = all(
    row["instructions"]["original"] != row["instructions"]["text_role_swap"]
    and row["instructions"]["original"].count("Product 1") == row["instructions"]["original"].count("Product 2") == 1
    and row["instructions"]["text_role_swap"].count("Product 1") == row["instructions"]["text_role_swap"].count("Product 2") == 1
    for row in bundles
)
checks["paraphrase_counterfactual_present"] = all("paraphrase_text_role_swap" in row["instructions"] for row in bundles)
visual = json.loads((ROOT / config["data"]["p048_visual_audit"]).read_text())
checks["inherited_visual_gate_passing"] = visual["status"] == "pass" and visual["bundles"] == config["data"]["counts"]["confirmation"]

temperature = config["training"]["temperature"]
margin = config["training"]["dod_margin"]
positive_original = torch.tensor(0.8, requires_grad=True)
positive_swap = torch.tensor(0.2, requires_grad=True)
negative_original = torch.tensor(0.4, requires_grad=True)
negative_swap = torch.tensor(0.3, requires_grad=True)
positive_delta = positive_original - positive_swap
negative_delta = negative_original - negative_swap
good = margin_loss(positive_delta, negative_delta, margin, temperature)
bad = margin_loss(torch.tensor(0.1), torch.tensor(0.6), margin, temperature)
good.backward()
checks["difference_in_differences_loss_direction"] = (
    float(good) < float(bad)
    and float(positive_original.grad) < 0
    and float(positive_swap.grad) > 0
    and float(negative_original.grad) > 0
    and float(negative_swap.grad) < 0
)
details["dod_unit_test"] = {"good": float(good), "bad": float(bad), "gradients": [float(positive_original.grad), float(positive_swap.grad), float(negative_original.grad), float(negative_swap.grad)]}

source_path = ROOT / "src/run_P049.py"
source = source_path.read_text()
checks["candidate_joint_cross_encoder_input"] = all(token in source for token in ["def joint_conversation", "Reference Product", "Candidate product", "candidate_id"])
checks["base_retriever_has_no_optimizer_path"] = source.index("if args.prepare_base:") < source.index("get_peft_model(core, lora)") < source.index("optimizer = torch.optim.AdamW")
checks["full_counterfactual_method_present"] = all(token in source for token in ["positive_delta = original[0] - swapped[0]", "negative_delta = original[1] - swapped[1]", "dod_loss = pairwise_margin_loss", "para_loss = F.smooth_l1_loss", "zscore(base_scores)"])
checks["matched_arm_initialization_and_budget"] = config["training"]["arms"] == ["ordinary_verifier", "positive_counterfactual", "dod_counterfactual"] and config["training"]["epochs"] == 3 and config["training"]["batch_size"] == 2

critical = [
    "frozen_config_and_hashes", "split_counts", "heldout_identical_to_P048", "train_heldout_product_disjointness",
    "same_images_same_order_for_text_swap", "paraphrase_counterfactual_present", "inherited_visual_gate_passing",
    "difference_in_differences_loss_direction", "candidate_joint_cross_encoder_input", "base_retriever_has_no_optimizer_path",
    "full_counterfactual_method_present", "matched_arm_initialization_and_budget",
]
if all(checks[key] for key in critical):
    classification = "faithful"
elif not checks["difference_in_differences_loss_direction"] or not checks["train_heldout_product_disjointness"]:
    classification = "incorrect"
elif not checks["full_counterfactual_method_present"] or not checks["candidate_joint_cross_encoder_input"]:
    classification = "minimal"
else:
    classification = "conservative"

paths = [CONFIG_PATH, ROOT / "scripts/prepare_P049.py", ROOT / "scripts/extract_P049_images.py", source_path, ROOT / "scripts/analyze_P049.py", ROOT / "scripts/run_P049.sh"]
summary = {
    "status": "pass" if classification == "faithful" else "fail",
    "classification": classification,
    "scope": "frozen data, counterfactual semantics, loss direction, candidate-aware input, frozen-base isolation, and method completeness; runtime gradient coverage is a separate required gate",
    "checks": checks,
    "details": details,
    "split_counts": Counter(row["split"] for row in bundles),
    "unique_products": len(set(train_products + heldout_products)),
    "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in paths},
}
(RUN_DIR / "implementation_audit_static.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if summary["status"] != "pass":
    raise SystemExit(2)
