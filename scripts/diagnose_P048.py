#!/usr/bin/env python3
"""Decompose P048 gains and failures after the frozen gate analysis."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P048_visible_role_contrast"
ARMS = ["ordinary_retrieval", "paraphrase_invariance", "image_role_swap", "text_role_swap"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


assert Path.cwd().resolve() == ROOT
summary = json.loads((RUN_DIR / "pilot_summary.json").read_text())
after = {
    arm: {row["bundle_id"]: row for row in load_jsonl(RUN_DIR / f"eval_after_{arm}.jsonl")}
    for arm in ARMS
}
frozen = {
    row["bundle_id"]: row
    for row in load_jsonl(RUN_DIR / "eval_before_ordinary_retrieval.jsonl")
}
ids = sorted(frozen)
assert all(sorted(rows) == ids for rows in after.values())
image_control = after["image_role_swap"]
text_method = after["text_role_swap"]

def average(rows: dict[str, dict], getter) -> float:
    return sum(float(getter(rows[bundle_id])) for bundle_id in ids) / len(ids)


classes = sorted({text_method[bundle_id]["target_class"] for bundle_id in ids})
by_class = {}
for target_class in classes:
    selected = [bundle_id for bundle_id in ids if text_method[bundle_id]["target_class"] == target_class]
    by_class[target_class] = {
        "rows": len(selected),
        "text_binding_accuracy": sum(text_method[value]["binding_correct"] for value in selected) / len(selected),
        "image_binding_accuracy": sum(image_control[value]["binding_correct"] for value in selected) / len(selected),
        "text_map_at_10": sum(text_method[value]["ap_at_10"] for value in selected) / len(selected),
        "image_map_at_10": sum(image_control[value]["ap_at_10"] for value in selected) / len(selected),
        "text_paraphrase_map_at_10": sum(text_method[value]["paraphrase_ap_at_10"] for value in selected) / len(selected),
        "image_paraphrase_map_at_10": sum(image_control[value]["paraphrase_ap_at_10"] for value in selected) / len(selected),
    }

margin_gains = [text_method[value]["binding_margin"] - image_control[value]["binding_margin"] for value in ids]
reciprocal_rank_gains = [
    1.0 / text_method[value]["rank"] - 1.0 / image_control[value]["rank"]
    for value in ids
]
frozen_summary = json.loads((RUN_DIR / "pilot_ordinary_retrieval.json").read_text())["before"]
geometry = {}
for arm in ARMS:
    current = summary["arms"][arm]
    geometry[arm] = {
        "map_at_10_delta_from_frozen": current["map_at_10"] - frozen_summary["map_at_10"],
        "recall_at_10_delta_from_frozen": current["recall_at_10"] - frozen_summary["recall_at_10"],
        "hard_negative_violation_delta_from_frozen": current["hard_negative_violation_rate"] - frozen_summary["hard_negative_violation_rate"],
    }

diagnostics = {
    "status": "complete",
    "pilot_decision": summary["status"],
    "validation_rows": len(ids),
    "active_role_counts": Counter(text_method[value]["active_role_key"] for value in ids),
    "text_vs_image_control": {
        "binding_flips_gained": sum(text_method[value]["binding_correct"] and not image_control[value]["binding_correct"] for value in ids),
        "binding_flips_lost": sum(image_control[value]["binding_correct"] and not text_method[value]["binding_correct"] for value in ids),
        "rank_improved": sum(text_method[value]["rank"] < image_control[value]["rank"] for value in ids),
        "rank_equal": sum(text_method[value]["rank"] == image_control[value]["rank"] for value in ids),
        "rank_worsened": sum(text_method[value]["rank"] > image_control[value]["rank"] for value in ids),
        "binding_margin_gain_to_reciprocal_rank_gain_correlation": correlation(margin_gains, reciprocal_rank_gains),
    },
    "text_vs_frozen": {
        "rank_improved": sum(text_method[value]["rank"] < frozen[value]["rank"] for value in ids),
        "rank_equal": sum(text_method[value]["rank"] == frozen[value]["rank"] for value in ids),
        "rank_worsened": sum(text_method[value]["rank"] > frozen[value]["rank"] for value in ids),
    },
    "geometry_deltas_from_frozen": geometry,
    "by_class": by_class,
    "mechanistic_readout": {
        "role_signal_supported": (
            summary["paired_bootstrap_comparisons"]["binding_margin"]["ci_lower"] > 0
            and summary["paired_bootstrap_comparisons"]["map_at_10"]["ci_lower"] > 0
        ),
        "current_training_objective_preserves_frozen_geometry": geometry["text_role_swap"]["map_at_10_delta_from_frozen"] >= 0,
        "failure_localization": "The text-controlled role signal is stronger than image swapping, but one-positive/one-hard-negative fine-tuning over-specializes the embedding geometry.",
        "next_falsifier": "Compare role-only training, in-batch multi-negative training, and frozen-reference geometry anchoring before any full experiment.",
    },
}
(RUN_DIR / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
