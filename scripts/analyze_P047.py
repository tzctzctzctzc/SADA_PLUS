#!/usr/bin/env python3
"""Apply frozen P047 gates with query-level paired bootstrap intervals."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P047_position_controlled_role_contrast"
CONFIG = json.loads((ROOT / "configs/P047_position_controlled_role_contrast.json").read_text())


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    method: list[dict],
    control: list[dict],
    getter: Callable[[dict], float],
    resamples: int,
    seed: int,
    ci: float,
) -> dict[str, float]:
    method_by_id = {row["bundle_id"]: row for row in method}
    control_by_id = {row["bundle_id"]: row for row in control}
    ids = sorted(method_by_id)
    assert ids == sorted(control_by_id)
    differences = [float(getter(method_by_id[value])) - float(getter(control_by_id[value])) for value in ids]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        estimates.append(sum(differences[rng.randrange(len(differences))] for _ in differences) / len(differences))
    alpha = (1.0 - ci) / 2.0
    return {
        "difference": observed,
        "ci_lower": percentile(estimates, alpha),
        "ci_upper": percentile(estimates, 1.0 - alpha),
    }


assert Path.cwd().resolve() == ROOT
arms = CONFIG["training"]["arms"]
results = {arm: json.loads((RUN_DIR / f"pilot_{arm}.json").read_text()) for arm in arms}
assert all(row["status"] == "complete" for row in results.values())
assert {row["seed"] for row in results.values()} == {CONFIG["training"]["seed"]}
assert {row["steps"] for row in results.values()} == {CONFIG["training"]["steps"]}
assert len({json.dumps(row["parameter_summary"], sort_keys=True) for row in results.values()}) == 1

eval_rows = {arm: load_jsonl(RUN_DIR / f"eval_after_{arm}.jsonl") for arm in arms}
assert {len(rows) for rows in eval_rows.values()} == {CONFIG["data"]["counts"]["validation"]}
method_name = "text_role_swap"
control_names = [arm for arm in arms if arm != method_name]
metric_specs = {
    "binding_accuracy": ("binding_pair_accuracy", lambda row: float(row["binding_correct"])),
    "binding_margin": ("mean_binding_margin", lambda row: float(row["binding_margin"])),
    "map_at_10": ("map_at_10", lambda row: float(row["ap_at_10"])),
    "recall_at_10": ("recall_at_10", lambda row: float(row["rank"] <= 10)),
    "paraphrase_map_at_10": ("paraphrase_map_at_10", lambda row: float(row["paraphrase_ap_at_10"])),
}
gates_cfg = CONFIG["pilot_gates"]
comparisons = {}
for offset, (name, (summary_key, getter)) in enumerate(metric_specs.items()):
    strongest = max(control_names, key=lambda arm: results[arm]["after"][summary_key])
    comparison = paired_bootstrap(
        eval_rows[method_name],
        eval_rows[strongest],
        getter,
        gates_cfg["bootstrap_resamples"],
        gates_cfg["bootstrap_seed"] + offset,
        gates_cfg["bootstrap_ci"],
    )
    comparison["strongest_control"] = strongest
    comparison["method_value"] = results[method_name]["after"][summary_key]
    comparison["control_value"] = results[strongest]["after"][summary_key]
    comparisons[name] = comparison

gates = {
    "binding_accuracy_gain": (
        comparisons["binding_accuracy"]["difference"] >= gates_cfg["binding_accuracy_gain_over_best_control_min"]
        and comparisons["binding_accuracy"]["ci_lower"] > 0
    ),
    "binding_margin_gain": (
        comparisons["binding_margin"]["difference"] >= gates_cfg["binding_margin_gain_over_best_control_min"]
        and comparisons["binding_margin"]["ci_lower"] > 0
    ),
    "map_at_10_gain": (
        comparisons["map_at_10"]["difference"] >= gates_cfg["map_at_10_gain_over_best_control_min"]
        and comparisons["map_at_10"]["ci_lower"] > 0
    ),
    "recall_at_10_noninferiority": (
        comparisons["recall_at_10"]["difference"] >= -gates_cfg["recall_at_10_noninferiority_tolerance"]
        and comparisons["recall_at_10"]["ci_lower"] >= -gates_cfg["recall_at_10_noninferiority_tolerance"]
    ),
    "paraphrase_map_at_10_noninferiority": (
        comparisons["paraphrase_map_at_10"]["difference"] >= -gates_cfg["paraphrase_map_at_10_noninferiority_tolerance"]
        and comparisons["paraphrase_map_at_10"]["ci_lower"] >= -gates_cfg["paraphrase_map_at_10_noninferiority_tolerance"]
    ),
}
passed = all(gates.values())
summary = {
    "status": "pass_requires_multiseed_confirmation" if passed else "fail_stop_no_full",
    "audit_status": {
        "static": json.loads((RUN_DIR / "implementation_audit_static.json").read_text())["classification"],
        "runtime": json.loads((RUN_DIR / "runtime_gradient_audit.json").read_text())["classification"],
        "corruption_data": json.loads((RUN_DIR / "corruption_audit_summary.json").read_text())["status"],
    },
    "arms": {arm: row["after"] for arm, row in results.items()},
    "paired_bootstrap_comparisons": comparisons,
    "gates": gates,
    "all_gates_pass": passed,
    "decision": (
        "Run a separately frozen multi-seed confirmation; this pilot does not authorize full-scale training."
        if passed
        else "Do not scale. Use the per-query rows to identify whether the failure comes from role sensitivity, gallery ranking, or paraphrase robustness."
    ),
}
(RUN_DIR / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
