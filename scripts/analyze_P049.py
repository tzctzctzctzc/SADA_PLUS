#!/usr/bin/env python3
"""Apply the preregistered P049 paired-bootstrap gates and localize failure."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Callable

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P049_counterfactual_role_verifier"
CONFIG = json.loads((ROOT / "configs/P049_counterfactual_role_verifier.json").read_text())
GATES = CONFIG["pilot_gates"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def ap10(row: dict) -> float:
    return 1.0 / row["rank"] if row["rank"] <= 10 else 0.0


def para_ap10(row: dict) -> float:
    return 1.0 / row["paraphrase_rank"] if row["paraphrase_rank"] <= 10 else 0.0


def paired_bootstrap(left: list[dict], right: list[dict], metric: Callable[[dict], float], seed_offset: int) -> dict:
    assert [row["bundle_id"] for row in left] == [row["bundle_id"] for row in right]
    observed = mean(metric(a) - metric(b) for a, b in zip(left, right))
    rng = random.Random(GATES["bootstrap_seed"] + seed_offset)
    values = []
    for _ in range(GATES["bootstrap_resamples"]):
        indices = [rng.randrange(len(left)) for _ in left]
        values.append(mean(metric(left[index]) - metric(right[index]) for index in indices))
    values.sort()
    alpha = (1.0 - GATES["bootstrap_ci"]) / 2.0
    lower = values[int(alpha * len(values))]
    upper = values[min(len(values) - 1, int((1.0 - alpha) * len(values)))]
    return {"difference": observed, "ci_lower": lower, "ci_upper": upper}


assert Path.cwd().resolve() == ROOT
base_cache = load_jsonl(RUN_DIR / "base_validation.jsonl")
base_rows = []
for row in base_cache:
    base_rows.append({
        "bundle_id": row["bundle_id"], "rank": row["base_rank"], "paraphrase_rank": row["base_paraphrase_rank"],
        "dod_margin": 0.0, "hard_negative_violation": row["base_negative_score"] >= row["base_positive_score"],
    })
arms = {arm: load_jsonl(RUN_DIR / f"eval_after_{arm}.jsonl") for arm in CONFIG["training"]["arms"]}
method = arms["dod_counterfactual"]
ordinary = arms["ordinary_verifier"]
comparisons = {
    "map_vs_frozen": paired_bootstrap(method, base_rows, ap10, 1),
    "map_vs_ordinary": paired_bootstrap(method, ordinary, ap10, 2),
    "paraphrase_map_vs_frozen": paired_bootstrap(method, base_rows, para_ap10, 3),
    "dod_margin_vs_ordinary": paired_bootstrap(method, ordinary, lambda row: row["dod_margin"], 4),
    "hard_negative_violation_vs_frozen": paired_bootstrap(method, base_rows, lambda row: float(row["hard_negative_violation"]), 5),
}
gates = {
    "map_gain_vs_frozen": comparisons["map_vs_frozen"]["ci_lower"] > GATES["map_at_10_gain_vs_frozen_ci_lower_min"],
    "map_gain_vs_ordinary": comparisons["map_vs_ordinary"]["ci_lower"] > GATES["map_at_10_gain_vs_ordinary_ci_lower_min"],
    "paraphrase_map_noninferiority": comparisons["paraphrase_map_vs_frozen"]["ci_lower"] >= -GATES["paraphrase_map_noninferiority_tolerance"],
    "dod_margin_gain_vs_ordinary": comparisons["dod_margin_vs_ordinary"]["ci_lower"] > GATES["dod_margin_gain_vs_ordinary_ci_lower_min"],
    "hard_negative_violation_reduction": comparisons["hard_negative_violation_vs_frozen"]["ci_upper"] < GATES["hard_negative_violation_reduction_ci_upper_max"],
}
metrics = {"frozen": json.loads((RUN_DIR / "base_validation_summary.json").read_text())["metrics"]}
for arm in CONFIG["training"]["arms"]:
    metrics[arm] = json.loads((RUN_DIR / f"pilot_{arm}.json").read_text())["metrics"]
diagnostics = {
    "method_vs_frozen_rank_improved": sum(a["rank"] < b["rank"] for a, b in zip(method, base_rows)),
    "method_vs_frozen_rank_equal": sum(a["rank"] == b["rank"] for a, b in zip(method, base_rows)),
    "method_vs_frozen_rank_worsened": sum(a["rank"] > b["rank"] for a, b in zip(method, base_rows)),
    "delta_beats_raw_within_method": sum(row["delta_rerank_rank"] < row["raw_rerank_rank"] for row in method),
    "delta_loses_to_raw_within_method": sum(row["delta_rerank_rank"] > row["raw_rerank_rank"] for row in method),
}
passed = all(gates.values())
summary = {
    "status": "pass_authorize_multiseed_confirmation" if passed else "fail_stop_no_full",
    "audit_status": {"static": json.loads((RUN_DIR / "implementation_audit_static.json").read_text())["classification"], "runtime": json.loads((RUN_DIR / "runtime_gradient_audit.json").read_text())["classification"], "inherited_visual_data": json.loads((RUN_DIR / "prepare_summary.json").read_text())["inherited_visual_gate"]["status"]},
    "metrics": metrics,
    "paired_bootstrap_comparisons": comparisons,
    "gates": gates,
    "all_gates_pass": passed,
    "diagnostics": diagnostics,
    "decision": "Proceed only to multi-seed and broader-role confirmation." if passed else "Do not scale. Localize whether failure is verifier relevance, counterfactual candidate discrimination, or paraphrase transfer.",
}
(RUN_DIR / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
if not passed:
    raise SystemExit(3)
