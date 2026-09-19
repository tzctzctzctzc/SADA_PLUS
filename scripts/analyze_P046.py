#!/usr/bin/env python3
"""Apply the frozen P046 pilot gates to the three completed arms."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/vepfs/vlms/1")
RUN_DIR = ROOT / "runs/P046_self_normalized_role_corruption"
CONFIG = json.loads((ROOT / "configs/P046_self_normalized_role_corruption.json").read_text())

assert Path.cwd().resolve() == ROOT
results = {
    arm: json.loads((RUN_DIR / f"pilot_{arm}.json").read_text())
    for arm in CONFIG["training"]["arms"]
}
assert all(row["status"] == "complete" for row in results.values())
assert {row["seed"] for row in results.values()} == {CONFIG["training"]["seed"]}
assert {row["steps"] for row in results.values()} == {CONFIG["training"]["steps"]}
assert len({json.dumps(row["before"], sort_keys=True) for row in results.values()}) == 1
assert len({json.dumps(row["parameter_summary"], sort_keys=True) for row in results.values()}) == 1

role = results["role_corruption"]["after"]
controls = [results["ordinary_retrieval"]["after"], results["generic_augmentation"]["after"]]
best_control_accuracy = max(row["role_pair_accuracy"] for row in controls)
best_control_margin = max(row["mean_role_margin"] for row in controls)
best_control_retrieval = max(row["retrieval_accuracy"] for row in controls)
gates_cfg = CONFIG["pilot_gates"]
gains = {
    "role_pair_accuracy": role["role_pair_accuracy"] - best_control_accuracy,
    "mean_role_margin": role["mean_role_margin"] - best_control_margin,
    "retrieval_accuracy_vs_best_control": role["retrieval_accuracy"] - best_control_retrieval,
}
gates = {
    "role_pair_accuracy_gain": gains["role_pair_accuracy"] >= gates_cfg["role_pair_accuracy_gain_over_best_control_min"],
    "role_margin_gain": gains["mean_role_margin"] >= gates_cfg["role_margin_gain_over_best_control_min"],
    "retrieval_noninferiority": gains["retrieval_accuracy_vs_best_control"] >= -gates_cfg["retrieval_accuracy_noninferiority_tolerance"],
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
    "gains_over_best_control": gains,
    "gates": gates,
    "all_gates_pass": passed,
    "decision": (
        "Freeze this configuration and run a separately frozen multi-seed confirmation before any full experiment."
        if passed
        else "Do not scale. The margin signal is encouraging, but the role arm did not improve held-out pair accuracy over generic augmentation."
    ),
}
(RUN_DIR / "pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
