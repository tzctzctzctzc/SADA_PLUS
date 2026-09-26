#!/usr/bin/env python3
"""冻结的 v3 标注扩量门槛；不通过时不生成训练可用清单。"""
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def grouped(path, cfg):
    groups = defaultdict(list)
    for row in map(json.loads, path.open()):
        assert row["screening_config"] == cfg and row["dataset_sha256"] == cfg["dataset_sha256"]
        groups[row["bundle_id"]].append(row)
    for rows in groups.values():
        assert len(rows) == 4 and {(r["kind"], r["mirrored"]) for r in rows} == {(kind, mirror) for kind in ["positive", "negative"] for mirror in [0, 1]}
    return groups


def main():
    run = Path("runs/P051_tracebind")
    cfg = json.loads(Path("configs/P051_annotation_v3.json").read_text())
    manual = json.loads(Path("results/P051_tracebind/stratified_visual_inspection_raw.json").read_text())
    cal, inspection, models = {}, {}, {}
    for model in cfg["models"]:
        cal[model] = grouped(run / f"visual_v3_{model}_96_mirror.jsonl", cfg)
        inspection[model] = grouped(run / f"visual_v3_{model}_inspection64.jsonl", cfg)
        assert len(cal[model]) == 96 and len(inspection[model]) == 64
        for suffix in ["96_mirror", "inspection64"]:
            runtime = json.loads((run / f"visual_v3_{model}_{suffix}.runtime.json").read_text())
            assert runtime["status"] == "pass" and runtime["screening_config"] == cfg
        passed = {bid for bid, rows in cal[model].items() if all(r["pass"] for r in rows)}
        stable, same_letter = Counter(), Counter()
        for rows in cal[model].values():
            for kind in ["positive", "negative"]:
                pair = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["mirrored"])
                decisions = [r["product_ids"][1 if r["prediction"] == "A" else 2] if r["prediction"] in "AB" else r["prediction"] for r in pair]
                stable[pair[0]["family"]] += int(decisions[0] == decisions[1])
                same_letter[pair[0]["family"]] += int(pair[0]["prediction"] == pair[1]["prediction"] and pair[0]["prediction"] in "AB")
        models[model] = {"passed_groups": len(passed), "passed_families": dict(Counter(cal[model][bid][0]["family"] for bid in passed)),
                         "all_four_predictions_agree_with_metadata_ignoring_confidence": sum(all(r["prediction"] == r["expected"] for r in rows) for rows in cal[model].values()),
                         "option_prediction_counts": dict(Counter(r["prediction"] for rows in cal[model].values() for r in rows)),
                         "position_consistent_comparisons_per_family_out_of_48": dict(stable),
                         "same_AB_letter_after_references_swapped": dict(same_letter),
                         "median_total_option_mass": statistics.median(r["total_option_probability_mass"] for rows in cal[model].values() for r in rows)}
        inspection_passed = {bid for bid, rows in inspection[model].items() if all(r["pass"] for r in rows)}
        models[model]["inspection_passed_group_assessments"] = dict(Counter(r["assessment"] for r in manual["groups"] if r["bundle_id"] in inspection_passed))
        models[model]["known_inspection_contradictions_accepted"] = [r["bundle_id"] for r in manual["groups"] if r["assessment"] == "contradiction" and r["bundle_id"] in inspection_passed]
    assert cal["primary"].keys() == cal["secondary"].keys() and inspection["primary"].keys() == inspection["secondary"].keys()
    passed = {bid for bid in cal["primary"] if all(r["pass"] for model in cal for r in cal[model][bid])}
    families = Counter(cal["primary"][bid][0]["family"] for bid in passed)
    contra = [r["bundle_id"] for r in manual["groups"] if r["assessment"] == "contradiction"]
    false_accepts = [bid for bid in contra if all(r["pass"] for model in inspection for r in inspection[model][bid])]
    def physical(row):
        return row["product_ids"][1 if row["prediction"] == "A" else 2] if row["prediction"] in "AB" else row["prediction"]
    comparisons = [(bid, i) for bid in cal["primary"] for i in range(4)]
    agreement = sum(physical(cal["primary"][bid][i]) == physical(cal["secondary"][bid][i]) for bid, i in comparisons) / len(comparisons)
    checks = {"at_least_four_joint_pass_groups_per_family": all(families[family] >= cfg["minimum_joint_pass_groups_per_family_before_scaling"] for family in ["color", "pattern", "shape", "construction"]),
              "known_assistant_visual_contradictions_rejected": not false_accepts}
    report = {"status": "annotation_quality_pilot_pass_can_scale_annotation_only" if all(checks.values()) else "annotation_quality_pilot_failed_do_not_scale",
              "checks": checks, "models": models, "joint_pass_groups": len(passed), "joint_pass_families": dict(families),
              "cross_model_per_check_physical_choice_agreement": agreement,
              "inspection_contradictions_checked": len(contra), "joint_false_accepts_of_known_contradictions": false_accepts,
              "quality_pilot_and_inspection_overlap": len(cal["primary"].keys() & inspection["primary"].keys()),
              "visual_inspection_is_assistant_assessment_not_independent_expert_truth": True,
              "thresholds_relaxed": False, "training_pilot_ready": False,
              "missing_final_verified_counts": {"train": 5000, "validation": 1000, "confirmation": 400},
              "absolute_all_role_ground_truth_certified": False, "is_efficacy_evidence": False,
              "dataset_sha256": cfg["dataset_sha256"]}
    (run / "visual_v3_quality_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
