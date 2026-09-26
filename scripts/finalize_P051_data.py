#!/usr/bin/env python3
"""冻结数据供应；数量、角色覆盖不足时明确失败，不降低门槛。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    cfg, run = config["data"], Path("runs/P051_tracebind")
    path = run / "orbits_byte_disjoint.jsonl"
    digest = sha(path)
    integrity = json.loads((run / "orbits_integrity_audit.json").read_text())
    assert integrity["dataset_sha256"] == digest and integrity["all_labels_recomputed"]
    decisions = {}
    for model in ["large", "small"]:
        groups = defaultdict(list)
        for row in map(json.loads, (run / f"visual_{model}.jsonl").open()):
            assert row["dataset_sha256"] == digest and row["screening_config"] == cfg["visual_screening"]
            assert row["revision"] == cfg["visual_screening"]["models"][model]["revision"]
            groups[row["bundle_id"]].append(row)
        decisions[model] = {}
        for bid, rows in groups.items():
            assert len(rows) == 4 and {(r["kind"], r["mirrored"]) for r in rows} == {(kind, mirror) for kind in ["positive", "negative"] for mirror in [0, 1]}
            assert len({r["corruption_assignment"] for r in rows}) == 1
            if all(r["pass"] for r in rows):
                decisions[model][bid] = rows[0]["corruption_assignment"]
    pool = defaultdict(list)
    for row in map(json.loads, path.open()):
        bid = row["bundle_id"]
        if bid not in decisions["large"] or bid not in decisions["small"]:
            continue
        assert decisions["large"][bid] == decisions["small"][bid]
        verified = decisions["large"][bid]
        assert verified in row["verified_dod_assignments"]
        row["metadata_dod_assignments_before_visual_gate"] = row["verified_dod_assignments"]
        row["verified_dod_assignments"] = [verified]
        # 未逐图验证的其余错误排列是未知项，不能因为元数据就全部处罚。
        row["metadata_supervised_assignments_before_visual_gate"] = row["supervised_assignments"]
        row["supervised_assignments"] = [valid or i == verified for i, valid in enumerate(row["valid_assignments"])]
        row["supervision_grade"] = "relative_active_role_dual_judge_pass_other_roles_metadata_weak"
        row["dod_supervision_grade"] = "active_role_relative_verified_other_role_cancellation_is_weak_metadata_assumption"
        row["absolute_all_role_correctness_certified"] = False
        pool[row["split"]].append(row)
    checks, splits, selected = {}, {}, []
    families = list(cfg["visual_families"])
    for split, required in cfg["construction"]["final_counts"].items():
        ordered = sorted(pool[split], key=lambda r: hashlib.sha256(r["bundle_id"].encode()).digest())
        buckets = {family: [r for r in ordered if r["active_family"] == family] for family in families}
        # 保持预先冻结的四族均衡，不靠结果把容易的颜色补成主体。
        counts = {family: required // len(families) + int(i < required % len(families)) for i, family in enumerate(families)}
        take = [row for family in families for row in buckets[family][:counts[family]]]
        family_counts = Counter(r["active_family"] for r in take)
        role_counts = Counter(r["active_role_key"] for r in take)
        checks[split + "_count"] = len(take) == required
        checks[split + "_four_families"] = len(family_counts) == cfg["minimum_visual_families"]
        checks[split + "_role_cap"] = max(role_counts.values(), default=0) <= required * cfg["maximum_active_role_fraction"]
        checks[split + "_three_references"] = sum(len(r["source_ids"]) == 3 for r in take) >= required * .20
        splits[split] = {"available": len(ordered), "available_families": dict(Counter(r["active_family"] for r in ordered)),
                         "required": required, "selected": len(take), "selected_families": dict(family_counts),
                         "source_counts": dict(Counter(len(r["source_ids"]) for r in take))}
        selected.extend(take)
    c = json.loads((run / "correspondence_integrity_audit.json").read_text())
    checks["correspondence_supply_and_integrity"] = c["status"] == "pass"
    checks["pixels_disjoint"] = integrity["train_heldout_exact_pixel_overlap"] == integrity["heldout_between_group_exact_pixel_overlap"] == 0
    output = run / ("pilot_orbits.jsonl" if all(checks.values()) else "pilot_orbits_incomplete.jsonl")
    with output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {"status": "relative_data_supply_pass_manual_inspection_and_driver_pending" if all(checks.values()) else "insufficient_verified_supply",
              "checks": checks, "splits": splits, "dataset_path": str(output), "dataset_sha256": sha(output),
              "source_dataset_sha256": digest, "same_family_judges_not_independent_ground_truth": True,
              "absolute_all_role_correctness_certified": False, "pilot_authorized": False,
              "remaining_gates": ["stratified image inspection", "full matched training driver and gallery metrics"]}
    (run / "data_supply_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
