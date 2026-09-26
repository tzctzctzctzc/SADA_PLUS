#!/usr/bin/env python3
"""汇总可复核证据，明确区分计算正确、数据完整与方法完整。"""
import hashlib
import json
from pathlib import Path


def main():
    run = Path("runs/P051_tracebind")
    names = ["math_audit", "end_to_end_runtime_audit", "orbits_prepare_summary", "orbits_integrity_audit",
             "correspondence_prepare_summary", "correspondence_integrity_audit", "image_cache_provenance_audit",
             "visual_calibration_audit", "visual_v3_quality_audit", "embedding_model_integrity_audit"]
    evidence = {name: json.loads((run / (name + ".json")).read_text()) for name in names}
    math, runtime = evidence["math_audit"], evidence["end_to_end_runtime_audit"]
    assert all(math["checks"].values()) and all(runtime["checks"].values())
    assert math["source_sha256"] == hashlib.sha256(Path("src/tracebind.py").read_bytes()).hexdigest()
    for path, digest in runtime["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
    assert evidence["image_cache_provenance_audit"]["status"] == evidence["correspondence_integrity_audit"]["status"] == "pass"
    assert evidence["embedding_model_integrity_audit"]["status"] == "pass"
    metadata = json.loads((run / "metadata_manifest.json").read_text())
    images = json.loads((run / "orbits_image_manifest.json").read_text())
    assert metadata["complete"] and images["complete"]
    report = {
        "classification": "core_computational_paths_correct_complete_training_method_not_ready",
        "audit_scope": "P051 TraceBind model, objective, encoder, checkpoint, construction/extraction/isolation/correspondence/visual-screening pipeline; historical methods are evidence, not reimplemented here",
        "math_checks_passed": len(math["checks"]), "live_runtime_checks_passed": len(runtime["checks"]),
        "fixed_issues": [
            "valid-assignment likelihood handles multiple positives and masks unknown permutations",
            "DoD uses the exact same per-permutation positive-minus-negative ordering as construction audit",
            "correspondence loss supports multiple correct cells; unverified neighboring cells are not false negatives",
            "teacher loss temperature matches ranking-score gradient scale; teacher encoded before adaptation and detached",
            "FP32 trainable visual merger with BF16 autocast prevents non-updating low-precision parameters",
            "query roles compile from original text; label-free inference score matches labelled audit path",
            "fresh-model checkpoint and optimizer restore validated",
            "cached image identity requires matching pinned product-to-byte manifests",
            "native model image token counts and actual generated first answer token checked before annotation"
        ],
        "blocking_findings": [
            {"priority": 0, "issue": "Matched multi-arm training driver and official full-gallery metric integration are missing", "effect": "A two-step runtime audit cannot certify faithful complete training or SOTA."},
            {"priority": 0, "issue": "Final visually verified 5000/1000/400 role-balanced orbit dataset is not certified", "effect": "Metadata-derived candidates and cross-domain pseudo correspondences cannot bypass the visual gate."},
            {"priority": 1, "issue": "Synthetic coarse-attribute queries can have unlabelled semantic positives", "effect": "Single-product-ID mAP over their full gallery can count correct targets as negatives; requires multi-positive relevance or separate official query evaluation."},
            {"priority": 1, "issue": "One distinct role per source and restricted same-attribute grammar", "effect": "Cannot claim unrestricted PinPoint query support; official instruction parser and coverage audit remain necessary."},
            {"priority": 1, "issue": "The permutation-augmentation control currently has equivalent query views, not genuine wrong-binding augmentation", "effect": "It is not yet the promised strong matched corruption-augmentation baseline."},
            {"priority": 1, "issue": "Active-role relative validation does not verify all roles or full-energy DoD assumptions", "effect": "Other-role error cancellation remains a weak metadata assumption; must retain that uncertainty in claims and supervision."},
            {"priority": 1, "issue": "Every non-identity bijective permutation changes at least two source-role bindings", "effect": "Permutation DoD is not a controlled single-role intervention. Checking only the active cue cannot certify the complete permutation ordering; isolate the intervention or verify the other changed roles before making that claim."},
            {"priority": 1, "issue": "Exact pixel isolation does not exclude all near-duplicate product photos", "effect": "Near-duplicate review still required before an independent generalization claim."}
        ],
        "theory_limits": ["R2 assignment posterior equals log-sigmoid of identity-minus-swap energy; the formula alone is not innovation.",
                          "Frozen anchors constrain geometry locally; they do not prove global recall is preserved.",
                          "Cross-domain correspondence agreement is pseudo supervision, not retrieval role ground truth."],
        "data": {"metadata_products": metadata["product_count"], "images_extracted": images["extracted"],
                 "raw_groups": evidence["orbits_prepare_summary"]["groups"],
                 "isolated_split_groups": evidence["orbits_integrity_audit"]["split_groups"],
                 "correspondence_pairs": evidence["correspondence_integrity_audit"]["retained_pairs"],
                 "correspondence_points": evidence["correspondence_prepare_summary"]["total_verified_points"],
                 "quality_status": evidence["visual_v3_quality_audit"]["status"]},
        "training_pilot_ready": False, "full_training_allowed": False, "sota_evidence_exists": False,
        "evidence_files": {name: {"path": str(run / (name + ".json")), "sha256": hashlib.sha256((run / (name + ".json")).read_bytes()).hexdigest()} for name in names},
        "source_hashes": runtime["source_hashes"]
    }
    (run / "comprehensive_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
