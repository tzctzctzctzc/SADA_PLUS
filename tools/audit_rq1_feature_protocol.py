"""Fail closed when an RQ1 feature run violates the frozen transfer protocol."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.rq1_feature_store import SCHEMA, sha256_file
from tools.run_clip_target_adapter import LABELS, read_jsonl


def parse_named_prefix(value):
    if "=" not in value:
        raise ValueError("feature must be NAME=PREFIX")
    name, prefix = value.split("=", 1)
    if not name or not prefix:
        raise ValueError("feature must be NAME=PREFIX")
    return name, Path(prefix)


def membership_map(rows, shift):
    roles = defaultdict(set)
    classes = defaultdict(Counter)
    for row in rows:
        for membership in row["memberships"]:
            if membership.get("shift") != shift:
                continue
            role = membership.get("role")
            roles[role].add(row["video_id"])
            classes[role][int(row["gt_label_id"])] += 1
    return roles, classes


def audit_splits(rows):
    findings = {}
    valid = True
    expected_roles = ("source_train", "source_val", "target_val")
    for shift in SHIFT_CONFIGS:
        roles, classes = membership_map(rows, shift)
        overlaps = {}
        for left_index, left in enumerate(expected_roles):
            for right in expected_roles[left_index + 1:]:
                overlap = sorted(roles[left].intersection(roles[right]))
                overlaps["{}_vs_{}".format(left, right)] = overlap
                valid = valid and not overlap
        missing_classes = {
            role: [
                LABELS[class_id]
                for class_id in range(len(LABELS))
                if classes[role][class_id] == 0
            ]
            for role in expected_roles
        }
        valid = valid and all(not values for values in missing_classes.values())
        findings[shift] = {
            "video_counts": {role: len(roles[role]) for role in expected_roles},
            "segment_counts": {
                role: int(sum(classes[role].values())) for role in expected_roles
            },
            "video_overlaps": overlaps,
            "missing_classes": missing_classes,
        }
    return valid, findings


def audit_feature_store(name, prefix, manifest, rows, expected_frame_count):
    meta_path = Path(str(prefix) + ".meta.json")
    feature_path = Path(str(prefix) + ".features.npy")
    status_path = Path(str(prefix) + ".status.npy")
    missing = [
        str(path) for path in (meta_path, feature_path, status_path)
        if not path.is_file()
    ]
    if missing:
        return False, {"name": name, "missing_files": missing}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    features = np.load(feature_path, mmap_mode="r")
    status = np.load(status_path, mmap_mode="r")
    finite_rows = np.isfinite(features).all(axis=1)
    norms = np.linalg.norm(np.asarray(features, dtype=np.float32), axis=1)
    complete = bool(
        meta.get("schema") == SCHEMA
        and meta.get("manifest_sha256") == sha256_file(manifest)
        and len(features) == len(rows)
        and len(status) == len(rows)
        and np.all(status == 1)
        and np.all(finite_rows)
        and np.all(norms > 1e-8)
    )
    declared_frames = int(meta.get("frame_count", -1))
    dense_anchor = declared_frames == 0
    frame_count_valid = dense_anchor or declared_frames == expected_frame_count
    valid = complete and frame_count_valid
    return valid, {
        "name": name,
        "model": meta.get("model"),
        "pooling": meta.get("pooling"),
        "rows": len(features),
        "dimensions": int(features.shape[1]) if features.ndim == 2 else None,
        "declared_frame_count": declared_frames,
        "dense_anchor": dense_anchor,
        "status_complete": bool(np.all(status == 1)),
        "finite_rows": int(finite_rows.sum()),
        "nonzero_rows": int((norms > 1e-8).sum()),
        "norm_min": float(norms.min()) if len(norms) else None,
        "norm_max": float(norms.max()) if len(norms) else None,
        "valid": valid,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--expected-frame-count", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_frame_count <= 0:
        raise ValueError("expected_frame_count must be positive")
    rows = read_jsonl(args.manifest)
    split_valid, split_findings = audit_splits(rows)
    feature_findings = []
    feature_valid = True
    for specification in args.feature:
        name, prefix = parse_named_prefix(specification)
        valid, finding = audit_feature_store(
            name,
            prefix,
            args.manifest,
            rows,
            args.expected_frame_count,
        )
        feature_valid = feature_valid and valid
        feature_findings.append(finding)
    report = {
        "protocol": "rq1-domain-invariant-semantics-audit-v1",
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "rows": len(rows),
        "expected_frame_count": args.expected_frame_count,
        "split_valid": split_valid,
        "splits": split_findings,
        "feature_stores_valid": feature_valid,
        "feature_stores": feature_findings,
        "valid": bool(split_valid and feature_valid),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
