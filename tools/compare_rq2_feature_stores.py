"""Verify numerical parity between two RQ2 time-aligned feature stores."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.run_clip_target_adapter import read_jsonl


def load_index(path):
    return {
        row["candidate_id"]: row
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-index", action="append", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--candidate-index", action="append", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-cosine", type=float, default=0.9999)
    return parser.parse_args()


def merge_indices(paths):
    merged = {}
    for path in paths:
        for candidate_id, row in load_index(path).items():
            if candidate_id in merged:
                raise ValueError("duplicate candidate: {}".format(candidate_id))
            merged[candidate_id] = row
    return merged


def main():
    args = parse_args()
    reference = merge_indices(args.reference_index)
    candidate = merge_indices(args.candidate_index)
    ids = sorted(set(reference).intersection(candidate))
    if not ids:
        raise RuntimeError("feature stores do not overlap")
    cosines = []
    maximum_absolute_error = 0.0
    for candidate_id in ids:
        ref = np.load(Path(args.reference_root) / reference[candidate_id]["feature_file"])
        new = np.load(Path(args.candidate_root) / candidate[candidate_id]["feature_file"])
        ref_features = ref["features"].astype(np.float32)
        new_features = new["features"].astype(np.float32)
        if ref_features.shape != new_features.shape:
            raise ValueError("feature shape mismatch: {}".format(candidate_id))
        if not np.array_equal(ref["timestamps_sec"], new["timestamps_sec"]):
            raise ValueError("timestamp mismatch: {}".format(candidate_id))
        numerator = np.sum(ref_features * new_features, axis=1)
        denominator = np.linalg.norm(ref_features, axis=1) * np.linalg.norm(
            new_features, axis=1
        )
        cosines.extend((numerator / np.maximum(denominator, 1e-12)).tolist())
        maximum_absolute_error = max(
            maximum_absolute_error,
            float(np.max(np.abs(ref_features - new_features))),
        )
    payload = {
        "status": "pass" if min(cosines) >= args.minimum_cosine else "fail",
        "windows": len(ids),
        "frames": len(cosines),
        "minimum_cosine": float(min(cosines)),
        "mean_cosine": float(np.mean(cosines)),
        "maximum_absolute_error": maximum_absolute_error,
        "required_minimum_cosine": args.minimum_cosine,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
