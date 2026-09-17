"""Merge disjoint RQ1 feature shards into one audited feature store."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.rq1_feature_store import FeatureStore, SCHEMA, sha256_file
from tools.run_clip_target_adapter import read_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-prefix", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    rows = read_jsonl(args.manifest)
    expected_hash = sha256_file(args.manifest)
    shard_data = []
    reference = None
    coverage = np.zeros(len(rows), dtype=np.uint8)
    for prefix_value in args.shard_prefix:
        prefix = Path(prefix_value)
        meta = json.loads(Path(str(prefix) + ".meta.json").read_text("utf-8"))
        if meta.get("schema") != SCHEMA or meta.get("manifest_sha256") != expected_hash:
            raise ValueError("invalid shard metadata: {}".format(prefix))
        comparable = {
            key: meta[key]
            for key in ("schema", "manifest_sha256", "model", "dimensions",
                        "frame_count", "pooling", "rows")
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError("feature shard metadata mismatch: {}".format(prefix))
        features = np.load(str(prefix) + ".features.npy", mmap_mode="r")
        status = np.load(str(prefix) + ".status.npy", mmap_mode="r")
        if features.shape != (len(rows), int(meta["dimensions"])) or len(status) != len(rows):
            raise ValueError("feature shard shape mismatch: {}".format(prefix))
        active = np.asarray(status) == 1
        if np.any(coverage[active]):
            raise ValueError("feature shards overlap: {}".format(prefix))
        coverage[active] = 1
        shard_data.append((prefix, features, np.flatnonzero(active)))
    if not np.all(coverage == 1):
        raise ValueError("feature shards are incomplete: {} rows missing".format(
            int(np.sum(coverage == 0))
        ))
    store = FeatureStore(
        args.output_prefix, args.manifest, rows, reference["model"],
        reference["dimensions"], reference["frame_count"], reference["pooling"],
    )
    for _, features, indices in shard_data:
        for offset in range(0, len(indices), args.chunk_size):
            batch = indices[offset:offset + args.chunk_size]
            store.write(batch.tolist(), features[batch])
    report = {
        "schema": "rq1-feature-shard-merge-v1",
        "manifest_sha256": expected_hash,
        "rows": len(rows),
        "dimensions": int(reference["dimensions"]),
        "shards": [
            {"prefix": str(prefix), "completed_rows": int(len(indices))}
            for prefix, _, indices in shard_data
        ],
        "output_prefix": str(args.output_prefix),
        "complete": bool(np.all(np.asarray(store.status) == 1)),
    }
    Path(args.report).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
