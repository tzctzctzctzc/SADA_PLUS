"""Extract frozen SlowFast GT-segment features for the unified RQ1 probe."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.rq1_feature_store import FeatureStore
from tools.run_clip_target_adapter import read_jsonl
from tools.run_rq1_slowfast_probe import segment_feature


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feat-stride", type=float, default=16.0)
    parser.add_argument("--feat-num-frames", type=float, default=32.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def feature_dimensions(feature_root, rows):
    for row in rows:
        path = Path(feature_root) / (row["video_id"] + ".npz")
        if path.is_file():
            with np.load(path) as payload:
                features = payload["feats"]
            if features.ndim != 2 or not features.shape[1]:
                raise ValueError("invalid SlowFast feature shape: {}".format(path))
            return int(features.shape[1])
    raise FileNotFoundError("no SlowFast feature file found for the RQ1 manifest")


def main():
    args = parse_args()
    if min(args.fps, args.feat_stride, args.feat_num_frames) <= 0:
        raise ValueError("feature timing values must be positive")
    rows = read_jsonl(args.manifest)
    dimensions = feature_dimensions(args.feature_root, rows)
    model = json.dumps({
        "name": "SADA-SlowFast-frozen-features",
        "fps": args.fps,
        "feat_stride": args.feat_stride,
        "feat_num_frames": args.feat_num_frames,
        "input_evidence": "dense_precomputed_video_features",
    }, sort_keys=True)
    store = FeatureStore(
        args.output_prefix,
        args.manifest,
        rows,
        model,
        dimensions,
        0,
        "gt_segment_grid_mean",
    )
    pending = store.pending_indices()
    if args.limit > 0:
        pending = pending[:args.limit]
    by_video = defaultdict(list)
    for index in pending:
        by_video[rows[index]["video_id"]].append(index)
    processed = 0
    for video_id in sorted(by_video):
        path = Path(args.feature_root) / (video_id + ".npz")
        with np.load(path) as payload:
            grids = payload["feats"]
        indices = by_video[video_id]
        values = [
            segment_feature(
                grids,
                [rows[index]["proposal_start_sec"], rows[index]["proposal_end_sec"]],
                fps=args.fps,
                feat_stride=args.feat_stride,
                num_frames=args.feat_num_frames,
            )
            for index in indices
        ]
        store.write(indices, values)
        processed += len(indices)
        print(
            "{}/{} segments {}".format(processed, len(pending), video_id),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
