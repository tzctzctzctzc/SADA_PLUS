"""Pack prompt-free Qwen visual features into a target relation cache."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.evaluate_qwen3_vl_proposal_features import normalize
from tools.run_clip_then_llava import read_jsonl, write_json


CACHE_SCHEMA = "vlm-cross-proposal-relation-cache-v1"
FORBIDDEN_TARGET_FIELDS = {
    "gt_assigned_class_ids",
    "center_all_gt_class_ids",
    "semantic_gt_kind",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_visual_frames(frames):
    frames = normalize(frames)
    middle = frames.shape[0] // 2
    static = normalize(frames.mean(axis=0, keepdims=True))[0]
    transition = normalize(
        (frames[middle:].mean(axis=0) - frames[:middle].mean(axis=0))[None]
    )[0]
    return static.astype(np.float16), transition.astype(np.float16)


def feature_rows(index_path):
    return {
        row["candidate_id"]: row
        for row in read_jsonl(index_path)
        if row.get("status") == "ok"
    }


def relation_distance_calibration(video_ids, centers, features, max_gap_seconds):
    by_video = {}
    for video_id, center, feature in zip(video_ids, centers, features):
        by_video.setdefault(video_id, []).append((float(center), feature))
    distances = []
    for entries in by_video.values():
        entries.sort(key=lambda item: item[0])
        for left, right in zip(entries, entries[1:]):
            if right[0] - left[0] <= max_gap_seconds:
                left_feature = np.asarray(left[1], dtype=np.float32)
                right_feature = np.asarray(right[1], dtype=np.float32)
                distances.append(1.0 - float(np.dot(left_feature, right_feature)))
    if not distances:
        raise ValueError("relation cache contains no temporally adjacent pairs")
    q10, q90 = np.quantile(distances, [0.1, 0.9])
    scale = float(q90 - q10)
    if not np.isfinite(scale) or scale <= 1e-6:
        raise ValueError("VLM relation distance scale is degenerate")
    return {
        "pair_count": len(distances),
        "distance_mean": float(np.mean(distances)),
        "distance_std": float(np.std(distances)),
        "distance_q10": float(q10),
        "distance_q90": float(q90),
        "distance_scale": scale,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-meta", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--feature-layer", default="deepstack_0")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if any(FORBIDDEN_TARGET_FIELDS.intersection(row) for row in rows):
        raise ValueError("target relation manifest contains forbidden GT fields")
    meta = json.loads(Path(args.manifest_meta).read_text(encoding="utf-8"))
    if meta.get("domain") != "target" or meta.get("selection_uses_gt") is not False:
        raise ValueError("relation manifest must be target and selected without GT")
    index = feature_rows(args.feature_index)
    if len(index) != len(rows):
        raise ValueError("relation feature index is incomplete")
    layer_kind, _, layer_number = args.feature_layer.partition("_")
    if layer_kind not in ("final", "deepstack"):
        raise ValueError("feature layer must be final or deepstack_N")
    layer_index = int(layer_number) if layer_kind == "deepstack" else None

    candidate_ids, video_ids = [], []
    starts, ends, centers, levels, scores = [], [], [], [], []
    static_features, transition_features = [], []
    model_revisions = set()
    for row in rows:
        candidate_id = row["candidate_id"]
        feature_row = index[candidate_id]
        arrays = np.load(Path(args.feature_root) / feature_row["feature_file"])
        frames = (
            arrays["final_frames"]
            if layer_index is None
            else arrays["deepstack_frames"][layer_index]
        )
        static, transition = summarize_visual_frames(frames)
        candidate_ids.append(candidate_id)
        video_ids.append(row["video_id"])
        starts.append(row["proposal_start_sec"])
        ends.append(row["proposal_end_sec"])
        centers.append(row["absolute_center_time"])
        levels.append(row["source_fpn_levels"][0])
        scores.append(row["teacher_score"])
        static_features.append(static)
        transition_features.append(transition)
        model_revisions.add(str(feature_row.get("model_revision")))
    if len(model_revisions) != 1:
        raise ValueError("relation features mix model revisions")

    calibration_max_gap_seconds = 30.0
    calibration = relation_distance_calibration(
        video_ids,
        centers,
        static_features,
        calibration_max_gap_seconds,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            candidate_ids=np.asarray(candidate_ids),
            video_ids=np.asarray(video_ids),
            starts=np.asarray(starts, dtype=np.float32),
            ends=np.asarray(ends, dtype=np.float32),
            centers=np.asarray(centers, dtype=np.float32),
            levels=np.asarray(levels, dtype=np.int16),
            teacher_scores=np.asarray(scores, dtype=np.float32),
            static_features=np.asarray(static_features, dtype=np.float16),
            transition_features=np.asarray(transition_features, dtype=np.float16),
        )
    os.replace(temporary, output)
    cache_meta = {
        "schema_version": CACHE_SCHEMA,
        "domain": "target",
        "split": "train",
        "target_gt_usage": "forbidden",
        "selection_uses_gt": False,
        "selection_teacher_provenance": meta["selection_teacher_provenance"],
        "manifest_sha256": sha256_file(manifest_path),
        "feature_index_sha256": sha256_file(args.feature_index),
        "feature_layer": args.feature_layer,
        "model_revision": next(iter(model_revisions)),
        "entry_count": len(rows),
        "feature_dim": int(static_features[0].shape[0]),
        "calibration_max_gap_seconds": calibration_max_gap_seconds,
        "relation_calibration": calibration,
    }
    write_json(str(output) + ".meta.json", cache_meta)
    print(json.dumps(cache_meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
