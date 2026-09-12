"""Weight target VLM semantics with a source-trained proposal-quality probe."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.evaluate_qwen3_vl_proposal_features import (
    _rank_metrics,
    group_ridge_predictions,
    load_feature_variants,
    normalize,
    proposal_quality_targets,
)
from tools.run_clip_then_llava import read_jsonl, write_json


PROTOCOL_VERSION = "vlm-source-quality-rank-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def structure_features(scores, starts, ends, levels):
    scores = np.asarray(scores, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.float32)
    ends = np.asarray(ends, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.int64)
    return np.asarray([
        [
            float(score),
            float(np.log1p(max(end - start, 0.0))),
            *[float(level == candidate) for candidate in range(3)],
        ]
        for score, start, end, level in zip(scores, starts, ends, levels)
    ], dtype=np.float32)


def normalize_structure(train, test):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    return (
        (train - mean) / np.maximum(std, 1e-6),
        (test - mean) / np.maximum(std, 1e-6),
    )


def fit_ridge(train_features, targets, alpha):
    center = float(targets.mean())
    kernel = train_features @ train_features.T
    dual = np.linalg.solve(
        kernel + float(alpha) * np.eye(len(train_features), dtype=np.float32),
        targets - center,
    )
    return center, dual


def within_group_rank_weights(entries, predictions):
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        groups[(str(entry["video_id"]), str(entry["semantic_label"]))].append(index)
    weights = np.ones(len(entries), dtype=np.float32)
    for indices in groups.values():
        if len(indices) == 1:
            continue
        order = sorted(indices, key=lambda index: (predictions[index], index))
        for rank, index in enumerate(order):
            weights[index] = 0.5 + rank / float(len(order) - 1)
    return weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--source-feature-index", required=True)
    parser.add_argument("--source-feature-root", required=True)
    parser.add_argument("--source-annotation-json", required=True)
    parser.add_argument("--target-relation-cache", required=True)
    parser.add_argument("--target-semantic-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.alpha <= 0.0:
        raise ValueError("ridge alpha must be positive")

    source_rows = read_jsonl(args.source_manifest)
    database = json.loads(
        Path(args.source_annotation_json).read_text(encoding="utf-8")
    )["database"]
    source_rows = proposal_quality_targets(source_rows, database)
    source_qwen = load_feature_variants(
        args.source_feature_index, args.source_feature_root
    )["deepstack_0_combined"]
    source_rows = [
        row for row in source_rows if row["candidate_id"] in source_qwen
    ]
    source_structure = structure_features(
        [row["teacher_score"] for row in source_rows],
        [row["proposal_start_sec"] for row in source_rows],
        [row["proposal_end_sec"] for row in source_rows],
        [row["source_fpn_levels"][0] for row in source_rows],
    )
    source_visual = normalize([
        source_qwen[row["candidate_id"]] for row in source_rows
    ])
    source_targets = np.asarray([
        row["quality_iou_class"] for row in source_rows
    ], dtype=np.float32)
    source_groups = np.asarray([row["video_id"] for row in source_rows])
    oof_predictions = group_ridge_predictions(
        {"structure": source_structure, "qwen": source_visual},
        source_targets,
        source_groups,
        args.folds,
        alpha=args.alpha,
    )

    relation = np.load(args.target_relation_cache, allow_pickle=False)
    target_ids = [str(value) for value in relation["candidate_ids"]]
    target_structure = structure_features(
        relation["teacher_scores"], relation["starts"], relation["ends"],
        relation["levels"],
    )
    source_structure, target_structure = normalize_structure(
        source_structure, target_structure
    )
    target_visual = normalize(np.concatenate((
        relation["static_features"].astype(np.float32),
        relation["transition_features"].astype(np.float32),
    ), axis=1))
    train_features = np.concatenate((source_structure, source_visual), axis=1)
    target_features = np.concatenate((target_structure, target_visual), axis=1)
    center, dual = fit_ridge(train_features, source_targets, args.alpha)
    target_predictions = np.clip(
        center + target_features @ train_features.T @ dual, 0.0, 1.0
    )
    prediction_by_id = dict(zip(target_ids, target_predictions.tolist()))

    semantic_path = Path(args.target_semantic_cache)
    payload = json.loads(semantic_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not entries or any(
        entry["candidate_id"] not in prediction_by_id for entry in entries
    ):
        raise ValueError("semantic and relation caches do not share all candidates")
    ordered_predictions = np.asarray([
        prediction_by_id[entry["candidate_id"]] for entry in entries
    ], dtype=np.float32)
    weights = within_group_rank_weights(entries, ordered_predictions)
    for entry, prediction, weight in zip(entries, ordered_predictions, weights):
        entry["proposal_quality_prediction"] = float(prediction)
        entry["proposal_quality_weight"] = float(weight)

    payload["quality_weighting"] = {
        "protocol_version": PROTOCOL_VERSION,
        "target_gt_usage": "forbidden",
        "source_target": "same-class temporal IoU",
        "feature": "Qwen3-VL-8B deepstack_0 static+transition",
        "weight": "0.5 + within-video-within-semantic-class percentile rank",
        "ridge_alpha": float(args.alpha),
        "source_count": len(source_rows),
        "target_count": len(entries),
        "source_group_oof": _rank_metrics(source_targets, oof_predictions),
        "source_manifest_sha256": file_sha256(args.source_manifest),
        "source_feature_index_sha256": file_sha256(args.source_feature_index),
        "target_relation_cache_sha256": file_sha256(args.target_relation_cache),
        "input_semantic_cache_sha256": file_sha256(semantic_path),
    }
    write_json(args.output, payload)
    print(json.dumps(payload["quality_weighting"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
