"""Fit fixed source-only linear heads on frozen RQ1 segment features."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.rq1_feature_store import SCHEMA, sha256_file
from tools.run_clip_target_adapter import LABELS, read_jsonl


def normalize_rows(features):
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def source_train_indices(rows, shift):
    return [
        index for index, row in enumerate(rows)
        if any(
            role.get("shift") == shift and role.get("role") == "source_train"
            for role in row["memberships"]
        )
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--feature-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_iter <= 0 or args.alpha <= 0:
        raise ValueError("max_iter and alpha must be positive")
    from sklearn.linear_model import SGDClassifier

    feature_rows = read_jsonl(args.feature_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    row_index = {row["candidate_id"]: index for index, row in enumerate(feature_rows)}
    if len(row_index) != len(feature_rows):
        raise ValueError("feature manifest contains duplicate candidate IDs")
    prefix = Path(args.feature_prefix)
    meta = json.loads(Path(str(prefix) + ".meta.json").read_text(encoding="utf-8"))
    if meta.get("schema") != SCHEMA:
        raise ValueError("unsupported feature schema")
    if meta.get("manifest_sha256") != sha256_file(args.feature_manifest):
        raise ValueError("feature manifest hash differs from the feature store")
    features = np.load(str(prefix) + ".features.npy", mmap_mode="r")
    status = np.load(str(prefix) + ".status.npy", mmap_mode="r")
    if len(features) != len(feature_rows) or not np.all(status == 1):
        raise ValueError("feature store is incomplete")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for shift in SHIFT_CONFIGS:
            train_indices = source_train_indices(feature_rows, shift)
            train_labels = np.asarray(
                [feature_rows[index]["gt_label_id"] for index in train_indices],
                dtype=np.int64,
            )
            classifier = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=args.alpha,
                class_weight="balanced",
                max_iter=args.max_iter,
                tol=1e-3,
                random_state=args.seed,
                average=True,
            )
            classifier.fit(normalize_rows(features[train_indices]), train_labels)
            if list(classifier.classes_) != list(range(len(LABELS))):
                raise RuntimeError("linear probe did not observe all ten classes")
            shift_rows = [
                row for row in eval_rows
                if any(role["shift"] == shift for role in row["memberships"])
            ]
            eval_indices = [row_index[row["candidate_id"]] for row in shift_rows]
            probabilities = classifier.predict_proba(
                normalize_rows(features[eval_indices])
            )
            for row, probs in zip(shift_rows, probabilities):
                order = np.argsort(probs)[::-1]
                top1, top2 = map(int, order[:2])
                stream.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "video_id": row["video_id"],
                    "shift": shift,
                    "status": "ok",
                    "model": args.model_name,
                    "source_supervision": True,
                    "class_id": top1,
                    "prediction": LABELS[top1],
                    "confidence": float(probs[top1]),
                    "margin": float(probs[top1] - probs[top2]),
                    "class_probabilities": probs.tolist(),
                }, sort_keys=True) + "\n")
            print(
                "{} train={} evaluate={}".format(
                    shift, len(train_indices), len(shift_rows)
                ), flush=True,
            )
    Path(str(output) + ".protocol.json").write_text(json.dumps({
        "model": args.model_name,
        "frozen_feature_model": meta["model"],
        "feature_schema": meta["schema"],
        "feature_pooling": meta["pooling"],
        "frame_count": meta["frame_count"],
        "source_supervision_only": True,
        "source_val_labels_used_for_training": False,
        "target_labels_used_for_training": False,
        "classifier": {
            "type": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": args.alpha,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
            "average": True,
            "seed": args.seed,
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
