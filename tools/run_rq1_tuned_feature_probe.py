"""Tune a source-only frozen-feature probe on source validation labels."""

import argparse
import json
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.rq1_feature_store import SCHEMA, sha256_file
from tools.run_clip_target_adapter import LABELS, read_jsonl


def normalize_rows(features):
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)


def membership_indices(rows, shift, role):
    return [
        index for index, row in enumerate(rows)
        if any(
            item.get("shift") == shift and item.get("role") == role
            for item in row["memberships"]
        )
    ]


def parse_csv_values(value, cast):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("value list must not be empty")
    return values


def fit_classifier(features, labels, alpha, seed, max_iter):
    from sklearn.linear_model import SGDClassifier

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=max_iter,
        tol=1e-3,
        random_state=seed,
        average=True,
    )
    classifier.fit(features, labels)
    if list(classifier.classes_) != list(range(len(LABELS))):
        raise RuntimeError("linear probe did not observe all ten classes")
    return classifier


def prediction_row(row, shift, model_name, probabilities, alpha, seed):
    order = np.argsort(probabilities)[::-1]
    top1, top2 = map(int, order[:2])
    return {
        "candidate_id": row["candidate_id"],
        "video_id": row["video_id"],
        "shift": shift,
        "status": "ok",
        "model": model_name,
        "source_supervision": True,
        "selected_alpha": alpha,
        "seed": seed,
        "class_id": top1,
        "prediction": LABELS[top1],
        "confidence": float(probabilities[top1]),
        "margin": float(probabilities[top1] - probabilities[top2]),
        "class_probabilities": probabilities.tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--eval-manifest", required=True)
    parser.add_argument("--feature-prefix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--alphas", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--selection-seeds", default=None,
        help="seeds used only for source-val alpha selection; defaults to --seeds",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--write-seed-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_iter <= 0:
        raise ValueError("max_iter must be positive")
    from sklearn.metrics import accuracy_score, f1_score

    alphas = parse_csv_values(args.alphas, float)
    seeds = parse_csv_values(args.seeds, int)
    selection_seeds = parse_csv_values(
        args.selection_seeds if args.selection_seeds is not None else args.seeds,
        int,
    )
    if any(alpha <= 0 for alpha in alphas):
        raise ValueError("alphas must be positive")

    feature_rows = read_jsonl(args.feature_manifest)
    eval_rows = read_jsonl(args.eval_manifest)
    row_index = {row["candidate_id"]: index for index, row in enumerate(feature_rows)}
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
    ensemble_stream = output.open("w", encoding="utf-8")
    seed_streams = {}
    if args.write_seed_predictions:
        for seed in seeds:
            path = output.with_name(output.stem + ".seed{}".format(seed) + output.suffix)
            seed_streams[seed] = path.open("w", encoding="utf-8")

    report = {"selection_metric": "source_val_macro_f1", "shifts": {}}
    try:
        for shift in SHIFT_CONFIGS:
            train_indices = membership_indices(feature_rows, shift, "source_train")
            val_indices = membership_indices(feature_rows, shift, "source_val")
            train_x = normalize_rows(features[train_indices])
            train_y = np.asarray(
                [feature_rows[index]["gt_label_id"] for index in train_indices],
                dtype=np.int64,
            )
            val_x = normalize_rows(features[val_indices])
            val_y = np.asarray(
                [feature_rows[index]["gt_label_id"] for index in val_indices],
                dtype=np.int64,
            )
            grid = {}
            for alpha in alphas:
                scores = []
                for seed in selection_seeds:
                    classifier = fit_classifier(
                        train_x, train_y, alpha, seed, args.max_iter
                    )
                    prediction = classifier.predict(val_x)
                    scores.append({
                        "seed": seed,
                        "macro_f1": float(f1_score(
                            val_y, prediction, labels=list(range(len(LABELS))),
                            average="macro", zero_division=0,
                        )),
                        "accuracy": float(accuracy_score(val_y, prediction)),
                    })
                grid[str(alpha)] = scores
            selected_alpha = max(
                alphas,
                key=lambda alpha: (
                    np.mean([item["macro_f1"] for item in grid[str(alpha)]]),
                    np.mean([item["accuracy"] for item in grid[str(alpha)]]),
                    -alpha,
                ),
            )
            shift_rows = [
                row for row in eval_rows
                if any(item["shift"] == shift for item in row["memberships"])
            ]
            eval_indices = [row_index[row["candidate_id"]] for row in shift_rows]
            eval_x = normalize_rows(features[eval_indices])
            seed_probabilities = []
            for seed in seeds:
                classifier = fit_classifier(
                    train_x, train_y, selected_alpha, seed, args.max_iter
                )
                probabilities = classifier.predict_proba(eval_x)
                seed_probabilities.append(probabilities)
                if seed in seed_streams:
                    for row, probs in zip(shift_rows, probabilities):
                        seed_streams[seed].write(json.dumps(prediction_row(
                            row, shift, args.model_name + "-seed{}".format(seed),
                            probs, selected_alpha, seed,
                        ), sort_keys=True) + "\n")
            ensemble = np.mean(seed_probabilities, axis=0)
            for row, probs in zip(shift_rows, ensemble):
                ensemble_stream.write(json.dumps(prediction_row(
                    row, shift, args.model_name, probs, selected_alpha, "ensemble",
                ), sort_keys=True) + "\n")
            report["shifts"][shift] = {
                "selected_alpha": selected_alpha,
                "train_count": len(train_indices),
                "source_val_count": len(val_indices),
                "grid": grid,
            }
            print(
                "{} alpha={} train={} val={} evaluate={}".format(
                    shift, selected_alpha, len(train_indices), len(val_indices),
                    len(shift_rows),
                ), flush=True,
            )
    finally:
        ensemble_stream.close()
        for stream in seed_streams.values():
            stream.close()

    protocol = {
        "model": args.model_name,
        "frozen_feature_model": meta["model"],
        "feature_pooling": meta["pooling"],
        "frame_count": meta["frame_count"],
        "source_train_labels_used_for_training": True,
        "source_val_labels_used_for_selection": True,
        "target_labels_used_for_training_or_selection": False,
        "alphas": alphas,
        "seeds": seeds,
        "selection_seeds": selection_seeds,
        "prediction_aggregation": "mean class probabilities across seeds",
        "selection_report": report,
    }
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
