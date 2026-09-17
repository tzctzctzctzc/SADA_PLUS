"""Tune a source-only frozen-feature probe with grouped source-train CV."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
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


def select_alpha_one_standard_error(grid, alphas):
    """Prefer the strongest regularization statistically tied with the CV best."""
    macro_f1_means = {
        alpha: float(np.mean([
            item["macro_f1"] for item in grid[str(alpha)]
        ]))
        for alpha in alphas
    }
    accuracy_means = {
        alpha: float(np.mean([
            item["accuracy"] for item in grid[str(alpha)]
        ]))
        for alpha in alphas
    }
    best_alpha = max(
        alphas,
        key=lambda alpha: (
            macro_f1_means[alpha], accuracy_means[alpha], -alpha,
        ),
    )
    best_scores = np.asarray([
        item["macro_f1"] for item in grid[str(best_alpha)]
    ], dtype=np.float64)
    standard_error = (
        float(best_scores.std(ddof=1) / np.sqrt(len(best_scores)))
        if len(best_scores) > 1 else 0.0
    )
    threshold = macro_f1_means[best_alpha] - standard_error
    eligible = [
        alpha for alpha in alphas
        if macro_f1_means[alpha] >= threshold
    ]
    selected_alpha = max(eligible)
    return selected_alpha, {
        "rule": "one_standard_error_then_strongest_regularization",
        "best_alpha_by_mean_macro_f1": best_alpha,
        "best_mean_macro_f1": macro_f1_means[best_alpha],
        "best_macro_f1_standard_error": standard_error,
        "macro_f1_threshold": threshold,
        "eligible_alphas": eligible,
        "selected_mean_macro_f1": macro_f1_means[selected_alpha],
        "selected_mean_accuracy": accuracy_means[selected_alpha],
    }


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


def grouped_cv_splits(labels, groups, folds, seed):
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    if len(unique_groups) < folds:
        raise ValueError(
            "only {} source-train videos for {} CV folds".format(
                len(unique_groups), folds
            )
        )
    class_count = int(labels.max()) + 1
    total = np.bincount(labels, minlength=class_count).astype(np.float64)
    group_counts = {
        group: np.bincount(
            labels[groups == group], minlength=class_count
        ).astype(np.float64)
        for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)
    shuffled.sort(
        key=lambda group: (
            -float(group_counts[group].std()),
            -float(group_counts[group].sum()),
        )
    )
    fold_counts = np.zeros((folds, class_count), dtype=np.float64)
    fold_groups = [set() for _ in range(folds)]
    for group in shuffled:
        counts = group_counts[group]
        best_fold = None
        best_score = None
        for fold in range(folds):
            fold_counts[fold] += counts
            proportions = np.divide(
                fold_counts,
                total[None, :],
                out=np.zeros_like(fold_counts),
                where=total[None, :] > 0,
            )
            balance = float(np.std(proportions, axis=0).mean())
            size_balance = float(np.std(fold_counts.sum(axis=1))) / max(
                float(total.sum()), 1.0
            )
            score = (balance, size_balance, len(fold_groups[fold]), fold)
            fold_counts[fold] -= counts
            if best_score is None or score < best_score:
                best_score = score
                best_fold = fold
        fold_counts[best_fold] += counts
        fold_groups[best_fold].add(group)
    splits = []
    all_indices = np.arange(len(labels))
    for validation_groups in fold_groups:
        validation_mask = np.isin(groups, list(validation_groups))
        validation_indices = all_indices[validation_mask]
        fit_indices = all_indices[~validation_mask]
        if not len(validation_indices) or not len(fit_indices):
            raise ValueError("grouped CV produced an empty fold")
        splits.append((fit_indices, validation_indices))
    return splits


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
    parser.add_argument(
        "--alphas",
        default="1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--selection-seeds", default=None,
        help="seeds used only for source-train grouped-CV alpha selection; defaults to --seeds",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--cv-seed", type=int, default=20260917)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--write-seed-predictions", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_iter <= 0 or args.cv_folds < 2 or args.jobs <= 0:
        raise ValueError(
            "max_iter and jobs must be positive and cv_folds must be at least two"
        )
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

    report = {
        "selection_metric": "source_train_grouped_cv_macro_f1",
        "selection_unit": "video_id",
        "shifts": {},
    }
    try:
        for shift in SHIFT_CONFIGS:
            train_indices = membership_indices(feature_rows, shift, "source_train")
            source_val_indices = membership_indices(feature_rows, shift, "source_val")
            train_x = normalize_rows(features[train_indices])
            train_y = np.asarray(
                [feature_rows[index]["gt_label_id"] for index in train_indices],
                dtype=np.int64,
            )
            train_groups = np.asarray([
                feature_rows[index]["video_id"] for index in train_indices
            ])
            unique_groups = np.unique(train_groups)
            try:
                cv_splits = grouped_cv_splits(
                    train_y, train_groups, args.cv_folds, args.cv_seed
                )
            except ValueError as error:
                raise ValueError("{}: {}".format(shift, error)) from error
            grid = {}
            tasks = [
                (alpha, fold, seed, fit_indices, validation_indices)
                for alpha in alphas
                for fold, (fit_indices, validation_indices) in enumerate(cv_splits)
                for seed in selection_seeds
            ]

            def score_task(task):
                alpha, fold, seed, fit_indices, validation_indices = task
                classifier = fit_classifier(
                    train_x[fit_indices], train_y[fit_indices],
                    alpha, seed, args.max_iter,
                )
                prediction = classifier.predict(train_x[validation_indices])
                return alpha, {
                    "fold": fold,
                    "seed": seed,
                    "validation_videos": int(len(np.unique(
                        train_groups[validation_indices]
                    ))),
                    "macro_f1": float(f1_score(
                        train_y[validation_indices], prediction,
                        labels=list(range(len(LABELS))),
                        average="macro", zero_division=0,
                    )),
                    "accuracy": float(accuracy_score(
                        train_y[validation_indices], prediction
                    )),
                }

            for alpha in alphas:
                grid[str(alpha)] = []
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                for alpha, score in executor.map(score_task, tasks):
                    grid[str(alpha)].append(score)
            selected_alpha, selection_summary = select_alpha_one_standard_error(
                grid, alphas
            )
            shift_rows = [
                row for row in eval_rows
                if any(item["shift"] == shift for item in row["memberships"])
            ]
            eval_indices = [row_index[row["candidate_id"]] for row in shift_rows]
            eval_x = normalize_rows(features[eval_indices])
            def fit_seed(seed):
                classifier = fit_classifier(
                    train_x, train_y, selected_alpha, seed, args.max_iter
                )
                return seed, classifier.predict_proba(eval_x)

            seed_probabilities = []
            with ThreadPoolExecutor(max_workers=min(args.jobs, len(seeds))) as executor:
                fitted = list(executor.map(fit_seed, seeds))
            for seed, probabilities in fitted:
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
                "selection_at_grid_boundary": selected_alpha in (
                    min(alphas), max(alphas)
                ),
                "train_count": len(train_indices),
                "source_train_video_count": len(unique_groups),
                "source_val_count": len(source_val_indices),
                "cv_folds": args.cv_folds,
                "selection_summary": selection_summary,
                "grid": grid,
            }
            print(
                "{} alpha={} train={} videos={} source_val={} evaluate={}".format(
                    shift, selected_alpha, len(train_indices), len(unique_groups),
                    len(source_val_indices), len(shift_rows),
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
        "source_train_grouped_cv_used_for_selection": True,
        "source_val_labels_used_for_selection": False,
        "target_labels_used_for_training_or_selection": False,
        "alphas": alphas,
        "seeds": seeds,
        "selection_seeds": selection_seeds,
        "cv_folds": args.cv_folds,
        "cv_seed": args.cv_seed,
        "parallel_jobs": args.jobs,
        "prediction_aggregation": "mean class probabilities across seeds",
        "selection_rule": "one_standard_error_then_strongest_regularization",
        "selection_report": report,
    }
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
