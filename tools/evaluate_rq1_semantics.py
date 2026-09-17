"""Evaluate RQ1 semantic robustness across the six SADA domain shifts."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_clip_target_adapter import LABELS, read_jsonl


def confusion_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.float64)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recalls = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    precisions = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros_like(diagonal),
        where=(precisions + recalls) > 0,
    )
    return {
        "accuracy": float(diagonal.sum() / max(confusion.sum(), 1.0)),
        "macro_recall": float(recalls.mean()),
        "macro_f1": float(f1.mean()),
        "per_class_recall": {
            LABELS[index]: float(recalls[index]) for index in range(len(LABELS))
        },
        "confusion": confusion.astype(int).tolist(),
    }


def calibration_metrics(rows, bins=10):
    if not rows:
        return {"ece": None, "brier": None}
    confidences = np.asarray([float(row["prediction"]["confidence"]) for row in rows])
    correct = np.asarray([
        int(row["prediction"]["class_id"]) == int(row["manifest"]["gt_label_id"])
        for row in rows
    ], dtype=np.float64)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (confidences >= lower) & (
            confidences <= upper if index == bins - 1 else confidences < upper
        )
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    brier_values = []
    for row in rows:
        probabilities = np.asarray(row["prediction"]["class_probabilities"], dtype=np.float64)
        target = np.zeros(len(LABELS), dtype=np.float64)
        target[int(row["manifest"]["gt_label_id"])] = 1.0
        brier_values.append(float(np.square(probabilities - target).sum()))
    return {"ece": float(ece), "brier": float(np.mean(brier_values))}


def rows_confusion(rows):
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for row in rows:
        gt = int(row["manifest"]["gt_label_id"])
        prediction = int(row["prediction"]["class_id"])
        confusion[gt, prediction] += 1
    return confusion


def bootstrap_intervals(rows, repeats, seed):
    if repeats <= 0 or not rows:
        return {}
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["manifest"]["video_id"]].append(row)
    matrices = [rows_confusion(video_rows) for video_rows in by_video.values()]
    rng = np.random.default_rng(seed)
    accuracy = []
    macro_f1 = []
    for _ in range(repeats):
        indices = rng.integers(0, len(matrices), size=len(matrices))
        confusion = sum((matrices[index] for index in indices), np.zeros_like(matrices[0]))
        metrics = confusion_metrics(confusion)
        accuracy.append(metrics["accuracy"])
        macro_f1.append(metrics["macro_f1"])
    return {
        "unit": "video",
        "repeats": repeats,
        "accuracy_95ci": [float(value) for value in np.percentile(accuracy, [2.5, 97.5])],
        "macro_f1_95ci": [float(value) for value in np.percentile(macro_f1, [2.5, 97.5])],
    }


def summarize(expected_rows, prediction_index, shift, repeats, seed):
    joined = []
    for row in expected_rows:
        prediction = prediction_index.get((row["candidate_id"], shift))
        if prediction is None:
            prediction = prediction_index.get((row["candidate_id"], None))
        if prediction is not None and prediction.get("status") == "ok":
            joined.append({"manifest": row, "prediction": prediction})
    confusion = rows_confusion(joined)
    metrics = confusion_metrics(confusion)
    metrics.update(calibration_metrics(joined))
    metrics.update({
        "expected_count": len(expected_rows),
        "success_count": len(joined),
        "success_rate": len(joined) / len(expected_rows) if expected_rows else 0.0,
        "tolerant_accuracy": (
            sum(
                int(item["prediction"]["class_id"])
                in item["manifest"]["gt_acceptable_class_ids"]
                for item in joined
            ) / len(joined)
            if joined else None
        ),
    })
    isolated = [item for item in joined if not item["manifest"]["is_temporally_ambiguous"]]
    ambiguous = [item for item in joined if item["manifest"]["is_temporally_ambiguous"]]
    metrics["isolated_count"] = len(isolated)
    metrics["isolated_accuracy"] = confusion_metrics(rows_confusion(isolated))["accuracy"] if isolated else None
    metrics["ambiguous_count"] = len(ambiguous)
    metrics["ambiguous_strict_accuracy"] = confusion_metrics(rows_confusion(ambiguous))["accuracy"] if ambiguous else None
    metrics["bootstrap"] = bootstrap_intervals(joined, repeats, seed)
    metrics["probability_means_by_gt_class"] = {
        LABELS[class_id]: np.mean([
            item["prediction"]["class_probabilities"]
            for item in joined
            if int(item["manifest"]["gt_label_id"]) == class_id
        ], axis=0).tolist()
        for class_id in range(len(LABELS))
        if any(int(item["manifest"]["gt_label_id"]) == class_id for item in joined)
    }
    return metrics


def jensen_shannon(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / max(first.sum(), 1e-12)
    second = second / max(second.sum(), 1e-12)
    mean = 0.5 * (first + second)
    first_mask = first > 0
    second_mask = second > 0
    return float(0.5 * (
        np.sum(first[first_mask] * np.log(first[first_mask] / mean[first_mask]))
        + np.sum(second[second_mask] * np.log(second[second_mask] / mean[second_mask]))
    ))


def prediction_drift(source, target):
    per_class = {}
    for label in LABELS:
        if (
            label in source["probability_means_by_gt_class"]
            and label in target["probability_means_by_gt_class"]
        ):
            per_class[label] = jensen_shannon(
                source["probability_means_by_gt_class"][label],
                target["probability_means_by_gt_class"][label],
            )
    return {
        "metric": "Jensen-Shannon divergence of mean predictions conditioned on GT class",
        "mean": float(np.mean(list(per_class.values()))) if per_class else None,
        "per_class": per_class,
    }


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 0 else None


def aggregate_shifts(shifts):
    metric_names = ("accuracy", "macro_recall", "macro_f1")
    aggregate = {}
    for metric in metric_names:
        source = np.asarray([
            shifts[shift]["source"][metric] for shift in SHIFT_CONFIGS
        ], dtype=np.float64)
        target = np.asarray([
            shifts[shift]["target"][metric] for shift in SHIFT_CONFIGS
        ], dtype=np.float64)
        source_mean = float(source.mean())
        target_mean = float(target.mean())
        aggregate[metric] = {
            "source_mean": source_mean,
            "target_mean": target_mean,
            "target_minus_source": target_mean - source_mean,
            "retention": safe_ratio(target_mean, source_mean),
            "target_worst_shift": float(target.min()),
            "target_shift_std": float(target.std()),
        }
    success_rates = [
        shifts[shift][domain]["success_rate"]
        for shift in SHIFT_CONFIGS
        for domain in ("source", "target")
    ]
    aggregate["validity"] = {
        "all_predictions_present": all(value == 1.0 for value in success_rates),
        "minimum_success_rate": float(min(success_rates)),
    }
    return aggregate


def parse_prediction_spec(spec):
    if "=" not in spec:
        raise ValueError("prediction must be NAME=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise ValueError("prediction must be NAME=PATH")
    return name, Path(path)


def stable_seed(base_seed, *values):
    payload = "|".join(map(str, values)).encode("utf-8")
    return base_seed + int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_repeats < 0:
        raise ValueError("bootstrap repeats must be non-negative")
    manifest = read_jsonl(args.manifest)
    results = {
        "protocol": "rq1-semantic-robustness-gt-segments-v1",
        "target_gt_role": "offline_evaluation_only",
        "models": {},
    }
    csv_rows = []
    for spec in args.prediction:
        model_name, path = parse_prediction_spec(spec)
        predictions = read_jsonl(path)
        prediction_index = {
            (row["candidate_id"], row.get("shift")): row
            for row in predictions
            if row.get("candidate_id")
        }
        model_result = {"prediction_file": str(path), "shifts": {}}
        for shift in SHIFT_CONFIGS:
            domains = {}
            for domain in ("source", "target"):
                expected = [
                    row for row in manifest
                    if {"shift": shift, "domain": domain} in row["memberships"]
                ]
                metrics = summarize(
                    expected,
                    prediction_index,
                    shift,
                    args.bootstrap_repeats,
                    stable_seed(args.bootstrap_seed, model_name, shift, domain),
                )
                domains[domain] = metrics
                csv_rows.append({
                    "model": model_name,
                    "shift": shift,
                    "domain": domain,
                    "expected_count": metrics["expected_count"],
                    "success_rate": metrics["success_rate"],
                    "accuracy": metrics["accuracy"],
                    "macro_recall": metrics["macro_recall"],
                    "macro_f1": metrics["macro_f1"],
                    "ece": metrics["ece"],
                    "brier": metrics["brier"],
                    "tolerant_accuracy": metrics["tolerant_accuracy"],
                })
            domains["target_minus_source"] = {
                key: domains["target"][key] - domains["source"][key]
                for key in ("accuracy", "macro_recall", "macro_f1")
            }
            domains["target_over_source"] = {
                key: safe_ratio(domains["target"][key], domains["source"][key])
                for key in ("accuracy", "macro_recall", "macro_f1")
            }
            domains["class_conditional_prediction_drift"] = prediction_drift(
                domains["source"], domains["target"]
            )
            model_result["shifts"][shift] = domains
        model_result["aggregate"] = aggregate_shifts(model_result["shifts"])
        results["models"][model_name] = model_result

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({
        "models": list(results["models"]),
        "rows": len(csv_rows),
        "output_json": str(output_json),
        "output_csv": str(output_csv),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
