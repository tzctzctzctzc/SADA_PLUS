"""Compare two source-side VLM proposal classifiers without using target GT."""

import argparse
import json
from pathlib import Path

from tools.run_clip_target_adapter import LABELS
from tools.run_clip_then_llava import read_jsonl, write_json


LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
DIRECTIONAL_PAIRS = (
    ("TAKE", "PUT"),
    ("OPEN", "CLOSE"),
    ("TURN_ON", "TURN_OFF"),
)


def is_known(row):
    return row.get("status") == "ok" and bool(row.get("gt_assigned_class_ids"))


def is_correct(row):
    return int(row.get("class_id", -1)) in row["gt_assigned_class_ids"]


def expected_calibration_error(rows, bins=10):
    total = len(rows)
    error = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        selected = [
            row for row in rows
            if lower <= float(row["confidence"]) <= upper
            and (bin_index == bins - 1 or float(row["confidence"]) < upper)
        ]
        if not selected:
            continue
        accuracy = sum(is_correct(row) for row in selected) / len(selected)
        confidence = sum(float(row["confidence"]) for row in selected) / len(selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return error


def multiclass_brier(rows):
    singleton = [row for row in rows if len(row["gt_assigned_class_ids"]) == 1]
    if not singleton:
        return None, 0
    scores = []
    for row in singleton:
        gt = int(row["gt_assigned_class_ids"][0])
        scores.append(sum(
            (float(probability) - float(class_id == gt)) ** 2
            for class_id, probability in enumerate(row["class_probabilities"])
        ))
    return sum(scores) / len(scores), len(singleton)


def precision_at_count(rows, count):
    if count <= 0:
        return None
    ranked = sorted(rows, key=lambda row: float(row["confidence"]), reverse=True)
    selected = ranked[:min(count, len(ranked))]
    return sum(is_correct(row) for row in selected) / len(selected)


def summarize(rows, threshold, bins):
    per_class_recall = {}
    recalls = []
    for class_id, label in enumerate(LABELS):
        class_rows = [row for row in rows if class_id in row["gt_assigned_class_ids"]]
        recall = (
            sum(is_correct(row) for row in class_rows) / len(class_rows)
            if class_rows else None
        )
        per_class_recall[label] = {"support": len(class_rows), "recall": recall}
        if recall is not None:
            recalls.append(recall)

    directional = {}
    for first, second in DIRECTIONAL_PAIRS:
        ids = {LABEL_TO_ID[first], LABEL_TO_ID[second]}
        pair_rows = [row for row in rows if ids.intersection(row["gt_assigned_class_ids"])]
        directional["{}/{}".format(first, second)] = {
            "support": len(pair_rows),
            "accuracy": (
                sum(is_correct(row) for row in pair_rows) / len(pair_rows)
                if pair_rows else None
            ),
        }

    selected = [row for row in rows if float(row["confidence"]) >= threshold]
    brier, brier_count = multiclass_brier(rows)
    return {
        "known_count": len(rows),
        "accuracy": sum(is_correct(row) for row in rows) / len(rows),
        "macro_recall": sum(recalls) / len(recalls),
        "per_class_recall": per_class_recall,
        "directional_subsets": directional,
        "mean_confidence": sum(float(row["confidence"]) for row in rows) / len(rows),
        "mean_margin": sum(float(row["margin"]) for row in rows) / len(rows),
        "ece": expected_calibration_error(rows, bins),
        "ece_bins": bins,
        "multiclass_brier_singleton_gt": brier,
        "brier_count": brier_count,
        "threshold": threshold,
        "threshold_coverage": len(selected) / len(rows),
        "threshold_precision": (
            sum(is_correct(row) for row in selected) / len(selected)
            if selected else None
        ),
    }


def protocol(rows):
    keys = (
        "model", "model_revision", "prompt_version", "base_prompt_sha256",
        "frame_count", "min_pixels", "max_pixels", "score_temperature",
    )
    return {key: sorted({str(row.get(key)) for row in rows}) for key in keys}


def compare(baseline_rows, candidate_rows, threshold=0.7, bins=10):
    baseline = {row["candidate_id"]: row for row in baseline_rows if is_known(row)}
    candidate = {row["candidate_id"]: row for row in candidate_rows if is_known(row)}
    shared_ids = sorted(set(baseline).intersection(candidate))
    if not shared_ids:
        raise ValueError("no shared known-action proposals")
    baseline_shared = [baseline[candidate_id] for candidate_id in shared_ids]
    candidate_shared = [candidate[candidate_id] for candidate_id in shared_ids]
    for old, new in zip(baseline_shared, candidate_shared):
        if old["gt_assigned_class_ids"] != new["gt_assigned_class_ids"]:
            raise ValueError("GT mismatch for {}".format(old["candidate_id"]))

    baseline_metrics = summarize(baseline_shared, threshold, bins)
    candidate_metrics = summarize(candidate_shared, threshold, bins)
    old_correct = [is_correct(row) for row in baseline_shared]
    new_correct = [is_correct(row) for row in candidate_shared]
    baseline_selected_count = sum(
        float(row["confidence"]) >= threshold for row in baseline_shared
    )
    return {
        "shared_known_count": len(shared_ids),
        "baseline_protocol": protocol(baseline_shared),
        "candidate_protocol": protocol(candidate_shared),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "paired_difference": {
            "accuracy_pp": 100 * (
                candidate_metrics["accuracy"] - baseline_metrics["accuracy"]
            ),
            "macro_recall_pp": 100 * (
                candidate_metrics["macro_recall"] - baseline_metrics["macro_recall"]
            ),
            "baseline_correct_retention": (
                sum(old and new for old, new in zip(old_correct, new_correct))
                / sum(old_correct) if any(old_correct) else None
            ),
            "corrected_baseline_errors": sum(
                (not old) and new for old, new in zip(old_correct, new_correct)
            ),
            "broken_baseline_correct": sum(
                old and (not new) for old, new in zip(old_correct, new_correct)
            ),
            "baseline_precision_at_baseline_threshold_coverage": (
                precision_at_count(baseline_shared, baseline_selected_count)
            ),
            "candidate_precision_at_baseline_threshold_coverage": (
                precision_at_count(candidate_shared, baseline_selected_count)
            ),
            "baseline_threshold_selected_count": baseline_selected_count,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--ece-bins", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be within [0, 1]")
    if args.ece_bins <= 0:
        raise ValueError("ECE bins must be positive")
    result = compare(
        read_jsonl(Path(args.baseline)),
        read_jsonl(Path(args.candidate)),
        args.confidence_threshold,
        args.ece_bins,
    )
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
