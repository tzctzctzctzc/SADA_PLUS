"""Estimate proposal-label reliability from source validation annotations."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.run_clip_target_adapter import LABELS
from tools.run_clip_then_llava import read_jsonl, write_json


def interval_has_class(row, database):
    start = float(row["proposal_start_sec"])
    end = float(row["proposal_end_sec"])
    class_id = int(row["class_id"])
    return any(
        int(annotation["label_id"]) == class_id
        and min(end, float(annotation["segment"][1]))
        > max(start, float(annotation["segment"][0]))
        for annotation in database[row["video_id"]]["annotations"]
    )


def calibrate(rows, database, confidence_threshold=0.7, shrinkage=5.0):
    selected = [
        row for row in rows
        if row.get("status") == "ok"
        and row.get("prediction") in LABELS
        and float(row.get("confidence", 0.0)) >= confidence_threshold
    ]
    correct = [interval_has_class(row, database) for row in selected]
    global_reliability = sum(correct) / len(correct)
    counts = []
    correct_counts = []
    raw_reliabilities = []
    smoothed = []
    for class_id in range(len(LABELS)):
        indices = [
            index for index, row in enumerate(selected)
            if int(row["class_id"]) == class_id
        ]
        class_correct = sum(correct[index] for index in indices)
        count = len(indices)
        counts.append(count)
        correct_counts.append(class_correct)
        raw_reliabilities.append(class_correct / count if count else None)
        smoothed.append(
            (class_correct + shrinkage * global_reliability)
            / (count + shrinkage)
        )
    return {
        "schema_version": "proposal-class-reliability-v1",
        "source_split": "val",
        "gt_view": "proposal_interval_any",
        "target_gt_usage": "forbidden",
        "confidence_threshold": float(confidence_threshold),
        "shrinkage": float(shrinkage),
        "selected_count": len(selected),
        "global_reliability": global_reliability,
        "labels": list(LABELS),
        "counts": counts,
        "correct_counts": correct_counts,
        "raw_reliabilities": raw_reliabilities,
        "smoothed_reliabilities": smoothed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence", type=float, default=0.7)
    parser.add_argument("--shrinkage", type=float, default=5.0)
    args = parser.parse_args()
    if args.shrinkage <= 0.0:
        raise ValueError("shrinkage must be positive")
    prediction_path = Path(args.predictions)
    annotation_path = Path(args.annotations)
    database = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )["database"]
    result = calibrate(
        read_jsonl(prediction_path),
        database,
        args.confidence,
        args.shrinkage,
    )
    result["prediction_sha256"] = hashlib.sha256(
        prediction_path.read_bytes()
    ).hexdigest()
    result["annotation_sha256"] = hashlib.sha256(
        annotation_path.read_bytes()
    ).hexdigest()
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
