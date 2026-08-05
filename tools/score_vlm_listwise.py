"""Score listwise candidate competition on fixed SADA anchors."""

import argparse
import collections
import json
from pathlib import Path


LABELS = {"A", "B", "C", "OTHER_ACTIVITY", "TRUE_BACKGROUND", "UNCERTAIN"}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, include_by_class=True):
    evaluated = []
    for row in rows:
        label = row["prediction"]
        selected = row["option_mapping"].get(label)
        all_gt = {int(value) for value in row["center_all_gt_class_ids"]}
        top1 = int(row["sada_top1_class"])
        evaluated.append(
            {
                **row,
                "top1_correct": top1 in all_gt,
                "selected_role": selected["role"] if selected else None,
                "selected_class": int(selected["class_id"]) if selected else None,
                "selected_correct": (
                    int(selected["class_id"]) in all_gt if selected else False
                ),
            }
        )

    top1_correct = [row for row in evaluated if row["top1_correct"]]
    top1_wrong = [row for row in evaluated if not row["top1_correct"]]
    top1_selected = [row for row in evaluated if row["selected_role"] == "top1"]
    candidate_selected = [row for row in evaluated if row["selected_role"]]
    correct_top1_selected = [row for row in top1_selected if row["top1_correct"]]
    wrong_top1_selected = [row for row in top1_selected if not row["top1_correct"]]
    pool_precision = ratio(len(top1_correct), len(evaluated))
    selected_precision = ratio(len(correct_top1_selected), len(top1_selected))

    result = {
        "anchors": len(evaluated),
        "candidate_pool_top1_correct_rate": pool_precision,
        "top1_selection_count": len(top1_selected),
        "top1_selection_rate": ratio(len(top1_selected), len(evaluated)),
        "selected_top1_precision": selected_precision,
        "selected_top1_precision_gain_vs_pool": (
            selected_precision - pool_precision
            if selected_precision is not None and pool_precision is not None
            else None
        ),
        "correct_top1_retention": ratio(
            len(correct_top1_selected), len(top1_correct)
        ),
        "wrong_top1_win_rate": ratio(len(wrong_top1_selected), len(top1_wrong)),
        "selected_any_candidate_accuracy": ratio(
            sum(row["selected_correct"] for row in candidate_selected),
            len(candidate_selected),
        ),
        "prediction_counts": dict(
            collections.Counter(row["prediction"] for row in evaluated)
        ),
        "selected_role_counts": dict(
            collections.Counter(
                row["selected_role"] or row["prediction"] for row in evaluated
            )
        ),
        "option_position_selection_rates": {
            letter: ratio(
                sum(row["prediction"] == letter for row in evaluated),
                len(evaluated),
            )
            for letter in "ABC"
        },
    }
    if include_by_class:
        result["by_sada_top1_class"] = {
            str(class_id): summarize(
                [row for row in rows if int(row["sada_top1_class"]) == class_id],
                include_by_class=False,
            )
            for class_id in sorted({int(row["sada_top1_class"]) for row in rows})
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest = {row["candidate_id"]: row for row in read_jsonl(args.manifest)}
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(args.predictions)
        if row.get("status") == "ok"
    }
    rows = [
        {**annotation, "prediction": predictions[candidate_id]["prediction"]}
        for candidate_id, annotation in manifest.items()
        if candidate_id in predictions
    ]
    metadata_path = Path(args.manifest + ".meta.json")
    result = {
        "audit": "listwise_candidate_competition",
        "valid_uda_result": False,
        "completed_candidates": len(rows),
        "missing_candidates": len(manifest) - len(rows),
        "metrics": summarize(rows),
        "manifest_metadata": (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
