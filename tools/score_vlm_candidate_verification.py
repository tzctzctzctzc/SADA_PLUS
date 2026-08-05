"""Score a four-way, candidate-conditioned VLM capability probe."""

import argparse
import collections
import json
from pathlib import Path


LABELS = {
    "CONFIRM_TARGET",
    "OTHER_ACTIVITY",
    "TRUE_BACKGROUND",
    "UNCERTAIN",
}


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    by_group = {}
    for group in sorted({row["probe_group"] for row in rows}):
        group_rows = [row for row in rows if row["probe_group"] == group]
        counts = collections.Counter(row["prediction"] for row in group_rows)
        by_group[group] = {
            "n": len(group_rows),
            "prediction_counts": dict(counts),
            "confirm_rate": ratio(counts["CONFIRM_TARGET"], len(group_rows)),
            "true_background_rate": ratio(
                counts["TRUE_BACKGROUND"], len(group_rows)
            ),
            "uncertain_rate": ratio(counts["UNCERTAIN"], len(group_rows)),
        }

    positives = [row for row in rows if row["probe_group"] == "candidate_action"]
    confirmed = [row for row in rows if row["prediction"] == "CONFIRM_TARGET"]
    correct_confirmed = [row for row in confirmed if row["probe_group"] == "candidate_action"]
    negatives = [row for row in rows if row["probe_group"] != "candidate_action"]
    return {
        "n": len(rows),
        "candidate_confirm_recall": ratio(len(correct_confirmed), len(positives)),
        "confirmed_precision": ratio(len(correct_confirmed), len(confirmed)),
        "non_candidate_false_confirm_rate": ratio(
            sum(row["prediction"] == "CONFIRM_TARGET" for row in negatives),
            len(negatives),
        ),
        "overall_uncertain_rate": ratio(
            sum(row["prediction"] == "UNCERTAIN" for row in rows), len(rows)
        ),
        "by_probe_group": by_group,
    }


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
    rows = []
    for candidate_id, annotation in manifest.items():
        if candidate_id not in predictions:
            continue
        prediction = predictions[candidate_id].get("prediction", "UNCERTAIN")
        if prediction not in LABELS:
            prediction = "UNCERTAIN"
        rows.append({**annotation, "prediction": prediction})

    result = {
        "audit": "candidate_conditioned_vlm_capability_probe",
        "valid_uda_result": False,
        "selection_uses_target_gt": True,
        "completed_candidates": len(rows),
        "missing_candidates": len(manifest) - len(rows),
        "metrics": summarize(rows),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
