"""Score whether a VLM selectively verifies candidates on the same clips."""

import argparse
import collections
import json
from pathlib import Path


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, include_by_class=True):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row["anchor_id"]].append(row)

    anchor_summaries = []
    for anchor_id, candidates in groups.items():
        confirmed = [
            row for row in candidates if row["prediction"] == "CONFIRM_TARGET"
        ]
        top1 = next(row for row in candidates if row["candidate_role"] == "top1")
        wrong_non_top1 = [
            row
            for row in candidates
            if row["candidate_role"] != "top1"
            and not row["candidate_is_center_gt"]
        ]
        exclusive_correct = (
            len(confirmed) == 1 and confirmed[0]["candidate_is_center_gt"]
        )
        anchor_summaries.append(
            {
                "anchor_id": anchor_id,
                "top1_class": int(top1["candidate_label_id"]),
                "top1_is_correct": bool(top1["candidate_is_center_gt"]),
                "top1_confirmed": top1["prediction"] == "CONFIRM_TARGET",
                "top1_selective": (
                    top1["prediction"] == "CONFIRM_TARGET"
                    and all(
                        row["prediction"] != "CONFIRM_TARGET"
                        for row in wrong_non_top1
                    )
                ),
                "confirm_count": len(confirmed),
                "exclusive_correct": exclusive_correct,
            }
        )

    wrong = [row for row in rows if not row["candidate_is_center_gt"]]
    correct = [row for row in rows if row["candidate_is_center_gt"]]
    exclusive = [item for item in anchor_summaries if item["confirm_count"] == 1]
    correct_top1 = [item for item in anchor_summaries if item["top1_is_correct"]]
    role_metrics = {}
    for role in ("top1", "top2", "near_confusion", "random_wrong"):
        role_rows = [row for row in rows if row["candidate_role"] == role]
        role_wrong = [row for row in role_rows if not row["candidate_is_center_gt"]]
        role_metrics[role] = {
            "n": len(role_rows),
            "confirm_rate": ratio(
                sum(row["prediction"] == "CONFIRM_TARGET" for row in role_rows),
                len(role_rows),
            ),
            "wrong_n": len(role_wrong),
            "wrong_candidate_confirmation_rate": ratio(
                sum(row["prediction"] == "CONFIRM_TARGET" for row in role_wrong),
                len(role_wrong),
            ),
        }

    result = {
        "anchors": len(anchor_summaries),
        "candidate_queries": len(rows),
        "candidate_selectivity_rate": ratio(
            sum(item["top1_selective"] for item in anchor_summaries),
            len(anchor_summaries),
        ),
        "candidate_selectivity_on_correct_top1": ratio(
            sum(item["top1_selective"] for item in correct_top1),
            len(correct_top1),
        ),
        "multi_confirmation_rate": ratio(
            sum(item["confirm_count"] >= 2 for item in anchor_summaries),
            len(anchor_summaries),
        ),
        "zero_confirmation_rate": ratio(
            sum(item["confirm_count"] == 0 for item in anchor_summaries),
            len(anchor_summaries),
        ),
        "mean_confirmations_per_clip": ratio(
            sum(item["confirm_count"] for item in anchor_summaries),
            len(anchor_summaries),
        ),
        "wrong_candidate_confirmation_rate": ratio(
            sum(row["prediction"] == "CONFIRM_TARGET" for row in wrong),
            len(wrong),
        ),
        "correct_candidate_confirm_recall": ratio(
            sum(row["prediction"] == "CONFIRM_TARGET" for row in correct),
            len(correct),
        ),
        "exclusive_confirmation_count": len(exclusive),
        "exclusive_confirmation_rate": ratio(len(exclusive), len(anchor_summaries)),
        "exclusive_confirmation_precision": ratio(
            sum(item["exclusive_correct"] for item in exclusive), len(exclusive)
        ),
        "confirmation_count_histogram": dict(
            collections.Counter(item["confirm_count"] for item in anchor_summaries)
        ),
        "by_candidate_role": role_metrics,
    }
    if include_by_class:
        result["by_sada_top1_class"] = {
            str(class_id): summarize(
                [row for row in rows if int(row["sada_top1_class"]) == class_id],
                include_by_class=False,
            )
            for class_id in sorted(
                {int(row["sada_top1_class"]) for row in rows}
            )
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
        "audit": "candidate_selectivity",
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
