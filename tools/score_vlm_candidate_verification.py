"""Score four-way candidate verification on probes or real SADA candidates."""

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


def summarize(rows, include_by_class=True):
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
    result = {
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
    if include_by_class and any(row.get("candidate_label_id") is not None for row in rows):
        result["by_candidate_class"] = {
            str(class_id): summarize(
                [row for row in rows if int(row["candidate_label_id"]) == class_id],
                include_by_class=False,
            )
            for class_id in sorted(
                {
                    int(row["candidate_label_id"])
                    for row in rows
                    if row.get("candidate_label_id") is not None
                }
            )
        }
    return result


def summarize_real(rows, metadata=None, include_by_class=True):
    confirmed = [row for row in rows if row["prediction"] == "CONFIRM_TARGET"]
    correct = [row for row in rows if row["sada_top1_is_correct"]]
    correct_confirmed = [row for row in confirmed if row["sada_top1_is_correct"]]
    wrong_action = [
        row
        for row in rows
        if row["gt_is_foreground"] and not row["sada_top1_is_correct"]
    ]
    background = [row for row in rows if not row["gt_is_foreground"]]

    metrics = {
        "n": len(rows),
        "candidate_correct_confirm_recall": ratio(
            len(correct_confirmed), len(correct)
        ),
        "confirmed_correct_action_precision": ratio(
            len(correct_confirmed), len(confirmed)
        ),
        "wrong_action_class_false_confirm_rate": ratio(
            sum(row["prediction"] == "CONFIRM_TARGET" for row in wrong_action),
            len(wrong_action),
        ),
        "background_false_confirm_rate": ratio(
            sum(row["prediction"] == "CONFIRM_TARGET" for row in background),
            len(background),
        ),
        "background_routing_recall": ratio(
            sum(row["prediction"] == "TRUE_BACKGROUND" for row in background),
            len(background),
        ),
        "overall_uncertain_rate": ratio(
            sum(row["prediction"] == "UNCERTAIN" for row in rows), len(rows)
        ),
        "prediction_counts": dict(
            collections.Counter(row["prediction"] for row in rows)
        ),
    }

    if include_by_class:
        by_class = {}
        for class_id in sorted({int(row["sada_top1_class"]) for row in rows}):
            class_rows = [
                row for row in rows if int(row["sada_top1_class"]) == class_id
            ]
            by_class[str(class_id)] = summarize_real(
                class_rows, include_by_class=False
            )
        metrics["by_sada_top1_class"] = by_class

    if metadata:
        denominator = metadata.get("gt_foreground_anchors", 0)
        original_correct = metadata.get(
            "original_selected_correct_action_anchors", 0
        )
        metrics["correct_action_routing_coverage_before_vlm"] = ratio(
            original_correct, denominator
        )
        metrics["correct_action_routing_coverage_after_vlm"] = ratio(
            original_correct + len(correct_confirmed), denominator
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata",
        help="Candidate exporter sidecar; defaults to <manifest>.meta.json.",
    )
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

    is_probe = bool(rows) and "probe_group" in rows[0]
    metadata_path = Path(args.metadata or (args.manifest + ".meta.json"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    result = {
        "audit": (
            "candidate_conditioned_vlm_capability_probe"
            if is_probe
            else "candidate_conditioned_vlm_sada_audit"
        ),
        "valid_uda_result": False,
        "selection_uses_target_gt": is_probe,
        "target_gt_use": (
            "probe construction and scoring"
            if is_probe
            else "post-inference scoring only"
        ),
        "completed_candidates": len(rows),
        "missing_candidates": len(manifest) - len(rows),
        "metrics": summarize(rows) if is_probe else summarize_real(rows, metadata),
        "candidate_export_metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
