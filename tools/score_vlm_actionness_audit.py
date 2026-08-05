"""Score cached VLM A/B/U outputs against target GT after inference."""

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


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    total = len(rows)
    foreground = sum(row["gt_is_foreground"] for row in rows)
    background = total - foreground
    predicted_action = sum(row["prediction"] == "A" for row in rows)
    true_positive = sum(
        row["prediction"] == "A" and row["gt_is_foreground"] for row in rows
    )
    false_positive = sum(
        row["prediction"] == "A" and not row["gt_is_foreground"] for row in rows
    )
    correct_action = sum(
        row["prediction"] == "A"
        and row["gt_is_foreground"]
        and row.get("sada_top1_is_correct", False)
        for row in rows
    )
    oracle_correct = sum(
        row["gt_is_foreground"] and row.get("sada_top1_is_correct", False)
        for row in rows
    )
    result = {
        "n": total,
        "gt_foreground_n": foreground,
        "gt_background_n": background,
        "prediction_counts": dict(collections.Counter(row["prediction"] for row in rows)),
        "candidate_foreground_recall": safe_ratio(true_positive, foreground),
        "background_false_promotion_rate": safe_ratio(false_positive, background),
        "promoted_action_precision": safe_ratio(true_positive, predicted_action),
        "uncertain_rate": safe_ratio(
            sum(row["prediction"] == "U" for row in rows), total
        ),
    }
    if any("sada_top1_is_correct" in row for row in rows):
        result.update(
            {
                "correct_action_routing_coverage": safe_ratio(
                    correct_action, foreground
                ),
                "class_accuracy_given_recovered_foreground": safe_ratio(
                    correct_action, true_positive
                ),
                "oracle_actionness_correct_action_coverage": safe_ratio(
                    oracle_correct, foreground
                ),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--metadata",
        help="Exporter sidecar; defaults to <manifest>.meta.json when present.",
    )
    args = parser.parse_args()

    manifest = {row["candidate_id"]: row for row in read_jsonl(args.manifest)}
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(args.predictions)
        if row.get("status") == "ok"
    }
    rows = []
    for candidate_id, prediction in predictions.items():
        if candidate_id not in manifest:
            continue
        rows.append({**manifest[candidate_id], **prediction})

    by_level = {}
    for level in sorted({row["fpn_level"] for row in rows if "fpn_level" in row}):
        by_level[str(level)] = summarize(
            [row for row in rows if row["fpn_level"] == level]
        )
    by_probe_group = {}
    for group in sorted({row["probe_group"] for row in rows if "probe_group" in row}):
        by_probe_group[group] = summarize(
            [row for row in rows if row["probe_group"] == group]
        )
    metadata_path = Path(args.metadata or (args.manifest + ".meta.json"))
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else None
    )
    global_metrics = summarize(rows)
    if metadata is not None:
        promoted_foreground = sum(
            row["prediction"] == "A" and row["gt_is_foreground"] for row in rows
        )
        promoted_correct = sum(
            row["prediction"] == "A"
            and row["gt_is_foreground"]
            and row["sada_top1_is_correct"]
            for row in rows
        )
        denominator = metadata["gt_foreground_anchors"]
        global_metrics["foreground_routing_coverage_before_vlm"] = safe_ratio(
            metadata["original_selected_foreground_anchors"], denominator
        )
        global_metrics["foreground_routing_coverage_after_vlm"] = safe_ratio(
            metadata["original_selected_foreground_anchors"] + promoted_foreground,
            denominator,
        )
        global_metrics["correct_action_routing_coverage_before_vlm"] = safe_ratio(
            metadata["original_selected_correct_action_anchors"], denominator
        )
        global_metrics["correct_action_routing_coverage_after_vlm"] = safe_ratio(
            metadata["original_selected_correct_action_anchors"] + promoted_correct,
            denominator,
        )

    result = {
        "audit": "offline_vlm_actionness",
        "target_gt_use": "post-API scoring only; never included in prompt",
        "completed_candidates": len(rows),
        "missing_candidates": len(manifest) - len(rows),
        "global": global_metrics,
        "by_fpn_level": by_level,
        "by_probe_group": by_probe_group,
        "candidate_export_metadata": metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
