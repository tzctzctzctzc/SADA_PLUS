import argparse
import collections
import json
from pathlib import Path


KNOWN = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows, manifest_count, thresholds):
    min_success_rate = thresholds["min_success_rate"]
    min_consensus_coverage = thresholds["min_consensus_coverage"]
    min_precision_gain = thresholds["min_precision_gain"]
    min_background_precision = thresholds["min_background_precision"]
    base_correct = sum(
        int(row["teacher_top1_class"]) in row["gt_assigned_class_ids"]
        for row in rows
    )
    consensus = [
        row for row in rows
        if row["prediction"] in KNOWN
        and KNOWN.index(row["prediction"]) == int(row["teacher_top1_class"])
    ]
    consensus_correct = sum(
        int(row["teacher_top1_class"]) in row["gt_assigned_class_ids"]
        for row in consensus
    )
    known = [row for row in rows if row["prediction"] in KNOWN]
    known_correct = sum(
        KNOWN.index(row["prediction"]) in row["gt_assigned_class_ids"]
        for row in known
    )
    disagreement = [
        row for row in known
        if KNOWN.index(row["prediction"]) != int(row["teacher_top1_class"])
    ]
    corrected = sum(
        KNOWN.index(row["prediction"]) in row["gt_assigned_class_ids"]
        for row in disagreement
    )
    background = [row for row in rows if row["prediction"] == "BACKGROUND"]
    background_correct = sum(
        row["semantic_gt_kind"] == "BACKGROUND" for row in background
    )
    unknown = [row for row in rows if row["prediction"] == "UNKNOWN_ACTION"]
    unknown_correct = sum(
        row["semantic_gt_kind"] == "UNKNOWN_ACTION" for row in unknown
    )

    success_rate = ratio(len(rows), manifest_count)
    base_precision = ratio(base_correct, len(rows))
    consensus_precision = ratio(consensus_correct, len(consensus))
    consensus_coverage = ratio(len(consensus), len(rows))
    precision_gain = (
        consensus_precision - base_precision
        if consensus_precision is not None and base_precision is not None
        else None
    )
    adapter_pass = all((
        success_rate is not None and success_rate >= min_success_rate,
        consensus_coverage is not None
        and consensus_coverage >= min_consensus_coverage,
        precision_gain is not None and precision_gain >= min_precision_gain,
    ))
    background_precision = ratio(background_correct, len(background))
    return {
        "gate_version": "vlm-target-adapter-source-gate-v1",
        "adapter_pass": adapter_pass,
        "background_enabled": (
            background_precision is not None
            and background_precision >= min_background_precision
        ),
        "thresholds": thresholds,
        "metrics": {
            "manifest_count": manifest_count,
            "completed_count": len(rows),
            "success_rate": success_rate,
            "base_teacher_precision": base_precision,
            "consensus_count": len(consensus),
            "consensus_coverage": consensus_coverage,
            "consensus_precision": consensus_precision,
            "consensus_precision_gain": precision_gain,
            "vlm_known_count": len(known),
            "vlm_known_precision": ratio(known_correct, len(known)),
            "disagreement_count": len(disagreement),
            "disagreement_correction_rate": ratio(corrected, len(disagreement)),
            "background_count": len(background),
            "background_precision": background_precision,
            "unknown_count": len(unknown),
            "unknown_precision": ratio(unknown_correct, len(unknown)),
            "prediction_counts": dict(collections.Counter(
                row["prediction"] for row in rows
            )),
        },
        "gt_use": "source held-out scoring only; never included in VLM prompt",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--predictions", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    args = parser.parse_args()

    manifest = {row["candidate_id"]: row for row in read_jsonl(args.manifest)}
    prediction_rows = [
        row
        for path in args.predictions
        for row in read_jsonl(path)
        if row.get("status") == "ok"
    ]
    predictions = {row["candidate_id"]: row for row in prediction_rows}
    rows = [
        {**row, "prediction": predictions[row_id]["prediction"]}
        for row_id, row in manifest.items()
        if row_id in predictions
    ]
    for row in rows:
        if "gt_assigned_class_ids" not in row:
            raise ValueError("source gate manifest has no GT audit fields")

    result = summarize(rows, len(manifest), {
            "min_success_rate": args.min_success_rate,
            "min_consensus_coverage": args.min_consensus_coverage,
            "min_precision_gain": args.min_precision_gain,
            "min_background_precision": args.min_background_precision,
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
