import argparse
import hashlib
import json
import math
from pathlib import Path

from tools.run_clip_target_adapter import LABELS


INVERSE_LABEL = {
    "TAKE": "PUT",
    "PUT": "TAKE",
    "OPEN": "CLOSE",
    "CLOSE": "OPEN",
    "TURN_ON": "TURN_OFF",
    "TURN_OFF": "TURN_ON",
}
FUSION_VERSION = "phase-transition-counterfactual-fusion-v1"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def successful_by_id(path):
    rows = {}
    for row in read_jsonl(path):
        if row.get("status") == "ok":
            rows[str(row["candidate_id"])] = row
    return rows


def normalize(values):
    total = sum(values)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("probability mass must be finite and positive")
    return [float(value) / total for value in values]


def counterfactual_probabilities(forward, reverse):
    if len(forward) != len(LABELS) or len(reverse) != len(LABELS):
        raise ValueError("expected one probability per action class")
    scores = []
    for index, label in enumerate(LABELS):
        if label in INVERSE_LABEL:
            reverse_index = LABELS.index(INVERSE_LABEL[label])
            score = math.sqrt(
                max(float(forward[index]), 1e-12)
                * max(float(reverse[reverse_index]), 1e-12)
            )
        else:
            score = max(float(forward[index]), 0.0)
        scores.append(score)
    return normalize(scores)


def fused_row(forward, reverse):
    probabilities = counterfactual_probabilities(
        forward["class_probabilities"], reverse["class_probabilities"]
    )
    class_id = max(range(len(LABELS)), key=probabilities.__getitem__)
    ordered = sorted(probabilities, reverse=True)
    protocol_payload = {
        "fusion_version": FUSION_VERSION,
        "forward_prompt_version": forward["prompt_version"],
        "forward_prompt_sha256": forward["base_prompt_sha256"],
        "reverse_prompt_version": reverse["prompt_version"],
        "reverse_prompt_sha256": reverse["base_prompt_sha256"],
    }
    protocol_sha = hashlib.sha256(
        json.dumps(protocol_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    mapped_reverse = INVERSE_LABEL.get(
        reverse["prediction"], reverse["prediction"]
    )
    return {
        **forward,
        "prediction": LABELS[class_id],
        "class_id": class_id,
        "confidence": probabilities[class_id],
        "margin": ordered[0] - ordered[1],
        "entropy": -sum(
            value * math.log(max(value, 1e-12)) for value in probabilities
        ),
        "class_probabilities": probabilities,
        "class_logits": [math.log(max(value, 1e-12)) for value in probabilities],
        "wire_api": "deterministic-temporal-counterfactual-fusion",
        "prompt_version": FUSION_VERSION,
        "base_prompt_sha256": protocol_sha,
        "prompt_sha256": protocol_sha,
        "counterfactual_forward_prediction": forward["prediction"],
        "counterfactual_reverse_prediction": reverse["prediction"],
        "counterfactual_mapped_reverse_prediction": mapped_reverse,
        "counterfactual_agreement": forward["prediction"] == mapped_reverse,
        "counterfactual_protocol": protocol_payload,
    }


def classification_metrics(manifest, predictions):
    rows = []
    for candidate_id, source in manifest.items():
        prediction = predictions.get(candidate_id)
        gt = [
            int(value)
            for value in source.get("gt_assigned_class_ids", [])
            if 0 <= int(value) < len(LABELS)
        ]
        if prediction is not None and gt:
            rows.append((gt, prediction))
    correct = sum(int(row[1]["class_id"]) in row[0] for row in rows)
    per_class = {}
    recalls = []
    for class_id, label in enumerate(LABELS):
        selected = [row for row in rows if class_id in row[0]]
        class_correct = sum(int(row[1]["class_id"]) == class_id for row in selected)
        recall = class_correct / len(selected) if selected else None
        if recall is not None:
            recalls.append(recall)
        per_class[label] = {
            "support": len(selected),
            "correct": class_correct,
            "recall": recall,
        }
    reversible = [
        row for row in rows
        if any(LABELS[class_id] in INVERSE_LABEL for class_id in row[0])
    ]
    reversible_correct = sum(
        int(row[1]["class_id"]) in row[0] for row in reversible
    )
    return {
        "count": len(rows),
        "accuracy": correct / len(rows) if rows else None,
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "reversible_count": len(reversible),
        "reversible_accuracy": (
            reversible_correct / len(reversible) if reversible else None
        ),
        "mean_confidence": (
            sum(float(row[1]["confidence"]) for row in rows) / len(rows)
            if rows else None
        ),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--phase-forward", required=True)
    parser.add_argument("--phase-reverse", required=True)
    parser.add_argument("--fused-output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--minimum-accuracy-gain", type=float, default=0.03)
    args = parser.parse_args()

    manifest_rows = read_jsonl(args.manifest)
    manifest = {str(row["candidate_id"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("manifest contains duplicate candidates")
    baseline = successful_by_id(args.baseline)
    forward = successful_by_id(args.phase_forward)
    reverse = successful_by_id(args.phase_reverse)
    shared = sorted(set(forward).intersection(reverse))
    fused = {candidate_id: fused_row(forward[candidate_id], reverse[candidate_id])
             for candidate_id in shared}

    fused_output = Path(args.fused_output)
    fused_output.parent.mkdir(parents=True, exist_ok=True)
    fused_output.write_text(
        "".join(json.dumps(fused[key], sort_keys=True) + "\n" for key in shared),
        encoding="utf-8",
    )

    metrics = {
        "baseline": classification_metrics(manifest, baseline),
        "phase_forward": classification_metrics(manifest, forward),
        "phase_counterfactual": classification_metrics(manifest, fused),
    }
    baseline_accuracy = metrics["baseline"]["accuracy"]
    best_name = max(
        ("phase_forward", "phase_counterfactual"),
        key=lambda name: metrics[name]["accuracy"],
    )
    best_accuracy = metrics[best_name]["accuracy"]
    report = {
        "schema_version": "phase-counterfactual-source-analysis-v1",
        "target_gt_usage": "forbidden",
        "selection_split": "source_val",
        "manifest_count": len(manifest),
        "shared_phase_predictions": len(shared),
        "minimum_accuracy_gain": args.minimum_accuracy_gain,
        "recommended_variant": best_name,
        "accuracy_gain": best_accuracy - baseline_accuracy,
        "source_gate_pass": (
            best_accuracy - baseline_accuracy >= args.minimum_accuracy_gain
        ),
        "metrics": metrics,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
