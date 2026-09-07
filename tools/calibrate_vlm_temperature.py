"""Fit one source-only temperature for closed-set VLM class logits."""

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = "vlm-source-temperature-calibration-v1"
CLASS_NAMES = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def temperature_scaled_probabilities(logits, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(logits) != len(CLASS_NAMES):
        raise ValueError("expected ten class logits")
    scaled = [float(value) / float(temperature) for value in logits]
    normalizer = logsumexp(scaled)
    return [math.exp(value - normalizer) for value in scaled]


def calibration_rows(rows):
    selected = []
    for row in rows:
        labels = row.get("gt_assigned_class_ids") or []
        logits = row.get("class_logits")
        if row.get("status") != "ok" or not labels or logits is None:
            continue
        labels = sorted({int(label) for label in labels})
        if (
            len(logits) != len(CLASS_NAMES)
            or any(label < 0 or label >= len(CLASS_NAMES) for label in labels)
        ):
            raise ValueError("invalid source calibration row")
        selected.append({"class_logits": [float(value) for value in logits], "labels": labels})
    if not selected:
        raise ValueError("no source rows with known action labels and class logits")
    return selected


def mean_nll(rows, temperature):
    losses = []
    for row in rows:
        scaled = [value / temperature for value in row["class_logits"]]
        target_mass = logsumexp([scaled[label] for label in row["labels"]])
        losses.append(logsumexp(scaled) - target_mass)
    return sum(losses) / len(losses)


def calibration_metrics(rows, temperature, bins=10):
    nll = mean_nll(rows, temperature)
    records = []
    for row in rows:
        probabilities = temperature_scaled_probabilities(
            row["class_logits"], temperature
        )
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        records.append((max(probabilities), prediction in row["labels"]))
    mean_confidence = sum(record[0] for record in records) / len(records)
    accuracy = sum(record[1] for record in records) / len(records)
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            record for record in records
            if lower <= record[0] < upper or (index == bins - 1 and record[0] == 1.0)
        ]
        if not bucket:
            continue
        bucket_confidence = sum(record[0] for record in bucket) / len(bucket)
        bucket_accuracy = sum(record[1] for record in bucket) / len(bucket)
        ece += len(bucket) / len(records) * abs(bucket_confidence - bucket_accuracy)
    return {
        "nll": nll,
        "accuracy": accuracy,
        "mean_confidence": mean_confidence,
        "ece_10bin": ece,
    }


def fit_temperature(rows, minimum=0.05, maximum=100.0, grid_size=2001):
    if minimum <= 0 or maximum <= minimum or grid_size < 3:
        raise ValueError("invalid temperature search range")
    log_minimum = math.log(minimum)
    log_maximum = math.log(maximum)
    candidates = [
        math.exp(log_minimum + index * (log_maximum - log_minimum) / (grid_size - 1))
        for index in range(grid_size)
    ]
    return min(candidates, key=lambda value: mean_nll(rows, value))


def protocol(rows):
    values = {
        (
            row.get("model"),
            row.get("prompt_version"),
            row.get("base_prompt_sha256"),
            row.get("wire_api"),
        )
        for row in rows if row.get("status") == "ok"
    }
    if len(values) != 1:
        raise ValueError("source predictions use mixed protocols")
    model, prompt_version, prompt_sha256, wire_api = values.pop()
    return {
        "model": model,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "wire_api": wire_api,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-temperature", type=float, default=0.05)
    parser.add_argument("--maximum-temperature", type=float, default=100.0)
    parser.add_argument("--grid-size", type=int, default=2001)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source_predictions)
    selected = calibration_rows(source_rows)
    temperature = fit_temperature(
        selected,
        minimum=args.minimum_temperature,
        maximum=args.maximum_temperature,
        grid_size=args.grid_size,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "selection_uses_target_gt": False,
        "source_predictions_sha256": sha256(args.source_predictions),
        "source_protocol": protocol(source_rows),
        "known_action_rows": len(selected),
        "temperature": temperature,
        "before": calibration_metrics(selected, 1.0),
        "after": calibration_metrics(selected, temperature),
        "search": {
            "minimum_temperature": args.minimum_temperature,
            "maximum_temperature": args.maximum_temperature,
            "grid_size": args.grid_size,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
