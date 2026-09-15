"""Paired video-bootstrap comparison of RQ1 eight- and sixteen-frame runs."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.evaluate_rq1_semantics import confusion_metrics
from tools.run_clip_target_adapter import LABELS, read_jsonl


def parse_pair(spec):
    parts = spec.split("=", 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("pair must be NAME=EIGHT_FRAME_PATH=SIXTEEN_FRAME_PATH")
    return parts[0], Path(parts[1]), Path(parts[2])


def prediction_index(path):
    return {
        (row["candidate_id"], row.get("shift")): row
        for row in read_jsonl(path)
        if row.get("candidate_id") and row.get("status") == "ok"
    }


def lookup(index, candidate_id, shift):
    return index.get((candidate_id, shift)) or index.get((candidate_id, None))


def confusion(rows, index, shift):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for row in rows:
        prediction = lookup(index, row["candidate_id"], shift)
        if prediction is None:
            continue
        matrix[int(row["gt_label_id"]), int(prediction["class_id"])] += 1
    return matrix


def stable_seed(base_seed, *values):
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def paired_bootstrap(rows, first, second, shift, repeats, seed):
    if not rows or repeats <= 0:
        return {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    pairs = [
        (confusion(video_rows, first, shift), confusion(video_rows, second, shift))
        for video_rows in grouped.values()
    ]
    rng = np.random.default_rng(seed)
    accuracy_deltas = []
    macro_f1_deltas = []
    zero = np.zeros_like(pairs[0][0])
    for _ in range(repeats):
        indices = rng.integers(0, len(pairs), len(pairs))
        first_matrix = sum((pairs[index][0] for index in indices), zero.copy())
        second_matrix = sum((pairs[index][1] for index in indices), zero.copy())
        first_metrics = confusion_metrics(first_matrix)
        second_metrics = confusion_metrics(second_matrix)
        accuracy_deltas.append(second_metrics["accuracy"] - first_metrics["accuracy"])
        macro_f1_deltas.append(second_metrics["macro_f1"] - first_metrics["macro_f1"])
    return {
        "unit": "video",
        "repeats": repeats,
        "accuracy_delta_95ci": np.percentile(accuracy_deltas, [2.5, 97.5]).tolist(),
        "macro_f1_delta_95ci": np.percentile(macro_f1_deltas, [2.5, 97.5]).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.bootstrap_repeats < 0:
        raise ValueError("bootstrap repeats must be non-negative")
    manifest = read_jsonl(args.manifest)
    results = {"protocol": "rq1-paired-frame-density-8f-vs-16f-v1", "models": {}}
    csv_rows = []
    for spec in args.pair:
        model, first_path, second_path = parse_pair(spec)
        first = prediction_index(first_path)
        second = prediction_index(second_path)
        model_result = {"eight_frame_file": str(first_path), "sixteen_frame_file": str(second_path), "shifts": {}}
        for shift in SHIFT_CONFIGS:
            domains = {}
            for domain in ("source", "target"):
                rows = [
                    row for row in manifest
                    if {"shift": shift, "domain": domain} in row["memberships"]
                ]
                first_matrix = confusion(rows, first, shift)
                second_matrix = confusion(rows, second, shift)
                first_metrics = confusion_metrics(first_matrix)
                second_metrics = confusion_metrics(second_matrix)
                first_count = int(first_matrix.sum())
                second_count = int(second_matrix.sum())
                if first_count != len(rows) or second_count != len(rows):
                    raise RuntimeError(
                        "incomplete pair {} {} {}: expected {}, got {}/{}".format(
                            model, shift, domain, len(rows), first_count, second_count
                        )
                    )
                paired = paired_bootstrap(
                    rows, first, second, shift, args.bootstrap_repeats,
                    stable_seed(args.bootstrap_seed, model, shift, domain),
                )
                metrics = {
                    "count": len(rows),
                    "accuracy_8f": first_metrics["accuracy"],
                    "accuracy_16f": second_metrics["accuracy"],
                    "accuracy_delta_16f_minus_8f": second_metrics["accuracy"] - first_metrics["accuracy"],
                    "macro_f1_8f": first_metrics["macro_f1"],
                    "macro_f1_16f": second_metrics["macro_f1"],
                    "macro_f1_delta_16f_minus_8f": second_metrics["macro_f1"] - first_metrics["macro_f1"],
                    "paired_bootstrap": paired,
                }
                domains[domain] = metrics
                csv_rows.append({
                    "model": model, "shift": shift, "domain": domain,
                    **{key: metrics[key] for key in (
                        "count", "accuracy_8f", "accuracy_16f",
                        "accuracy_delta_16f_minus_8f", "macro_f1_8f",
                        "macro_f1_16f", "macro_f1_delta_16f_minus_8f",
                    )},
                    "accuracy_delta_ci_low": paired["accuracy_delta_95ci"][0],
                    "accuracy_delta_ci_high": paired["accuracy_delta_95ci"][1],
                    "macro_f1_delta_ci_low": paired["macro_f1_delta_95ci"][0],
                    "macro_f1_delta_ci_high": paired["macro_f1_delta_95ci"][1],
                })
            model_result["shifts"][shift] = domains
        results["models"][model] = model_result
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    output_csv = Path(args.output_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"models": list(results["models"]), "rows": len(csv_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
