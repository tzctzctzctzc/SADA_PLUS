"""Evaluate oracle-class temporal boundary understanding across SADA shifts."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_clip_target_adapter import read_jsonl


def temporal_iou(first, second):
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return intersection / union if union > 0 else 0.0


def prediction_interval(row, prediction):
    count = int(row["frame_count"])
    start = min(max(float(prediction["pred_start_position"]), 0.0), count)
    end = min(max(float(prediction["pred_end_position"]), 0.0), count)
    if end <= start:
        return None
    window_start = float(row["proposal_start_sec"])
    scale = (float(row["proposal_end_sec"]) - window_start) / count
    return window_start + start * scale, window_start + end * scale


def stable_seed(base_seed, *values):
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def bootstrap(rows, repeats, seed):
    if not rows or repeats <= 0:
        return {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(float(row["effective_tiou"]))
    units = list(grouped.values())
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        chosen = rng.integers(0, len(units), len(units))
        means.append(float(np.mean([value for index in chosen for value in units[index]])))
    return {
        "unit": "video", "repeats": repeats,
        "effective_mean_tiou_95ci": np.percentile(means, [2.5, 97.5]).tolist(),
    }


def summarize(expected, prediction_index, shift, repeats, seed, prior=False):
    scored = []
    valid_tious = []
    start_errors = []
    end_errors = []
    for row in expected:
        if prior:
            prediction = {
                "status": "ok",
                "pred_start_position": row["frame_count"] * 0.625 / 2.25,
                "pred_end_position": row["frame_count"] * 1.625 / 2.25,
            }
        else:
            prediction = prediction_index.get((row["candidate_id"], shift))
            if prediction is None:
                prediction = prediction_index.get((row["candidate_id"], None))
        interval = None
        if prediction is not None and prediction.get("status") == "ok":
            interval = prediction_interval(row, prediction)
        tiou = 0.0
        if interval is not None:
            truth = (float(row["gt_start_sec"]), float(row["gt_end_sec"]))
            tiou = temporal_iou(interval, truth)
            valid_tious.append(tiou)
            duration = float(row["duration_sec"])
            start_errors.append(abs(interval[0] - truth[0]) / duration)
            end_errors.append(abs(interval[1] - truth[1]) / duration)
        scored.append({
            "video_id": row["video_id"], "effective_tiou": tiou,
            "valid": interval is not None,
        })
    count = len(expected)
    valid = sum(row["valid"] for row in scored)
    result = {
        "expected_count": count,
        "valid_count": valid,
        "valid_rate": valid / count if count else 0.0,
        "mean_tiou_valid": float(np.mean(valid_tious)) if valid_tious else None,
        "effective_mean_tiou": float(np.mean([row["effective_tiou"] for row in scored])) if scored else None,
        "recall_at_0.3": sum(row["effective_tiou"] >= 0.3 for row in scored) / count if count else None,
        "recall_at_0.5": sum(row["effective_tiou"] >= 0.5 for row in scored) / count if count else None,
        "recall_at_0.7": sum(row["effective_tiou"] >= 0.7 for row in scored) / count if count else None,
        "start_nmae_valid": float(np.mean(start_errors)) if start_errors else None,
        "end_nmae_valid": float(np.mean(end_errors)) if end_errors else None,
        "bootstrap": bootstrap(scored, repeats, seed),
    }
    return result


def parse_prediction(spec):
    if "=" not in spec:
        raise ValueError("prediction must be NAME=PATH")
    name, path = spec.split("=", 1)
    return name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prediction", action="append", default=[])
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
    models = [("Position-prior", None)]
    for spec in args.prediction:
        name, path = parse_prediction(spec)
        predictions = read_jsonl(path)
        models.append((name, {
            (row["candidate_id"], row.get("shift")): row
            for row in predictions if row.get("candidate_id")
        }))
    results = {
        "protocol": "rq2-oracle-class-jittered-context-boundary-v1",
        "target_gt_role": "offline_evaluation_only",
        "models": {},
    }
    csv_rows = []
    for model_name, index in models:
        model_result = {"shifts": {}}
        for shift in SHIFT_CONFIGS:
            shift_result = {}
            for split_name, selector in (
                ("all", lambda row: True),
                ("isolated", lambda row: not row["is_challenge"]),
                ("challenge", lambda row: row["is_challenge"]),
            ):
                domains = {}
                for domain in ("source", "target"):
                    expected = [
                        row for row in manifest
                        if selector(row) and {"shift": shift, "domain": domain} in row["memberships"]
                    ]
                    metrics = summarize(
                        expected, index or {}, shift, args.bootstrap_repeats,
                        stable_seed(args.bootstrap_seed, model_name, shift, split_name, domain),
                        prior=index is None,
                    )
                    domains[domain] = metrics
                    csv_rows.append({
                        "model": model_name, "shift": shift, "split": split_name,
                        "domain": domain,
                        **{key: metrics[key] for key in (
                            "expected_count", "valid_rate", "effective_mean_tiou",
                            "recall_at_0.3", "recall_at_0.5", "recall_at_0.7",
                            "start_nmae_valid", "end_nmae_valid",
                        )},
                    })
                domains["target_minus_source"] = {
                    key: domains["target"][key] - domains["source"][key]
                    for key in ("effective_mean_tiou", "recall_at_0.5")
                }
                shift_result[split_name] = domains
            model_result["shifts"][shift] = shift_result
        results["models"][model_name] = model_result
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
