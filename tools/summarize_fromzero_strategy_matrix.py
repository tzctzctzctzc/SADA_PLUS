#!/usr/bin/env python3
"""Summarize source-selected, from-zero SADA/VLM benchmark runs."""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(
    r"^fromzero_sada_(s[345])_(control|clip|qwen8|qwen32)_"
    r"(control|hard|soft|verify|soft_boundary|embedding|positive_rank)_"
    r"e(?P<epochs>\d+)_seed(?P<seed>\d+)_20260914$"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-results", type=Path, default=Path("model_results"))
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/fromzero_sada_strategy_matrix_20260914"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_jsonl(path):
    records = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL: {path}:{line_number}") from exc
    return records


def percent(value):
    if value is None:
        return None
    value = float(value)
    return value * 100.0 if abs(value) <= 1.5 else value


def best_epoch(run_dir):
    checkpoints = list(
        run_dir.glob("saved_ckpt/best_ckpt/checkpoint-epoch*.pt")
    )
    epochs = []
    for path in checkpoints:
        match = re.search(r"checkpoint-epoch(\d+)\.pt$", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return max(epochs) if epochs else None


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_runs(model_results):
    rows = []
    incomplete = []
    for eval_path in sorted(model_results.rglob("benchmark_eval_metrics.jsonl")):
        run_dir = eval_path.parent
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        selected_epoch = best_epoch(run_dir)
        if selected_epoch is None:
            incomplete.append({"run_name": run_dir.name, "reason": "missing_best_ckpt"})
            continue
        eval_records = load_jsonl(eval_path)
        selected_eval = next(
            (
                record for record in reversed(eval_records)
                if int(record.get("epoch", -1)) == selected_epoch
            ),
            None,
        )
        if selected_eval is None:
            incomplete.append({
                "run_name": run_dir.name,
                "reason": f"missing_eval_at_source_selected_epoch_{selected_epoch}",
            })
            continue

        train_records = load_jsonl(run_dir / "benchmark_epoch_metrics.jsonl")
        selected_train = next(
            (
                record for record in reversed(train_records)
                if int(record.get("epoch", -1)) == selected_epoch
            ),
            train_records[-1] if train_records else {},
        )
        row = {
            "run_name": run_dir.name,
            "shift": match.group(1).upper(),
            "model": match.group(2),
            "strategy": match.group(3),
            "seed": int(match.group("seed")),
            "planned_epochs": int(match.group("epochs")),
            "selected_epoch": selected_epoch,
            "target_map_percent": percent(selected_eval.get("v_tgt_map_final")),
            "source_map_percent": percent(selected_eval.get("v_src_map_final")),
            "source_validation_loss": selected_train.get("v_loss_src"),
            "target_gt_used_for_selection": selected_eval.get(
                "target_gt_used_for_selection"
            ),
            "training_initialization": selected_train.get(
                "benchmark_audit", {}
            ).get("training_initialization"),
            "target_selection_rate": selected_train.get(
                "t_vlm_target_selection_rate"
            ),
            "target_agreement_rate": selected_train.get(
                "t_vlm_target_agreement_rate"
            ),
            "target_mean_selected_weight": selected_train.get(
                "t_vlm_target_mean_selected_weight"
            ),
            "target_boundary_rate": selected_train.get(
                "t_vlm_target_boundary_rate"
            ),
            "proposal_selection_rate": selected_train.get(
                "t_proposal_semantic_selection_rate"
            ),
            "proposal_agreement_rate": selected_train.get(
                "t_proposal_semantic_agreement_rate"
            ),
            "vlm_loss_total": selected_train.get("t_vlm_target_loss_total"),
            "proposal_loss_total": selected_train.get(
                "t_proposal_semantic_loss_total"
            ),
        }
        for key, value in selected_train.items():
            if key.startswith("t_vlm_target_class_") and key.endswith("_samples_total"):
                row[key] = value
        for index in range(5):
            row[f"target_map_tiou_{index}_percent"] = percent(
                selected_eval.get(f"v_tgt_map_int_{index}_final")
            )
        rows.append(row)
    return rows, incomplete


def aggregate_runs(rows):
    controls = {
        (row["shift"], row["seed"]): row["target_map_percent"]
        for row in rows
        if row["model"] == "control" and row["target_map_percent"] is not None
    }
    grouped = defaultdict(list)
    for row in rows:
        key = (row["shift"], row["model"], row["strategy"])
        grouped[key].append(row)

    aggregates = []
    for (shift, model, strategy), group in sorted(grouped.items()):
        maps = [
            row["target_map_percent"] for row in group
            if row["target_map_percent"] is not None
        ]
        paired = [
            row["target_map_percent"] - controls[(shift, row["seed"])]
            for row in group
            if row["target_map_percent"] is not None
            and (shift, row["seed"]) in controls
        ]
        aggregates.append({
            "shift": shift,
            "model": model,
            "strategy": strategy,
            "n_seeds": len(maps),
            "seeds": ",".join(str(row["seed"]) for row in sorted(group, key=lambda x: x["seed"])),
            "target_map_mean_percent": statistics.mean(maps) if maps else None,
            "target_map_std_percent": statistics.stdev(maps) if len(maps) > 1 else 0.0 if maps else None,
            "paired_delta_vs_control_mean_percent": statistics.mean(paired) if paired else None,
            "paired_delta_vs_control_std_percent": statistics.stdev(paired) if len(paired) > 1 else 0.0 if paired else None,
        })
    return aggregates


def main():
    args = parse_args()
    output_dir = args.output_dir or args.artifact_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, incomplete = read_runs(args.model_results)
    aggregates = aggregate_runs(rows)
    write_csv(output_dir / "per_run.csv", rows)
    write_csv(output_dir / "aggregate.csv", aggregates)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": {
                    "checkpoint_selection": "source_validation_loss",
                    "reported_target_metric": "target mAP at source-selected best epoch",
                    "target_gt_role": "reporting_only",
                },
                "completed_runs": len(rows),
                "incomplete_runs": incomplete,
                "aggregates": aggregates,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(json.dumps({
        "completed_runs": len(rows),
        "incomplete_runs": len(incomplete),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
