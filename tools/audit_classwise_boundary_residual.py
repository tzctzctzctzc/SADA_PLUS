"""Audit source-calibrated classwise residual correction for Stage 2."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.utils.utils import _update_config, read_yaml
from tools.audit_boundary_residual_correction import (
    collect_records,
    corrected_offsets,
    fit_ridge,
    metrics,
)


def parse_numbers(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def global_residual(features, calibration, device):
    features = features.to(device, dtype=torch.float64)
    normalized = (
        features - calibration["mean"].to(device)
    ) / calibration["std"].to(device)
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    return augmented.matmul(calibration["weight"].to(device)).float()


def fit_class_bias(classes, residuals, num_classes, prior, device):
    classes = classes.to(device)
    residuals = residuals.to(device)
    sums = torch.zeros((num_classes, 2), device=device, dtype=residuals.dtype)
    counts = torch.zeros((num_classes, 1), device=device, dtype=residuals.dtype)
    sums.index_add_(0, classes, residuals)
    counts.index_add_(0, classes, torch.ones_like(residuals[:, :1]))
    return sums / (counts + prior)


def classwise_corrected_offsets(
        features, offsets, classes, calibration, class_bias, gamma, device):
    predicted = global_residual(features, calibration, device)
    predicted = predicted + class_bias.to(device)[classes.to(device)]
    return (offsets.to(device) + gamma * predicted).clamp_min(0).float()


def load_model_and_data(args, domain):
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    split = cfg["src_train_split"] if domain == "source_train" else (
        cfg["src_val_split"] if domain == "source_val" else cfg["tgt_val_split"]
    )
    train = domain == "source_train"
    dataset = make_dataset(cfg["dataset_name"], train, split, **cfg["dataset"])
    return cfg, checkpoint, model, annotations, dataset


def source_select(args):
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations, train = load_model_and_data(args, "source_train")
    val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"],
    )
    levels = [int(value) for value in args.levels.split(",")]
    train_records = collect_records(model, train, annotations, cfg, "source", "train", levels, args.device)
    val_records = collect_records(model, val, annotations, cfg, "source", "val", levels, args.device)
    num_classes = int(cfg["dataset"]["num_classes"])
    candidates, best = [], None
    for l2 in parse_numbers(args.ridge_l2):
        calibration = fit_ridge(
            train_records["hidden"],
            train_records["gt_offsets"] - train_records["offsets"],
            l2, args.device,
        )
        train_global = global_residual(
            train_records["hidden"], calibration, args.device,
        ).cpu()
        remaining = (
            train_records["gt_offsets"] - train_records["offsets"]
            - train_global
        )
        for prior in parse_numbers(args.class_priors):
            class_bias = fit_class_bias(
                train_records["predicted_classes"], remaining, num_classes,
                prior, args.device,
            ).cpu()
            for gamma in parse_numbers(args.correction_scales):
                corrected = classwise_corrected_offsets(
                    val_records["hidden"], val_records["offsets"],
                    val_records["predicted_classes"], calibration, class_bias,
                    gamma, args.device,
                ).cpu()
                candidate = {
                    "l2": l2, "prior": prior, "gamma": gamma,
                    "metrics": metrics(corrected, val_records["gt_offsets"]),
                }
                candidates.append(candidate)
                if best is None or (
                    candidate["metrics"]["offset_mse"]
                    < best["metrics"]["offset_mse"]
                ):
                    best = candidate
                    best_calibration = calibration
                    best_class_bias = class_bias
    artifact = {
        "calibration": best_calibration,
        "class_bias": best_class_bias,
        "l2": best["l2"], "prior": best["prior"], "gamma": best["gamma"],
        "levels": levels, "config": args.config, "checkpoint": args.checkpoint,
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_only_classwise_signed_boundary_residual_selection",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_val_baseline": metrics(val_records["offsets"], val_records["gt_offsets"]),
        "source_val_corrected": best["metrics"],
        "selected": {key: best[key] for key in ("l2", "prior", "gamma")},
        "selection_metric": "source_val_offset_mse",
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location="cpu")
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations, val = load_model_and_data(args, "target_val")
    records = collect_records(
        model, val, annotations, cfg, "target", "val", artifact["levels"], args.device,
    )
    corrected = classwise_corrected_offsets(
        records["hidden"], records["offsets"], records["predicted_classes"],
        artifact["calibration"], artifact["class_bias"], artifact["gamma"],
        args.device,
    ).cpu()
    return {
        "audit": "fixed_source_only_classwise_signed_boundary_residual_target_evaluation",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "target_baseline": metrics(records["offsets"], records["gt_offsets"]),
        "target_corrected": metrics(corrected, records["gt_offsets"]),
        "fixed_source_selection": {
            key: artifact[key] for key in ("l2", "prior", "gamma")
        },
        "target_gt_usage": "offline evaluation only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_select", "target_eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--ridge-l2", default="0.1,1,10,100")
    parser.add_argument("--correction-scales", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--class-priors", default="10,50,100")
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    if args.mode == "source_select" and not args.calibration_output:
        parser.error("--calibration-output is required for source_select")
    if args.mode == "target_eval" and not args.calibration:
        parser.error("--calibration is required for target_eval")
    result = source_select(args) if args.mode == "source_select" else target_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
