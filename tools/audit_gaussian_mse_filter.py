"""Audit a locked source-calibrated Gaussian MSE filter on target GT."""

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
from tools.audit_original_sada_pseudolabels import move_video


def _rate(mask, values):
    return float(values[mask].float().mean()) if mask.any() else None


def _summary(route, labels):
    correct = labels.gather(
        -1, route["pseudo_class"].unsqueeze(-1)
    ).squeeze(-1) > 0
    gt_action = labels.sum(dim=-1) > 0
    candidate = route["action_candidate"]
    reliable = route["reliable_action"]
    rejected = candidate & ~reliable
    background = route["stable_background"]
    return {
        "valid_n": int(route["valid"].sum()),
        "action_candidate": {
            "n": int(candidate.sum()),
            "correct_class_rate": _rate(candidate, correct),
            "gt_action_rate": _rate(candidate, gt_action),
        },
        "reliable_action": {
            "n": int(reliable.sum()),
            "correct_class_rate": _rate(reliable, correct),
            "gt_action_rate": _rate(reliable, gt_action),
        },
        "rejected_high_confidence": {
            "n": int(rejected.sum()),
            "correct_class_rate": _rate(rejected, correct),
            "gt_action_rate": _rate(rejected, gt_action),
        },
        "background": {
            "n": int(background.sum()),
            "gt_action_rate": _rate(background, gt_action),
        },
    }


def _merge_items(items):
    merged = {
        key: torch.cat([item[key] for item in items])
        for key in items[0]
    }
    labels = merged.pop("labels")
    return _summary(merged, labels)


def _merge(records):
    return {
        str(level): _merge_items(items)
        for level, items in records.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    calibration = json.loads(
        Path(args.calibration).read_text(encoding="utf-8")
    )
    cfg = read_yaml(args.config)
    cfg["stage1"] = {
        "enabled": True,
        "perturbation": "gaussian_mse",
        "gaussian_std": 0.05,
        "perturbation_seeds": [20260730, 20260731, 20260732, 20260733],
        "levels": [int(level) for level in calibration["levels"]],
        "gaussian_mse_calibration": calibration,
    }
    _update_config(cfg)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    if calibration.get("method") != "confidence_then_gaussian_mse":
        raise ValueError("calibration artifact has the wrong method")
    if abs(float(calibration["alpha"]) - alpha) > 1e-8:
        raise ValueError("calibration alpha does not match the config")

    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()

    records = {level: [] for level in range(model.n_levels)}
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            videos = [
                move_video(dataset[index], args.device)
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            routes, signals = model._stage1_gaussian_routing(
                videos, return_signals=True
            )
            points = model.point_generator(signals["fpn"])
            gt_cls, _ = model.label_points(
                points,
                [video["segments"] for video in videos],
                [video["labels"] for video in videos],
            )
            gt_cls = torch.stack(gt_cls)
            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                labels = gt_cls[:, start:start + length]
                start += length
                route = routes[level]
                records[level].append({
                    "labels": labels.detach().cpu().reshape(
                        -1, labels.shape[-1]
                    ),
                    **{
                        key: value.detach().cpu().flatten()
                        for key, value in route.items()
                        if isinstance(value, torch.Tensor)
                    },
                })

    enabled_levels = [int(level) for level in calibration["levels"]]
    result = {
        "audit": "locked_source_calibrated_gaussian_mse_filter",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "calibration": args.calibration,
        "crop_seed": args.seed,
        "target_gt_role": "post_hoc_audit_only",
        "videos": len(dataset),
        "per_level": _merge(records),
        "enabled_levels": enabled_levels,
        "enabled_levels_total": _merge_items([
            item for level in enabled_levels for item in records[level]
        ]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
