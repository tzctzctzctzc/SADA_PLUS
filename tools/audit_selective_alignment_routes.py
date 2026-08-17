"""Audit selective-alignment routes against target GT without training on it."""

import argparse
import json
import random
from pathlib import Path

import torch


def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_group(records, mask, denominator=None):
    mask = mask.bool()
    count = int(mask.sum().item())
    gt_action = records["gt_action"]
    correct = records["correct_class"]
    denominator = mask.numel() if denominator is None else denominator
    return {
        "n": count,
        "selection_rate": _rate(count, denominator),
        "gt_action_rate": _rate(
            int((mask & gt_action).sum().item()), count
        ),
        "gt_background_rate": _rate(
            int((mask & ~gt_action).sum().item()), count
        ),
        "correct_class_rate": _rate(
            int((mask & correct).sum().item()), count
        ),
    }


def summarize_weight_quantiles(records, mask, weight, bins=5):
    selected = torch.where(mask.bool())[0]
    if selected.numel() == 0:
        return []
    order = torch.argsort(weight[selected])
    chunks = torch.tensor_split(selected[order], min(bins, selected.numel()))
    output = []
    for index, chunk in enumerate(chunks):
        chunk_mask = torch.zeros_like(mask, dtype=torch.bool)
        chunk_mask[chunk] = True
        summary = summarize_group(records, chunk_mask, selected.numel())
        values = weight[chunk]
        summary.update({
            "quantile": index + 1,
            "weight_min": float(values.min().item()),
            "weight_mean": float(values.mean().item()),
            "weight_max": float(values.max().item()),
        })
        output.append(summary)
    return output


def summarize_selector_records(records, alpha):
    ordinary_valid = records["ordinary_valid"].bool()
    strict_valid = records["strict_valid"].bool()
    gt_action = records["gt_action"].bool()
    correct = records["correct_class"].bool()
    baseline_action = ordinary_valid & (records["score"] >= alpha)
    baseline_background = ordinary_valid & ~baseline_action
    strict_action = strict_valid & baseline_action
    strict_background = strict_valid & baseline_background
    reliable_action = records["reliable_action"].bool()
    stable_background = records["stable_background"].bool()
    ignored = strict_valid & ~(reliable_action | stable_background)
    removed_action = baseline_action & ~reliable_action
    removed_background = baseline_background & ~stable_background

    false_action = baseline_action & ~gt_action
    correct_action = baseline_action & correct
    missed_action = baseline_background & gt_action
    true_background = baseline_background & ~gt_action
    denominator = int(ordinary_valid.sum().item())
    groups = {
        "ordinary_valid": ordinary_valid,
        "strict_valid": strict_valid,
        "edge_excluded": ordinary_valid & ~strict_valid,
        "baseline_action": baseline_action,
        "baseline_background": baseline_background,
        "strict_baseline_action": strict_action,
        "strict_baseline_background": strict_background,
        "reliable_action": reliable_action,
        "stable_background": stable_background,
        "ignored_strict_valid": ignored,
        "removed_action": removed_action,
        "removed_background": removed_background,
        "class_flip_action": strict_action & ~records["class_consistent"].bool(),
        "unstable_consistent_action": (
            strict_action
            & records["class_consistent"].bool()
            & ~reliable_action
        ),
        "unstable_low_background": (
            strict_background & ~stable_background
        ),
    }
    summary = {
        "groups": {
            name: summarize_group(records, mask, denominator)
            for name, mask in groups.items()
        },
        "error_handling": {
            "false_action_n": int(false_action.sum().item()),
            "false_action_removal_recall": _rate(
                int((false_action & removed_action).sum().item()),
                int(false_action.sum().item()),
            ),
            "correct_action_n": int(correct_action.sum().item()),
            "correct_action_retention": _rate(
                int((correct_action & reliable_action).sum().item()),
                int(correct_action.sum().item()),
            ),
            "baseline_background_missed_action_n": int(
                missed_action.sum().item()
            ),
            "missed_action_removal_recall": _rate(
                int((missed_action & removed_background).sum().item()),
                int(missed_action.sum().item()),
            ),
            "baseline_true_background_n": int(true_background.sum().item()),
            "true_background_retention": _rate(
                int((true_background & stable_background).sum().item()),
                int(true_background.sum().item()),
            ),
        },
        "action_weight_quantiles": summarize_weight_quantiles(
            records,
            reliable_action,
            records["action_weight"],
        ),
        "background_weight_quantiles": summarize_weight_quantiles(
            records,
            stable_background,
            records["background_weight"],
        ),
    }
    return summary


def parse_levels(value):
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("levels must be a non-empty unique comma-separated list")
    return levels


def main():
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.stage1_utils import (
        build_soft_selective_alignment_weights,
    )
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import move_video

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--weights",
        choices=("state_dict", "ema_state_dict"),
        default="state_dict",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    levels = parse_levels(args.levels)

    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    required = ("beta", "tau_act", "tau_bg")
    if any(key not in calibration for key in required):
        raise ValueError("calibration must contain beta, tau_act, and tau_bg")

    cfg = read_yaml(args.config)
    cfg["stage1"] = {
        "enabled": True,
        "perturbation": "shift",
        "seed": args.seed + 100003,
        "calibration_seed": args.seed + 100004,
        **{key: calibration[key] for key in required},
    }
    cfg.setdefault("diagnostic", {})["selective_alignment"] = "off"
    _update_config(cfg)

    split = (
        cfg["tgt_train_split"]
        if args.split == "train"
        else cfg["tgt_val_split"]
    )
    dataset = make_dataset(
        cfg["dataset_name"], args.split == "train", split, **cfg["dataset"]
    )
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if args.weights not in checkpoint:
        raise ValueError("checkpoint has no {}".format(args.weights))
    model.load_state_dict(checkpoint[args.weights])
    model.eval()
    route_rng = torch.Generator(device="cpu")
    route_rng.manual_seed(args.seed + 100003)

    record_lists = {
        key: []
        for key in (
            "level",
            "ordinary_valid",
            "strict_valid",
            "score",
            "gt_action",
            "correct_class",
            "reliable_action",
            "stable_background",
            "class_consistent",
            "action_weight",
            "background_weight",
            "action_std",
            "background_std",
        )
    }

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            inputs, masks = model.preprocessing(videos, mode="validation")
            feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(feats, backbone_masks)
            points = model.point_generator(fpn_feats)
            routes = model._stage1_shift_routing(
                videos, points, rng=route_rng
            )
            gt_cls, _ = model.label_points(
                points,
                [video["segments"] for video in videos],
                [video["labels"] for video in videos],
            )
            gt_cls = torch.stack(gt_cls)

            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                if level not in levels:
                    start += length
                    continue
                route = routes[level]
                weights = build_soft_selective_alignment_weights(
                    route,
                    model.stage1_tau_act,
                    model.stage1_tau_bg,
                )
                labels = gt_cls[:, start:start + length]
                pseudo_class = route["pseudo_class"]
                values = {
                    "level": torch.full_like(pseudo_class, level),
                    "ordinary_valid": fpn_masks[level].squeeze(1).bool(),
                    "strict_valid": route["valid"].bool(),
                    "score": route["score_original"],
                    "gt_action": labels.sum(dim=-1) > 0,
                    "correct_class": labels.gather(
                        -1, pseudo_class.unsqueeze(-1)
                    ).squeeze(-1) > 0,
                    "reliable_action": route["reliable_action"],
                    "stable_background": route["stable_background"],
                    "class_consistent": route["class_consistent"],
                    "action_weight": weights["action"],
                    "background_weight": weights["background"],
                    "action_std": route["action_std"],
                    "background_std": route["background_std"],
                }
                for key, value in values.items():
                    record_lists[key].append(value.reshape(-1).cpu())
                start += length

    records = {
        key: torch.cat(values) for key, values in record_lists.items()
    }
    per_level = {}
    for level in levels:
        selected = records["level"] == level
        per_level[str(level)] = summarize_selector_records(
            {key: value[selected] for key, value in records.items()},
            model.stage1_alpha,
        )

    result = {
        "audit": "selective_alignment_target_gt_routes",
        "target_gt_usage": "offline_reporting_only",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "calibration": args.calibration,
        "thresholds": {
            "alpha": model.stage1_alpha,
            **{key: float(calibration[key]) for key in required},
        },
        "config": args.config,
        "split": "target_{}".format(args.split),
        "seed": args.seed,
        "route_seed": args.seed + 100003,
        "levels": list(levels),
        "videos": len(dataset),
        "batch_size": batch_size,
        "comparison_scope": (
            "baseline and selective routes on identical frozen-model crops; "
            "strict-valid and edge-excluded anchors are reported separately"
        ),
        "global": summarize_selector_records(records, model.stage1_alpha),
        "per_level": per_level,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["global"], indent=2))
    print("wrote", output_path)


if __name__ == "__main__":
    main()
