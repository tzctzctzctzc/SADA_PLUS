"""Target-GT audit for frozen-path Stage2 boundary stability."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage2_utils import (
    aligned_boundary_iou_from_offsets,
    deterministic_channel_dropout_perturbation,
    deterministic_gaussian_perturbation,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def rank_values(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    sorted_values, order = torch.sort(values, stable=True)
    ranks = torch.empty_like(values)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def correlation(values_x, values_y, rank=False):
    x = torch.as_tensor(values_x, dtype=torch.float64)
    y = torch.as_tensor(values_y, dtype=torch.float64)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2:
        return None
    if rank:
        x, y = rank_values(x), rank_values(y)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.square().mean().sqrt() * y.square().mean().sqrt()
    if denominator == 0:
        return None
    return float((x * y).mean() / denominator)


def quantiles(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    if not values.numel():
        return {}
    probabilities = torch.tensor(
        [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0],
        dtype=torch.float64,
    )
    result = torch.quantile(values, probabilities)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p10", "p25", "p50", "p75", "p90", "p99", "max"),
            result,
        )
    }


def summarize_subset(records, indices, baseline_foreground):
    selected = [records[index] for index in indices]
    count = len(selected)
    foreground = sum(record["foreground"] for record in selected)
    class_correct = sum(record["class_correct"] for record in selected)
    good_boundary = sum(record["boundary_iou"] >= 0.5 for record in selected)
    correct_boundary = sum(
        record["foreground"]
        and record["class_correct"]
        and record["boundary_iou"] >= 0.5
        for record in selected
    )
    foreground_ious = [
        record["boundary_iou"]
        for record in selected
        if record["foreground"]
    ]
    return {
        "n": count,
        "retained_fraction": ratio(count, len(records)),
        "foreground_precision": ratio(foreground, count),
        "class_correct_precision": ratio(class_correct, count),
        "mean_boundary_iou_all": ratio(
            sum(record["boundary_iou"] for record in selected), count
        ),
        "mean_boundary_iou_given_foreground": ratio(
            sum(foreground_ious), len(foreground_ious)
        ),
        "iou50_precision_all": ratio(good_boundary, count),
        "iou50_rate_given_foreground": ratio(good_boundary, foreground),
        "correct_class_and_iou50_precision": ratio(correct_boundary, count),
        "selected_foreground_coverage": ratio(
            foreground, baseline_foreground
        ),
        "good_boundary_coverage_of_selected_foreground": ratio(
            good_boundary, baseline_foreground
        ),
    }


def summarize_ranking(records, score, retain_fractions):
    foreground = torch.tensor([
        record["foreground"] for record in records
    ], dtype=torch.bool)
    baseline_foreground = int(foreground.sum())
    descending = torch.argsort(score, descending=True, stable=True)
    crop_groups = {}
    for index, record in enumerate(records):
        crop_key = (record["video_id"], record["crop_start"])
        crop_groups.setdefault(crop_key, []).append(index)

    output = {"top_budget": {}, "top_budget_per_crop": {}}
    for fraction in retain_fractions:
        count = min(
            len(records),
            max(1, int(np.ceil(len(records) * fraction))),
        )
        fraction_key = "{:.2f}".format(fraction)
        output["top_budget"][fraction_key] = summarize_subset(
            records, descending[:count].tolist(), baseline_foreground
        )
        crop_indices = []
        for group_indices in crop_groups.values():
            group_count = min(
                len(group_indices),
                max(1, int(np.ceil(len(group_indices) * fraction))),
            )
            local_order = torch.argsort(
                score[group_indices], descending=True, stable=True
            )[:group_count]
            crop_indices.extend(
                group_indices[int(local_index)]
                for local_index in local_order
            )
        output["top_budget_per_crop"][fraction_key] = summarize_subset(
            records, crop_indices, baseline_foreground
        )
    return output


def summarize_noise(
    records, stability_key, thresholds, retain_fractions, confidence_powers
):
    stability = torch.tensor([
        record[stability_key] for record in records
    ], dtype=torch.float64)
    confidence = torch.tensor([
        record["confidence"] for record in records
    ], dtype=torch.float64)
    boundary_iou = torch.tensor([
        record["boundary_iou"] for record in records
    ], dtype=torch.float64)
    foreground = torch.tensor([
        record["foreground"] for record in records
    ], dtype=torch.bool)
    baseline_foreground = int(foreground.sum())
    weight_sum = stability.sum()
    normalized_ess = (
        weight_sum.square() / stability.square().sum().clamp_min(1e-12)
        / max(stability.numel(), 1)
    )
    output = {
        "n": len(records),
        "stability": {
            "mean": float(stability.mean()),
            "std": float(stability.std(unbiased=False)),
            "quantiles": quantiles(stability),
            "normalized_effective_sample_size": float(normalized_ess),
        },
        "correlation": {
            "all_anchors_pearson": correlation(stability, boundary_iou),
            "all_anchors_spearman": correlation(
                stability, boundary_iou, rank=True
            ),
            "foreground_only_pearson": correlation(
                stability[foreground], boundary_iou[foreground]
            ),
            "foreground_only_spearman": correlation(
                stability[foreground], boundary_iou[foreground], rank=True
            ),
        },
        "thresholds": {},
        "ranking_results": {
            "confidence": summarize_ranking(
                records, confidence, retain_fractions
            ),
            "stability": summarize_ranking(
                records, stability, retain_fractions
            ),
        },
        "stability_deciles_low_to_high": [],
        "per_fpn_level": {},
    }

    for threshold in thresholds:
        indices = torch.where(stability >= threshold)[0].tolist()
        output["thresholds"]["{:.3f}".format(threshold)] = summarize_subset(
            records, indices, baseline_foreground
        )

    for power in confidence_powers:
        if power == 0:
            continue
        key = "stability_x_confidence_pow_{:.3f}".format(power)
        output["ranking_results"][key] = summarize_ranking(
            records, stability * confidence.pow(power), retain_fractions
        )

    ascending = torch.argsort(stability, stable=True)
    for chunk in torch.tensor_split(ascending, 10):
        indices = chunk.tolist()
        item = summarize_subset(records, indices, baseline_foreground)
        item["mean_stability"] = (
            float(stability[chunk].mean()) if chunk.numel() else None
        )
        output["stability_deciles_low_to_high"].append(item)

    levels = sorted({record["level"] for record in records})
    for level in levels:
        indices = [
            index for index, record in enumerate(records)
            if record["level"] == level
        ]
        level_stability = stability[indices]
        level_iou = boundary_iou[indices]
        level_foreground = foreground[indices]
        output["per_fpn_level"][str(level)] = {
            "n": len(indices),
            "mean_stability": float(level_stability.mean()),
            "foreground_precision": float(level_foreground.float().mean()),
            "mean_boundary_iou_all": float(level_iou.mean()),
            "stability_vs_iou_spearman": correlation(
                level_stability, level_iou, rank=True
            ),
            "foreground_stability_vs_iou_spearman": correlation(
                level_stability[level_foreground],
                level_iou[level_foreground],
                rank=True,
            ),
        }
    return output


def run_frozen_path(model, inputs, masks):
    features, feature_masks, _ = model.backbone_pl(inputs, masks)
    fpn_features, fpn_masks = model.neck_pl(features, feature_masks)
    offsets = run_frozen_head(model, fpn_features, fpn_masks)
    return offsets, fpn_features, fpn_masks


def run_frozen_head(model, fpn_features, fpn_masks):
    """Run only the frozen localization head from already computed FPN features."""
    offsets, _ = model.reg_head_pl(fpn_features, fpn_masks)
    offsets = torch.cat([
        value.permute(0, 2, 1) for value in offsets
    ], dim=1)
    return offsets


def head_channel_dropout_features(
        fpn_features, fpn_masks, sample_keys, drop_probability, seed, epoch):
    """Perturb head inputs without changing the frozen backbone or FPN path."""
    return [
        deterministic_channel_dropout_perturbation(
            features,
            masks,
            ['{}:fpn{}'.format(key, level) for key in sample_keys],
            drop_probability,
            seed,
            epoch,
        )
        for level, (features, masks) in enumerate(zip(fpn_features, fpn_masks))
    ]


def run_current_classification(model, inputs, masks, domain):
    if domain == "source":
        backbone = model.backbone_source
        neck = model.neck_source
        classification_head = model.cls_head
    else:
        backbone = model.backbone_target
        neck = model.neck_target
        classification_head = model.cls_head_target
    features, feature_masks, _ = backbone(inputs, masks)
    fpn_features, fpn_masks = neck(features, feature_masks)
    logits, _ = classification_head(fpn_features, fpn_masks)
    logits = torch.cat([
        value.permute(0, 2, 1) for value in logits
    ], dim=1)
    return logits, fpn_features, fpn_masks


def audit(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    levels = parse_numbers(args.levels, int)
    noise_stds = parse_numbers(args.noise_stds, float)
    channel_drop_probabilities = parse_numbers(
        args.channel_drop_probabilities, float
    )
    perturbation_values = (
        noise_stds if args.perturbation == "gaussian"
        else channel_drop_probabilities
    )
    perturbation_seeds = parse_numbers(args.perturbation_seeds, int)
    thresholds = parse_numbers(args.thresholds, float)
    retain_fractions = parse_numbers(args.retain_fractions, float)
    confidence_powers = parse_numbers(args.confidence_powers, float)
    if (
        not levels
        or not perturbation_values
        or not perturbation_seeds
        or not confidence_powers
        or any(value <= 0 for value in noise_stds)
        or any(
            value <= 0 or value >= 1
            for value in channel_drop_probabilities
        )
        or any(value < 0 for value in confidence_powers)
    ):
        raise ValueError(
            "levels, perturbation values, perturbation seeds, and confidence "
            "powers "
            "are required"
        )

    split_key = "{}_{}_split".format(
        "src" if args.domain == "source" else "tgt",
        args.split,
    )
    split = cfg[split_key]
    dataset = make_dataset(
        cfg["dataset_name"], args.split == "train", split, **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(
        cfg["model_name"], cfg, device=args.device
    ).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    for module in (model.backbone_pl, model.neck_pl, model.reg_head_pl):
        module.eval()

    batch_size = args.batch_size or int(
        cfg.get("loader", {}).get("batch_size", 1)
    )
    stop = len(dataset)
    if args.max_videos is not None:
        stop = min(stop, args.max_videos)
    records = []
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    audit_epoch = (
        args.perturbation_epoch
        if args.perturbation_epoch is not None
        else int(checkpoint.get("epoch", 0))
    )

    with torch.no_grad():
        for batch_start in range(0, stop, batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, stop)
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            preprocessing_mode = (
                "train" if args.split == "train" else "validation"
            )
            inputs, masks = model.preprocessing(
                videos, mode=preprocessing_mode
            )
            logits, current_features, current_masks = (
                run_current_classification(
                    model, inputs, masks, args.domain
                )
            )
            pseudo_offsets, pseudo_fpn_features, pseudo_masks = run_frozen_path(
                model, inputs, masks
            )
            points = model.point_generator(current_features)
            valid = torch.cat([
                value.squeeze(1) for value in current_masks
            ], dim=1)
            pseudo_valid = torch.cat([
                value.squeeze(1) for value in pseudo_masks
            ], dim=1)
            if not torch.equal(valid, pseudo_valid):
                raise RuntimeError("current and frozen paths have different masks")
            level_mask = torch.cat([
                torch.full(
                    (len(videos), point.shape[0]),
                    level in levels,
                    dtype=torch.bool,
                    device=args.device,
                )
                for level, point in enumerate(points)
            ], dim=1)
            confidence, pseudo_class = logits.sigmoid().max(dim=-1)
            selected = valid & level_mask & (confidence >= alpha)

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = raw_video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    args.device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, gt_offsets = model.label_points(
                points, gt_segments, gt_labels
            )
            gt_cls = torch.stack(gt_cls)
            gt_offsets = torch.stack(gt_offsets)
            foreground = gt_cls.sum(dim=-1) > 0
            class_correct = gt_cls.gather(
                -1, pseudo_class.unsqueeze(-1)
            ).squeeze(-1) > 0
            boundary_iou = aligned_boundary_iou_from_offsets(
                pseudo_offsets, gt_offsets
            )
            boundary_iou = torch.where(
                foreground, boundary_iou, torch.zeros_like(boundary_iou)
            )

            batch_stability = {}
            sample_keys = [
                "{}:{}".format(
                    video["metadata"].get("video_id", index),
                    video["metadata"].get("crop_start", 0),
                )
                for index, video in enumerate(videos)
            ]
            for perturbation_value in perturbation_values:
                seed_stabilities = []
                for perturbation_seed in perturbation_seeds:
                    if args.perturbation == "gaussian":
                        perturbed = deterministic_gaussian_perturbation(
                            inputs,
                            masks,
                            sample_keys,
                            perturbation_value,
                            perturbation_seed,
                            audit_epoch,
                        )
                        perturbed_offsets, _, _ = run_frozen_path(
                            model, perturbed, masks
                        )
                    elif args.perturbation == "channel_dropout":
                        perturbed = deterministic_channel_dropout_perturbation(
                            inputs,
                            masks,
                            sample_keys,
                            perturbation_value,
                            perturbation_seed,
                            audit_epoch,
                        )
                        perturbed_offsets, _, _ = run_frozen_path(
                            model, perturbed, masks
                        )
                    else:
                        perturbed_features = head_channel_dropout_features(
                            pseudo_fpn_features,
                            pseudo_masks,
                            sample_keys,
                            perturbation_value,
                            perturbation_seed,
                            audit_epoch,
                        )
                        perturbed_offsets = run_frozen_head(
                            model, perturbed_features, pseudo_masks
                        )
                    seed_stabilities.append(
                        aligned_boundary_iou_from_offsets(
                            pseudo_offsets, perturbed_offsets
                        )
                    )
                batch_stability[perturbation_value] = torch.stack(
                    seed_stabilities
                ).mean(dim=0)

            level_ids = torch.cat([
                torch.full(
                    (point.shape[0],), level,
                    dtype=torch.long, device=args.device,
                )
                for level, point in enumerate(points)
            ])
            for batch_index, anchor_index in zip(
                *torch.where(selected)
            ):
                record = {
                    "video_id": videos[int(batch_index)]["metadata"].get(
                        "video_id"
                    ),
                    "crop_start": float(
                        videos[int(batch_index)]["metadata"].get(
                            "crop_start", 0
                        )
                    ),
                    "anchor_index": int(anchor_index),
                    "level": int(level_ids[anchor_index]),
                    "confidence": float(
                        confidence[batch_index, anchor_index]
                    ),
                    "foreground": bool(
                        foreground[batch_index, anchor_index]
                    ),
                    "class_correct": bool(
                        class_correct[batch_index, anchor_index]
                    ),
                    "boundary_iou": float(
                        boundary_iou[batch_index, anchor_index]
                    ),
                }
                for perturbation_value in perturbation_values:
                    record["stability_{:.6f}".format(perturbation_value)] = float(
                        batch_stability[perturbation_value][
                            batch_index, anchor_index
                        ]
                    )
                records.append(record)

    output = {
        "audit": "frozen_path_stage2_stability_noise_target_gt",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": args.config,
        "domain": args.domain,
        "split": args.split,
        "videos": stop,
        "selected_anchors": len(records),
        "protocol": {
            "routing": "current classification confidence >= alpha",
            "alpha": alpha,
            "levels": levels,
            "pseudo_boundary": "checkpoint frozen backbone_pl/neck_pl/reg_head_pl",
            "perturbation": {
                "gaussian": "sample-specific deterministic Gaussian input noise",
                "channel_dropout": (
                    "sample-specific deterministic input-channel dropout"
                ),
                "head_channel_dropout": (
                    "sample-specific deterministic FPN-channel dropout before "
                    "the frozen regression head"
                ),
            }[args.perturbation],
            "perturbation_type": args.perturbation,
            "noise_stds": noise_stds,
            "channel_drop_probabilities": channel_drop_probabilities,
            "perturbation_values": perturbation_values,
            "perturbation_seeds": perturbation_seeds,
            "confidence_powers": confidence_powers,
            "perturbation_epoch": audit_epoch,
            "stability": (
                "mean same-anchor boundary IoU across perturbation seeds"
            ),
            "target_gt_usage": "offline audit only",
        },
        "baseline_selected_anchors": summarize_subset(
            records,
            list(range(len(records))),
            sum(record["foreground"] for record in records),
        ),
        "noise_results": {},
    }
    for perturbation_value in perturbation_values:
        key = "stability_{:.6f}".format(perturbation_value)
        output["noise_results"]["{:.6f}".format(perturbation_value)] = (
            summarize_noise(
                records,
                key,
                thresholds,
                retain_fractions,
                confidence_powers,
            )
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--domain", choices=("source", "target"), default="target"
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument(
        "--noise-stds", default="0.005,0.01,0.02,0.05,0.1"
    )
    parser.add_argument(
        "--perturbation",
        choices=("gaussian", "channel_dropout", "head_channel_dropout"),
        default="gaussian",
    )
    parser.add_argument(
        "--channel-drop-probabilities", default="0.05,0.1,0.2,0.3"
    )
    parser.add_argument(
        "--perturbation-seeds",
        default="20260730,20260731,20260732,20260733",
    )
    parser.add_argument("--perturbation-epoch", type=int, default=None)
    parser.add_argument(
        "--thresholds", default="0.2,0.5,0.7,0.8,0.9,0.95,0.98,0.99"
    )
    parser.add_argument(
        "--retain-fractions", default="0.25,0.5,0.75,1.0"
    )
    parser.add_argument("--confidence-powers", default="0,0.5,1,2")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    output = audit(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "checkpoint_epoch": output["checkpoint_epoch"],
        "videos": output["videos"],
        "selected_anchors": output["selected_anchors"],
        "baseline": output["baseline_selected_anchors"],
        "noise_results": {
            key: {
                "stability": value["stability"],
                "correlation": value["correlation"],
            }
            for key, value in output["noise_results"].items()
        },
    }, indent=2))
    print("wrote", output_path)


if __name__ == "__main__":
    main()
