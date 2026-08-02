"""Audit frozen TAL structure as foreground evidence for low-confidence anchors.

All candidate classes and structure scores are frozen before target GT is used.
Target GT is read only to report routing metrics.
"""

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.audit_low_confidence_topk_consensus import (
    candidate_correct,
    load_models,
    ratio,
    rule_summary,
)


RECOVERY_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20)


def interpolate_probabilities(probabilities, valid_mask, points, query):
    grid = points[:, 0]
    insertion = torch.searchsorted(grid.contiguous(), query.contiguous())
    exact_index = insertion.clamp_max(grid.numel() - 1)
    exact = (
        (insertion < grid.numel())
        & torch.isclose(grid[exact_index], query, rtol=0.0, atol=1e-6)
    )
    right = insertion.clamp(1, grid.numel() - 1)
    left = right - 1
    weight = (query - grid[left]) / (grid[right] - grid[left]).clamp_min(1e-12)
    interpolated = probabilities[:, left] + weight[None, :, None] * (
        probabilities[:, right] - probabilities[:, left]
    )
    output = torch.where(
        exact[None, :, None], probabilities[:, exact_index], interpolated
    )
    endpoint_valid = valid_mask[:, left] & valid_mask[:, right]
    valid = torch.where(
        exact[None], valid_mask[:, exact_index], endpoint_valid
    ) & (query >= grid[0])[None] & (query <= grid[-1])[None]
    return output, valid


def shifted(values, delta, fill_value=0):
    """Return values at i + delta without temporal wraparound."""
    output = torch.full_like(values, fill_value)
    length = values.shape[1]
    if delta < 0:
        output[:, -delta:] = values[:, :length + delta]
    elif delta > 0:
        output[:, :length - delta] = values[:, delta:]
    else:
        output.copy_(values)
    return output


def segment_iou(first, second):
    intersection = (
        torch.minimum(first[..., 1], second[..., 1])
        - torch.maximum(first[..., 0], second[..., 0])
    ).clamp_min(0)
    union = (
        torch.maximum(first[..., 1], second[..., 1])
        - torch.minimum(first[..., 0], second[..., 0])
    ).clamp_min(1e-6)
    return intersection / union


def same_level_scores(probabilities, offsets, points, valid, radius):
    """Compute local semantic continuity and decoded-segment agreement."""
    if offsets.shape[-1] != 2:
        raise ValueError("structure audit requires direct two-sided offsets")
    candidate = probabilities.argmax(dim=-1)
    centers = points[:, 0].view(1, -1).expand(probabilities.shape[0], -1)
    strides = points[:, 3].view(1, -1)
    segments = torch.stack((
        centers - offsets[..., 0] * strides,
        centers + offsets[..., 1] * strides,
    ), dim=-1)

    continuity_sum = torch.zeros_like(centers)
    segment_vote_sum = torch.zeros_like(centers)
    segment_iou_sum = torch.zeros_like(centers)
    neighbor_count = torch.zeros_like(centers)
    for delta in range(-radius, radius + 1):
        if delta == 0:
            continue
        neighbor_valid = shifted(valid, delta, False)
        neighbor_probability = shifted(probabilities, delta, 0.0)
        neighbor_candidate = shifted(candidate, delta, -1)
        neighbor_segments = shifted(segments, delta, 0.0)
        candidate_probability = neighbor_probability.gather(
            -1, candidate.unsqueeze(-1)
        ).squeeze(-1)
        same_class = neighbor_candidate == candidate
        contains_center = (
            (neighbor_segments[..., 0] <= centers)
            & (neighbor_segments[..., 1] >= centers)
        )
        valid_float = neighbor_valid.float()
        continuity_sum += candidate_probability * valid_float
        segment_vote_sum += (
            candidate_probability * same_class * contains_center * valid_float
        )
        segment_iou_sum += (
            segment_iou(segments, neighbor_segments)
            * candidate_probability
            * same_class
            * valid_float
        )
        neighbor_count += valid_float

    denominator = neighbor_count.clamp_min(1.0)
    has_neighbors = neighbor_count > 0
    return {
        "local_continuity_r{}".format(radius): continuity_sum / denominator,
        "segment_vote_r{}".format(radius): segment_vote_sum / denominator,
        "segment_iou_r{}".format(radius): segment_iou_sum / denominator,
        "local_has_neighbors_r{}".format(radius): has_neighbors.float(),
    }


def cross_level_scores(probabilities, masks, points, levels):
    """Measure support for each anchor's class at the same absolute time."""
    output = {}
    for level in levels:
        candidate = probabilities[level].argmax(dim=-1)
        query = points[level][:, 0]
        support_sum = torch.zeros_like(candidate, dtype=probabilities[level].dtype)
        agreement_sum = torch.zeros_like(support_sum)
        valid_count = torch.zeros_like(support_sum)
        for other_level in levels:
            if other_level == level:
                continue
            aligned, aligned_valid = interpolate_probabilities(
                probabilities[other_level],
                masks[other_level],
                points[other_level],
                query,
            )
            candidate_support = aligned.gather(
                -1, candidate.unsqueeze(-1)
            ).squeeze(-1)
            agreement = aligned.argmax(dim=-1) == candidate
            support_sum += candidate_support * aligned_valid
            agreement_sum += agreement * aligned_valid
            valid_count += aligned_valid
        denominator = valid_count.clamp_min(1.0)
        output[level] = {
            "cross_level_support": support_sum / denominator,
            "cross_level_agreement": agreement_sum / denominator,
            "cross_level_has_support": (valid_count > 0).float(),
        }
    return output


def equal_frequency_bins(scores, levels, universe, bin_count=4):
    bins = torch.full_like(levels, -1)
    for level in levels.unique(sorted=True):
        indices = torch.where(universe & (levels == level))[0]
        if not indices.numel():
            continue
        order = indices[torch.argsort(scores[indices], stable=True)]
        rank = torch.arange(order.numel(), device=order.device)
        bins[order] = torch.clamp(rank * bin_count // order.numel(), max=bin_count - 1)
    return bins


def percentile_ranks(values, levels, bins, universe):
    ranks = torch.zeros_like(values)
    for level in levels.unique(sorted=True):
        for confidence_bin in bins[universe].unique(sorted=True):
            indices = torch.where(
                universe & (levels == level) & (bins == confidence_bin)
            )[0]
            if not indices.numel():
                continue
            order = indices[torch.argsort(values[indices], stable=True)]
            if order.numel() == 1:
                ranks[order] = 1.0
            else:
                ranks[order] = torch.arange(
                    order.numel(), dtype=values.dtype, device=values.device
                ) / (order.numel() - 1)
    return ranks


def exact_top_fraction(universe, ranking, fraction):
    selected = torch.zeros_like(universe)
    indices = torch.where(universe)[0]
    if not indices.numel():
        return selected
    count = max(1, int(round(indices.numel() * fraction)))
    order = torch.argsort(ranking[indices], descending=True, stable=True)
    selected[indices[order[:count]]] = True
    return selected


def grouped_top_fraction(universe, ranking, levels, bins, fraction):
    selected = torch.zeros_like(universe)
    for level in levels.unique(sorted=True):
        for confidence_bin in bins[universe].unique(sorted=True):
            group = universe & (levels == level) & (bins == confidence_bin)
            selected |= exact_top_fraction(group, ranking, fraction)
    return selected


def evaluate_ranking(
    ranking,
    universe,
    correct,
    low_action,
    low_background,
    baseline_correct,
    all_gt_action_count,
    levels,
    bins,
):
    output = {"global": {}, "confidence_stratified": {}, "within_bin_top_10pct": {}}
    for fraction in RECOVERY_FRACTIONS:
        key = "top_{:g}pct".format(fraction * 100)
        output["global"][key] = rule_summary(
            exact_top_fraction(universe, ranking, fraction),
            correct, low_action, low_background,
            baseline_correct, all_gt_action_count,
        )
        output["confidence_stratified"][key] = rule_summary(
            grouped_top_fraction(universe, ranking, levels, bins, fraction),
            correct, low_action, low_background,
            baseline_correct, all_gt_action_count,
        )
    for confidence_bin in range(4):
        group = universe & (bins == confidence_bin)
        output["within_bin_top_10pct"][str(confidence_bin)] = rule_summary(
            exact_top_fraction(group, ranking, 0.10),
            correct, low_action, low_background,
            baseline_correct, all_gt_action_count,
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-key", default="state_dict")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--primary-radius", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if args.primary_radius not in args.radii:
        raise ValueError("primary radius must be included in radii")

    from libs.datasets import make_dataset
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import crop_targets, move_video

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.load(
        open(cfg["dataset"]["json_file"], encoding="utf-8")
    )["database"]
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    video_count = min(len(dataset), args.max_videos or len(dataset))
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    models, metadata = load_models(
        cfg, [("base", args.checkpoint, args.state_key)], args.device
    )
    model = models["base"]
    probability_records = []
    score_records = {}
    gt_records = []
    level_records = []

    with torch.inference_mode():
        for batch_start in range(0, video_count, batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, video_count)
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            inputs, masks = model.preprocessing(videos, mode="validation")
            backbone, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn, fpn_masks = model.neck_target(backbone, backbone_masks)
            logits, _ = model.cls_head_target(fpn, fpn_masks)
            offsets, _ = model.reg_head_target(fpn, fpn_masks)
            probabilities = [value.permute(0, 2, 1).sigmoid() for value in logits]
            offsets = [value.permute(0, 2, 1) for value in offsets]
            points = model.point_generator(fpn)

            batch_scores = {level: {} for level in args.levels}
            for level in args.levels:
                for radius in args.radii:
                    batch_scores[level].update(same_level_scores(
                        probabilities[level], offsets[level], points[level],
                        fpn_masks[level].squeeze(1), radius,
                    ))
            cross_scores = cross_level_scores(
                probabilities, [mask.squeeze(1) for mask in fpn_masks],
                points, args.levels,
            )
            for level in args.levels:
                batch_scores[level].update(cross_scores[level])

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id], video["metadata"],
                    video["feats"].shape[-1], cfg, args.device,
                )
                if not torch.equal(segments, raw_video["segments"].to(args.device)) or not torch.equal(
                    labels, raw_video["labels"].to(args.device)
                ):
                    raise RuntimeError(
                        "target GT reconstruction mismatch for {}".format(video_id)
                    )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls).bool()

            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                if level in args.levels:
                    valid = fpn_masks[level].squeeze(1)
                    probability_records.append(probabilities[level][valid].cpu())
                    gt_records.append(gt_cls[:, start:start + length][valid].cpu())
                    level_records.append(torch.full(
                        (int(valid.sum().item()),), level, dtype=torch.long
                    ))
                    for name, values in batch_scores[level].items():
                        score_records.setdefault(name, []).append(values[valid].cpu())
                start += length

    probabilities = torch.cat(probability_records)
    gt_labels = torch.cat(gt_records)
    levels = torch.cat(level_records)
    scores = {name: torch.cat(values) for name, values in score_records.items()}
    base_score, base_class = probabilities.max(dim=-1)
    gt_action = gt_labels.sum(dim=-1) > 0
    gt_background = ~gt_action
    base_high = base_score >= alpha
    universe = ~base_high
    low_action = universe & gt_action
    low_background = universe & gt_background
    correct = candidate_correct(base_class, gt_labels)
    baseline_correct = int((base_high & correct).sum().item())
    all_gt_action_count = int(gt_action.sum().item())
    confidence_bins = equal_frequency_bins(base_score, levels, universe)

    primary_names = [
        "local_continuity_r{}".format(args.primary_radius),
        "segment_vote_r{}".format(args.primary_radius),
        "cross_level_support",
    ]
    normalized = [
        percentile_ranks(scores[name], levels, confidence_bins, universe)
        for name in primary_names
    ]
    scores["primary_equal_weight_joint"] = torch.stack(normalized).mean(dim=0)

    ranking_results = {
        "confidence": evaluate_ranking(
            base_score, universe, correct, low_action, low_background,
            baseline_correct, all_gt_action_count, levels, confidence_bins,
        )
    }
    for name, values in scores.items():
        if name.startswith("local_has_") or name == "cross_level_has_support":
            continue
        ranking_results[name] = evaluate_ranking(
            values, universe, correct, low_action, low_background,
            baseline_correct, all_gt_action_count, levels, confidence_bins,
        )

    result = {
        "audit": "low_confidence_tal_foreground_structure",
        "seed": args.seed,
        "videos": video_count,
        "levels": args.levels,
        "radii": args.radii,
        "primary_radius": args.primary_radius,
        "alpha": alpha,
        "checkpoint": metadata["base"],
        "protocol": {
            "cohort": "base max class score below alpha on valid target-train anchors",
            "candidate_class": "frozen base-view top-1; GT never selects a class",
            "primary_scores": primary_names,
            "joint": "equal mean of within-level and confidence-quartile percentile ranks",
            "target_gt": "read only after frozen probabilities, offsets, and structure scores",
            "radius_selection": "radius 2 preregistered; radii 1 and 4 are sensitivity controls",
        },
        "cohort_counts": {
            "valid": int(probabilities.shape[0]),
            "all_gt_action": all_gt_action_count,
            "all_gt_background": int(gt_background.sum().item()),
            "base_low_gt_action": int(low_action.sum().item()),
            "base_low_gt_background": int(low_background.sum().item()),
            "base_high_correct_action": baseline_correct,
            "base_correct_da_utilization": ratio(baseline_correct, all_gt_action_count),
        },
        "coverage_checks": {
            name: ratio(int((values.bool() & universe).sum().item()), int(universe.sum().item()))
            for name, values in scores.items()
            if name.startswith("local_has_") or name == "cross_level_has_support"
        },
        "rankings": ranking_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "cohort_counts": result["cohort_counts"],
        "coverage_checks": result["coverage_checks"],
        "top_10pct_global": {
            name: values["global"]["top_10pct"]
            for name, values in ranking_results.items()
            if name in ("confidence", *primary_names, "primary_equal_weight_joint")
        },
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
