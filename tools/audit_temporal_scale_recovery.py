"""Audit temporal-scale cues for Stage1 pseudo-label filtering and recovery.

The frozen model sees the original target crop plus two fixed temporal spans.
Target GT is joined only after predictions are detached, and is used solely to
measure whether the perturbation exposes errors or recovers missed actions.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_utils import (
    interpolate_probabilities,
    map_original_points_to_scale_view,
    population_std,
    temporal_scale_view_on_cpu,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_context_action_stability import (
    build_exact_rank_retention,
    build_score_stratified_retention,
)
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_shift_action_stability import build_summary


RECOVERY_FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def _selected_summary(selected, universe, gt_action, correct):
    count = int(selected.sum().item())
    universe_n = int(universe.sum().item())
    gt_total = int(gt_action[universe].sum().item())
    return {
        "n": count,
        "selection_rate_within_low_confidence": count / max(universe_n, 1),
        "gt_action_rate": (
            float(gt_action[selected].float().mean().item()) if count else None
        ),
        "correct_class_rate": (
            float(correct[selected].float().mean().item()) if count else None
        ),
        "gt_action_coverage": (
            int(gt_action[selected].sum().item()) / max(gt_total, 1)
        ),
        "correct_class_coverage": (
            int(correct[selected].sum().item()) / max(gt_total, 1)
        ),
    }


def _top_k(universe, ranking, keep):
    indices = torch.where(universe)[0]
    selected = torch.zeros_like(universe)
    if not indices.numel():
        return selected
    keep = min(max(1, keep), int(indices.numel()))
    order = torch.argsort(ranking[indices], descending=True)
    selected[indices[order[:keep]]] = True
    return selected


def build_low_confidence_summary(records):
    score = records["score"]
    alpha = records["alpha"]
    gt_action = records["gt_action"]
    low = score < alpha
    low_n = int(low.sum().item())
    low_gt_n = int(gt_action[low].sum().item())
    any_cross = low & (records["any_recovery_score"] >= alpha)
    consensus_cross = (
        low
        & records["scale_consistent"]
        & (records["consensus_recovery_score"] >= alpha)
    )

    result = {
        "low_confidence": {
            "n": low_n,
            "gt_action_n": low_gt_n,
            "gt_action_rate": (
                float(gt_action[low].float().mean().item()) if low_n else None
            ),
        },
        "operational_threshold": {
            "any_scale_crosses_alpha": _selected_summary(
                any_cross, low, gt_action, records["any_recovery_correct"]
            ),
            "two_scale_consensus_crosses_alpha": _selected_summary(
                consensus_cross,
                low,
                gt_action,
                records["consensus_recovery_correct"],
            ),
        },
        "scale_consistent_n": int(
            (low & records["scale_consistent"]).sum().item()
        ),
        "matched_budget": {},
    }

    for fraction in RECOVERY_FRACTIONS:
        keep = max(1, int(math.ceil(fraction * low_n))) if low_n else 0
        baseline = _top_k(low, score, keep)
        any_scale = _top_k(low, records["any_recovery_score"], keep)
        consensus = _top_k(
            low, records["consensus_recovery_ranking"], keep
        )
        representation = _top_k(
            low, records["representation_mse"], keep
        )
        key = "top_{:03d}".format(int(round(fraction * 100)))
        result["matched_budget"][key] = {
            "budget_n": keep,
            "original_confidence": _selected_summary(
                baseline, low, gt_action, records["correct"]
            ),
            "best_single_scale": _selected_summary(
                any_scale, low, gt_action, records["any_recovery_correct"]
            ),
            "two_scale_consensus": _selected_summary(
                consensus,
                low,
                gt_action,
                records["consensus_recovery_correct"],
            ),
            "largest_representation_mse": _selected_summary(
                representation, low, gt_action, records["correct"]
            ),
            "consensus_selected_consistent_rate": (
                float(
                    records["scale_consistent"][consensus]
                    .float()
                    .mean()
                    .item()
                )
                if keep
                else None
            ),
        }
    return result


def _subset_records(records, mask):
    subset = {
        key: value[mask]
        for key, value in records.items()
        if key != "alpha"
    }
    subset["alpha"] = records["alpha"]
    return subset


def _high_confidence_summary(records):
    summary = build_summary(records)
    high = records["score"] >= records["alpha"]
    flipped = high & ~records["consistent"]
    high_error = 1.0 - float(
        records["correct"][high].float().mean().item()
    ) if high.any() else None
    flip_error = 1.0 - float(
        records["correct"][flipped].float().mean().item()
    ) if flipped.any() else None
    summary["class_flip_error_enrichment"] = (
        flip_error / high_error
        if high_error is not None and high_error > 0 and flip_error is not None
        else None
    )
    summary["exact_rank"] = build_exact_rank_retention(records)
    summary["score_stratified_retention"] = (
        build_score_stratified_retention(records)
    )
    return summary


def _records_with_stability(records, metric):
    selected = dict(records)
    selected["stability"] = records[metric]
    return selected


def _forward_target(model, videos):
    inputs, masks = model.preprocessing(videos, mode="validation")
    strict = model._stage1_strict_masks(masks)
    feats, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn, fpn_masks = model.neck_target(feats, backbone_masks)
    logits, head_features = model.cls_head_target(fpn, fpn_masks)
    return fpn, fpn_masks, strict, logits, head_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--scales", type=float, nargs=2, default=(0.75, 1.25))
    parser.add_argument("--levels", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    scales = tuple(args.scales)
    if not scales[0] < 1.0 < scales[1]:
        raise ValueError("--scales must contain one value below and above 1")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "temporal_scale"}
    cfg["dataset"]["stage1_context"] = True
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.load(
        open(cfg["dataset"]["json_file"], encoding="utf-8")
    )["database"]
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    if any(level < 0 or level >= model.n_levels for level in args.levels):
        raise ValueError("--levels contains an unavailable FPN level")
    enabled_levels = set(args.levels)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    record_keys = (
        "score",
        "consistent",
        "stability",
        "gt_action",
        "correct",
        "level",
        "scale_consistent",
        "consensus_recovery_score",
        "consensus_recovery_ranking",
        "consensus_recovery_correct",
        "any_recovery_score",
        "any_recovery_correct",
        "representation_mse",
    )
    record_lists = {key: [] for key in record_keys}
    video_total = 0
    video_valid = 0
    strict_original_n = 0
    common_valid_n = 0

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            scaled_video_batches = [[], []]
            scaled_metadata = [[], []]
            valid_rows = []

            for raw_video, video in zip(raw_videos, videos):
                original = raw_video["feats"]
                full = raw_video["full_feats"]
                original_start = int(video["metadata"].get("crop_start", 0))
                length = original.shape[-1]
                if not torch.equal(
                    original, full[:, original_start:original_start + length]
                ):
                    raise RuntimeError("crop does not match full features")

                views = [
                    temporal_scale_view_on_cpu(
                        full, original_start, length, scale
                    )
                    for scale in scales
                ]
                row_valid = all(view is not None for view in views)
                valid_rows.append(row_valid)
                for view_index, view in enumerate(views):
                    if view is None:
                        view = {
                            "feats": original.detach().cpu(),
                            "start": original_start,
                            "span": length,
                            "length": length,
                            "scale": scales[view_index],
                        }
                    scaled_video = dict(video)
                    scaled_video["feats"] = view["feats"]
                    scaled_video_batches[view_index].append(scaled_video)
                    scaled_metadata[view_index].append(view)
                video_total += 1
                video_valid += int(row_valid)

            fpn0, _, strict0, logits0, head_features0 = _forward_target(
                model, videos
            )
            scaled_outputs = [
                _forward_target(model, scaled_videos)
                for scaled_videos in scaled_video_batches
            ]
            points = model.point_generator(fpn0)

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    args.device,
                )
                if not torch.equal(
                    segments, raw_video["segments"].to(args.device)
                ) or not torch.equal(
                    labels, raw_video["labels"].to(args.device)
                ):
                    raise RuntimeError(
                        "target GT mismatch for {}".format(video_id)
                    )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)

            label_start = 0
            valid_video_rows = torch.tensor(
                valid_rows, dtype=torch.bool, device=args.device
            ).unsqueeze(1)
            for level, point_grid in enumerate(points):
                level_length = point_grid.shape[0]
                labels = gt_cls[:, label_start:label_start + level_length]
                label_start += level_length
                if level not in enabled_levels:
                    continue

                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                aligned_probabilities = []
                aligned_features = []
                aligned_validities = []
                for view_index, (
                    _,
                    _,
                    strict,
                    logits,
                    head_features,
                ) in enumerate(scaled_outputs):
                    view_prob = logits[level].permute(0, 2, 1).sigmoid()
                    view_features = head_features[level].permute(0, 2, 1)
                    aligned_rows, feature_rows, validity_rows = [], [], []
                    for batch_index, (video, metadata) in enumerate(
                        zip(videos, scaled_metadata[view_index])
                    ):
                        original_start = int(
                            video["metadata"].get("crop_start", 0)
                        )
                        query = map_original_points_to_scale_view(
                            point_grid[:, 0],
                            original_start,
                            metadata["start"],
                            metadata["span"],
                            metadata["length"],
                        )
                        aligned, aligned_valid = interpolate_probabilities(
                            view_prob[batch_index:batch_index + 1],
                            strict[level][batch_index:batch_index + 1]
                            .squeeze(1),
                            point_grid,
                            query,
                        )
                        aligned_rows.append(aligned)
                        aligned_feature, _ = interpolate_probabilities(
                            view_features[batch_index:batch_index + 1],
                            strict[level][batch_index:batch_index + 1]
                            .squeeze(1),
                            point_grid,
                            query,
                        )
                        feature_rows.append(aligned_feature)
                        validity_rows.append(aligned_valid)
                    aligned_probabilities.append(torch.cat(aligned_rows))
                    aligned_features.append(torch.cat(feature_rows))
                    aligned_validities.append(torch.cat(validity_rows))

                strict_original = strict0[level].squeeze(1)
                common_valid = strict_original & valid_video_rows
                for validity in aligned_validities:
                    common_valid &= validity
                strict_original_n += int(
                    (strict_original & valid_video_rows).sum().item()
                )
                common_valid_n += int(common_valid.sum().item())

                score0, class0 = prob0.max(dim=-1)
                view_scores, view_classes, base_class_scores = [], [], []
                for aligned in aligned_probabilities:
                    view_score, view_class = aligned.max(dim=-1)
                    view_scores.append(view_score)
                    view_classes.append(view_class)
                    base_class_scores.append(
                        aligned.gather(-1, class0.unsqueeze(-1)).squeeze(-1)
                    )
                consistent = (
                    (class0 == view_classes[0])
                    & (class0 == view_classes[1])
                )
                stability = population_std(
                    torch.stack(
                        (score0, base_class_scores[0], base_class_scores[1])
                    ),
                    dim=0,
                )
                original_features = head_features0[level].permute(0, 2, 1)
                representation_mse = torch.stack(
                    [
                        (original_features - aligned_feature)
                        .square()
                        .mean(dim=-1)
                        for aligned_feature in aligned_features
                    ]
                ).mean(dim=0)
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1) > 0

                scale_consistent = view_classes[0] == view_classes[1]
                consensus_score = torch.minimum(
                    view_scores[0], view_scores[1]
                )
                consensus_correct = labels.gather(
                    -1, view_classes[0].unsqueeze(-1)
                ).squeeze(-1) > 0
                consensus_ranking = torch.where(
                    scale_consistent,
                    consensus_score,
                    torch.full_like(consensus_score, -1.0),
                )
                first_is_best = view_scores[0] >= view_scores[1]
                any_score = torch.maximum(view_scores[0], view_scores[1])
                any_class = torch.where(
                    first_is_best, view_classes[0], view_classes[1]
                )
                any_correct = labels.gather(
                    -1, any_class.unsqueeze(-1)
                ).squeeze(-1) > 0

                values = {
                    "score": score0,
                    "consistent": consistent,
                    "stability": stability,
                    "gt_action": gt_action,
                    "correct": correct,
                    "level": torch.full_like(class0, level),
                    "scale_consistent": scale_consistent,
                    "consensus_recovery_score": consensus_score,
                    "consensus_recovery_ranking": consensus_ranking,
                    "consensus_recovery_correct": consensus_correct,
                    "any_recovery_score": any_score,
                    "any_recovery_correct": any_correct,
                    "representation_mse": representation_mse,
                }
                for key, value in values.items():
                    record_lists[key].append(
                        value[common_valid].detach().cpu()
                    )

    records = {key: torch.cat(value) for key, value in record_lists.items()}
    records["alpha"] = alpha
    per_level = {}
    for level in sorted(enabled_levels):
        level_records = _subset_records(records, records["level"] == level)
        per_level[str(level)] = {
            "high_confidence_error_exposure": {
                "output_probability": _high_confidence_summary(level_records),
                "representation_mse": _high_confidence_summary(
                    _records_with_stability(
                        level_records, "representation_mse"
                    )
                ),
            },
            "low_confidence_action_recovery": build_low_confidence_summary(
                level_records
            ),
        }

    result = {
        "audit": "temporal_scale_stage1_error_exposure_and_action_recovery",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": args.config,
        "split": "target_train",
        "seed": args.seed,
        "scales": scales,
        "levels": sorted(enabled_levels),
        "alpha": alpha,
        "model_mode": "eval_frozen_student",
        "target_gt_role": "post_hoc_audit_only",
        "comparison_scope": "same common strict-valid target FPN anchors",
        "videos": video_total,
        "scale_valid_videos": video_valid,
        "scale_valid_video_ratio": video_valid / max(video_total, 1),
        "strict_original_n_on_scale_valid_videos": strict_original_n,
        "common_valid_n": common_valid_n,
        "common_valid_ratio": common_valid_n / max(strict_original_n, 1),
        "high_confidence_error_exposure": {
            "output_probability": _high_confidence_summary(records),
            "representation_mse": _high_confidence_summary(
                _records_with_stability(records, "representation_mse")
            ),
        },
        "low_confidence_action_recovery": build_low_confidence_summary(records),
        "per_level": per_level,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    brief = {
        "scale_valid_video_ratio": result["scale_valid_video_ratio"],
        "common_valid_ratio": result["common_valid_ratio"],
        "high_confidence": result["high_confidence_error_exposure"][
            "output_probability"
        ]["high_confidence"],
        "representation_mse_retention": result[
            "high_confidence_error_exposure"
        ]["representation_mse"][
            "exact_rank"
        ],
        "class_flip_rejected": result["high_confidence_error_exposure"][
            "output_probability"
        ]["class_flip_rejected"],
        "low_confidence_operational": result[
            "low_confidence_action_recovery"
        ]["operational_threshold"],
    }
    print(json.dumps(brief, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
