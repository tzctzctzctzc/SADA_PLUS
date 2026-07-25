"""Audit whether adjacent-crop Context stability purifies action labels.

Target GT is used only after a frozen model has produced detached routing
records. Baseline and Context gates are compared on the same original crops
and common strict-valid FPN anchors.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_utils import (
    context_view_on_cpu,
    interpolate_probabilities,
    population_std,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_shift_action_stability import (
    build_summary,
    summarize_group,
    summarize_route,
)


RETENTION_FRACTIONS = (0.8, 0.6, 0.4, 0.2)


def describe(values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "min": float(tensor.min().item()),
        "median": float(tensor.median().item()),
        "mean": float(tensor.mean().item()),
        "max": float(tensor.max().item()),
    }


def records_for_metric(records, metric):
    selected = dict(records)
    selected["stability"] = records[metric]
    return selected


def build_exact_rank_retention(records):
    """Select an exact Top-K so tied distances cannot inflate the budget."""
    score = records["score"]
    stability = records["stability"]
    gt_action = records["gt_action"]
    correct = records["correct"]
    candidate = (score >= records["alpha"]) & records["consistent"]
    candidate_indices = torch.where(candidate)[0]
    valid_gt_actions = int(gt_action.sum().item())
    result = {"stability_retention": {}, "confidence_retention": {}}
    if not candidate_indices.numel():
        return result

    stability_order = torch.argsort(stability[candidate_indices])
    confidence_order = torch.argsort(
        score[candidate_indices], descending=True
    )
    for fraction in RETENTION_FRACTIONS:
        keep = max(1, int(math.ceil(fraction * candidate_indices.numel())))
        stable_indices = candidate_indices[stability_order[:keep]]
        confidence_indices = candidate_indices[confidence_order[:keep]]
        stable = torch.zeros_like(candidate)
        confidence = torch.zeros_like(candidate)
        stable[stable_indices] = True
        confidence[confidence_indices] = True
        rejected = candidate & ~stable
        key = "retain_{:03d}".format(int(round(fraction * 100)))
        result["stability_retention"][key] = {
            "diagnostic_tau": float(stability[stable_indices[-1]].item()),
            "reliable_action": summarize_route(stable, gt_action, correct),
            "stability_rejected": summarize_group(
                rejected, gt_action, correct, valid_gt_actions
            ),
        }
        result["confidence_retention"][key] = {
            "diagnostic_score_floor": float(
                score[confidence_indices[-1]].item()
            ),
            "reliable_action": summarize_route(
                confidence, gt_action, correct
            ),
        }
    return result


def build_score_stratified_retention(records, bin_width=0.05):
    """Rank by stability while preserving the confidence-bin mixture."""
    score = records["score"]
    stability = records["stability"]
    gt_action = records["gt_action"]
    correct = records["correct"]
    candidate = (score >= records["alpha"]) & records["consistent"]

    bins = []
    lower = float(records["alpha"])
    while lower < 1.0:
        upper = min(lower + bin_width, 1.0)
        if upper == 1.0:
            mask = candidate & (score >= lower) & (score <= upper)
        else:
            mask = candidate & (score >= lower) & (score < upper)
        bins.append((lower, upper, torch.where(mask)[0]))
        lower = upper

    result = {}
    for fraction in RETENTION_FRACTIONS:
        reliable = torch.zeros_like(candidate)
        bin_stats = {}
        for lower, upper, indices in bins:
            count = int(indices.numel())
            if not count:
                continue
            keep = max(1, int(math.ceil(fraction * count)))
            order = torch.argsort(stability[indices])
            chosen = indices[order[:keep]]
            reliable[chosen] = True
            key = "{:.2f}_{:.2f}".format(lower, upper)
            bin_stats[key] = {
                "n": count,
                "kept_n": keep,
                "candidate_correct_class_rate": float(
                    correct[indices].float().mean().item()
                ),
                "stable_correct_class_rate": float(
                    correct[chosen].float().mean().item()
                ),
            }
        key = "retain_{:03d}".format(int(round(fraction * 100)))
        summary = summarize_route(reliable, gt_action, correct)
        summary["actual_candidate_retention"] = (
            summary["n"] / max(int(candidate.sum().item()), 1)
        )
        result[key] = {"reliable_action": summary, "score_bins": bin_stats}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {
        "enabled": False,
        "perturbation": "context",
        "seed": args.seed,
    }
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
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    record_lists = {
        key: []
        for key in (
            "score",
            "consistent",
            "stability",
            "representation_cosine",
            "representation_mse",
            "gt_action",
            "correct",
            "level",
            "direction",
        )
    }
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    rho = float(cfg["stage1"].get("context_rho", 64.0 / 2304.0))
    rho_max = float(cfg["stage1"].get("context_rho_max", 2.0 * rho))
    direction_rng = torch.Generator(device="cpu")
    direction_rng.manual_seed(args.seed)

    video_total = 0
    video_valid = 0
    direction_counts = {-1: 0, 0: 0, 1: 0}
    offsets = []
    ordinary_valid_n = 0
    eligible_strict_n = 0
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
            context_videos = []
            directions = []
            context_valid = []
            batch_offsets = []

            for raw_video, video in zip(raw_videos, videos):
                original = raw_video["feats"]
                full = raw_video["full_feats"]
                start = int(video["metadata"].get("crop_start", 0))
                length = original.shape[-1]
                if not torch.equal(original, full[:, start:start + length]):
                    raise RuntimeError("Context crop does not match full features")

                offset = max(1, int(round(rho * length)))
                legal = []
                if offset / float(length) <= rho_max:
                    if start - offset >= 0:
                        legal.append(-1)
                    if start + length + offset <= full.shape[-1]:
                        legal.append(1)
                if legal:
                    pick = torch.randint(
                        0, len(legal), (1,), generator=direction_rng
                    ).item()
                    direction = legal[pick]
                    context_start = start + direction * offset
                    is_valid = True
                else:
                    direction = 0
                    context_start = start
                    is_valid = False

                context_video = dict(video)
                context_video["feats"] = context_view_on_cpu(
                    original, full, context_start, length
                )
                context_videos.append(context_video)
                directions.append(direction)
                context_valid.append(is_valid)
                direction_counts[direction] += 1
                batch_offsets.append(offset)
                video_total += 1
                video_valid += int(is_valid)
            offsets.extend(batch_offsets)

            inputs0, masks0 = model.preprocessing(videos, mode="validation")
            inputs1, masks1 = model.preprocessing(
                context_videos, mode="validation"
            )
            strict0 = model._stage1_strict_masks(masks0)
            strict1 = model._stage1_strict_masks(masks1)
            feats0, backbone_masks0, _ = model.backbone_target(inputs0, masks0)
            fpn0, fpn_masks0 = model.neck_target(feats0, backbone_masks0)
            logits0, head_feats0 = model.cls_head_target(fpn0, fpn_masks0)
            feats1, backbone_masks1, _ = model.backbone_target(inputs1, masks1)
            fpn1, fpn_masks1 = model.neck_target(feats1, backbone_masks1)
            logits1, head_feats1 = model.cls_head_target(fpn1, fpn_masks1)
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
                        "Target GT reconstruction mismatch for {}".format(video_id)
                    )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)

            start = 0
            valid_video_rows = torch.tensor(
                context_valid, dtype=torch.bool, device=args.device
            ).unsqueeze(1)
            direction_rows = torch.tensor(
                directions, dtype=torch.long, device=args.device
            ).unsqueeze(1)
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                prob1 = logits1[level].permute(0, 2, 1).sigmoid()
                aligned_rows, aligned_valid_rows = [], []
                for batch_idx, (direction, offset) in enumerate(
                    zip(directions, batch_offsets)
                ):
                    query = point_grid[:, 0] - direction * offset
                    aligned, aligned_valid = interpolate_probabilities(
                        prob1[batch_idx:batch_idx + 1],
                        strict1[level][batch_idx:batch_idx + 1].squeeze(1),
                        point_grid,
                        query,
                    )
                    aligned_rows.append(aligned)
                    aligned_valid_rows.append(aligned_valid)
                aligned_prob1 = torch.cat(aligned_rows, dim=0)
                aligned_valid = torch.cat(aligned_valid_rows, dim=0)
                feature0 = head_feats0[level].permute(0, 2, 1)
                feature1 = head_feats1[level].permute(0, 2, 1)
                aligned_feature_rows = []
                for batch_idx, (direction, offset) in enumerate(
                    zip(directions, batch_offsets)
                ):
                    query = point_grid[:, 0] - direction * offset
                    aligned_feature, _ = interpolate_probabilities(
                        feature1[batch_idx:batch_idx + 1],
                        strict1[level][batch_idx:batch_idx + 1].squeeze(1),
                        point_grid,
                        query,
                    )
                    aligned_feature_rows.append(aligned_feature)
                aligned_feature1 = torch.cat(aligned_feature_rows, dim=0)
                strict_valid0 = strict0[level].squeeze(1)
                eligible = strict_valid0 & valid_video_rows
                common_valid = eligible & aligned_valid

                score0, class0 = prob0.max(dim=-1)
                _, class1 = aligned_prob1.max(dim=-1)
                class1_score = aligned_prob1.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1)
                stability = population_std(
                    torch.stack((score0, class1_score), dim=0), dim=0
                )
                representation_cosine = 1.0 - F.cosine_similarity(
                    feature0, aligned_feature1, dim=-1, eps=1e-8
                )
                representation_mse = F.mse_loss(
                    feature0, aligned_feature1, reduction="none"
                ).mean(dim=-1)
                labels = gt_cls[:, start:start + length]
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1) > 0

                ordinary_valid_n += int(fpn_masks0[level].sum().item())
                eligible_strict_n += int(eligible.sum().item())
                common_valid_n += int(common_valid.sum().item())
                values = {
                    "score": score0,
                    "consistent": class0 == class1,
                    "stability": stability,
                    "representation_cosine": representation_cosine,
                    "representation_mse": representation_mse,
                    "gt_action": gt_action,
                    "correct": correct,
                    "level": torch.full_like(class0, level),
                    "direction": direction_rows.expand_as(class0),
                }
                for key, value in values.items():
                    record_lists[key].append(value[common_valid].detach().cpu())
                start += length

    records = {key: torch.cat(value) for key, value in record_lists.items()}
    records["alpha"] = alpha
    cosine_records = records_for_metric(records, "representation_cosine")
    mse_records = records_for_metric(records, "representation_mse")
    per_level = {}
    representation_per_level = {}
    for level in range(model.n_levels):
        selected = records["level"] == level
        level_records = {
            key: value[selected]
            for key, value in records.items()
            if key != "alpha"
        }
        level_records["alpha"] = alpha
        per_level[str(level)] = build_summary(level_records)
        representation_per_level[str(level)] = build_summary(
            records_for_metric(level_records, "representation_cosine")
        )

    per_direction = {}
    representation_per_direction = {}
    for direction in (-1, 1):
        selected = records["direction"] == direction
        direction_records = {
            key: value[selected]
            for key, value in records.items()
            if key != "alpha"
        }
        direction_records["alpha"] = alpha
        per_direction[str(direction)] = build_summary(direction_records)
        representation_per_direction[str(direction)] = build_summary(
            records_for_metric(direction_records, "representation_cosine")
        )

    result = {
        "audit": "context_action_stability_direction_test",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": args.config,
        "split": "target_train",
        "seed": args.seed,
        "videos": video_total,
        "context_video_valid": video_valid,
        "context_video_valid_ratio": video_valid / max(video_total, 1),
        "direction_counts": {str(key): value for key, value in direction_counts.items()},
        "offset": describe(offsets),
        "context_rho": rho,
        "context_rho_max": rho_max,
        "batch_size": batch_size,
        "alpha": alpha,
        "model_mode": "eval",
        "comparison_scope": "same common strict-valid target FPN anchors",
        "ordinary_valid_n": ordinary_valid_n,
        "eligible_strict_n": eligible_strict_n,
        "common_valid_n": common_valid_n,
        "common_overlap_ratio": common_valid_n / max(eligible_strict_n, 1),
        "global": build_summary(records),
        "per_direction": per_direction,
        "per_level": per_level,
        "representation_stability": {
            "primary_metric": "one_minus_cosine_of_penultimate_anchor_features",
            "cosine": {
                "global": build_summary(cosine_records),
                "exact_rank": build_exact_rank_retention(cosine_records),
                "score_stratified_retention": build_score_stratified_retention(
                    cosine_records
                ),
                "per_direction": representation_per_direction,
                "per_level": representation_per_level,
            },
            "raw_mse": {
                "global": build_summary(mse_records),
                "exact_rank": build_exact_rank_retention(mse_records),
                "score_stratified_retention": build_score_stratified_retention(
                    mse_records
                ),
            },
        },
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["global"], indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
