"""Audit feature noise for Stage1 pseudo-label filtering and recovery."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_utils import (
    deterministic_channel_dropout_perturbation,
    deterministic_gaussian_perturbation,
    population_std,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_shift_action_stability import summarize_group, summarize_route


HIGH_RETENTION = (0.8, 0.6, 0.4, 0.2)
LOW_BUDGETS = (0.01, 0.02, 0.05, 0.10)


def _top_k(universe, ranking, keep, descending=True):
    indices = torch.where(universe)[0]
    selected = torch.zeros_like(universe)
    if not indices.numel():
        return selected
    keep = min(max(1, keep), int(indices.numel()))
    order = torch.argsort(
        ranking[indices], descending=descending, stable=True
    )
    selected[indices[order[:keep]]] = True
    return selected


def _selection_summary(selected, universe, gt_action, correct):
    count = int(selected.sum().item())
    universe_n = int(universe.sum().item())
    gt_total = int(gt_action[universe].sum().item())
    return {
        "n": count,
        "selection_rate": count / max(universe_n, 1),
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


def _exact_high_retention(records, metric):
    score = records["score"]
    candidate = (score >= records["alpha"]) & records["consistent"]
    candidate_n = int(candidate.sum().item())
    result = {}
    for fraction in HIGH_RETENTION:
        keep = max(1, int(math.ceil(fraction * candidate_n))) if candidate_n else 0
        stable = _top_k(candidate, records[metric], keep, descending=False)
        confidence = _top_k(candidate, score, keep, descending=True)
        key = "retain_{:03d}".format(int(round(100 * fraction)))
        result[key] = {
            "budget_n": keep,
            "perturbation_metric": summarize_route(
                stable, records["gt_action"], records["correct"]
            ),
            "confidence": summarize_route(
                confidence, records["gt_action"], records["correct"]
            ),
        }
    return result


def _score_stratified_high_retention(records, metric, bin_width=0.05):
    score = records["score"]
    candidate = (score >= records["alpha"]) & records["consistent"]
    bins = []
    lower = float(records["alpha"])
    while lower < 1.0:
        upper = min(lower + bin_width, 1.0)
        upper_mask = score <= upper if upper == 1.0 else score < upper
        indices = torch.where(
            candidate & (score >= lower) & upper_mask
        )[0]
        bins.append((lower, upper, indices))
        lower = upper

    result = {}
    for fraction in HIGH_RETENTION:
        selected = torch.zeros_like(candidate)
        expected_correct = 0.0
        expected_n = 0
        bin_stats = {}
        for lower, upper, indices in bins:
            if not indices.numel():
                continue
            keep = max(1, int(math.ceil(fraction * indices.numel())))
            order = torch.argsort(
                records[metric][indices], descending=False, stable=True
            )
            chosen = indices[order[:keep]]
            selected[chosen] = True
            candidate_rate = float(
                records["correct"][indices].float().mean().item()
            )
            expected_correct += keep * candidate_rate
            expected_n += keep
            bin_stats["{:.2f}_{:.2f}".format(lower, upper)] = {
                "n": int(indices.numel()),
                "kept_n": keep,
                "candidate_correct_class_rate": candidate_rate,
                "selected_correct_class_rate": float(
                    records["correct"][chosen].float().mean().item()
                ),
            }
        summary = summarize_route(
            selected, records["gt_action"], records["correct"]
        )
        expected_rate = expected_correct / max(expected_n, 1)
        key = "retain_{:03d}".format(int(round(100 * fraction)))
        result[key] = {
            "selected": summary,
            "confidence_bin_expected_correct_class_rate": expected_rate,
            "independent_gain": (
                summary["correct_class_rate"] - expected_rate
                if summary["correct_class_rate"] is not None
                else None
            ),
            "score_bins": bin_stats,
        }
    return result


def build_high_summary(records):
    score = records["score"]
    high = score >= records["alpha"]
    consistent = high & records["consistent"]
    flipped = high & ~records["consistent"]
    valid_gt_actions = int(records["gt_action"].sum().item())
    return {
        "valid": {
            "n": int(score.numel()),
            "gt_action_rate": float(records["gt_action"].float().mean().item()),
        },
        "high_confidence": summarize_route(
            high, records["gt_action"], records["correct"]
        ),
        "all_view_class_consistent": summarize_route(
            consistent, records["gt_action"], records["correct"]
        ),
        "class_flip_rejected": summarize_group(
            flipped,
            records["gt_action"],
            records["correct"],
            valid_gt_actions,
        ),
        "probability_std": {
            "exact_rank": _exact_high_retention(records, "probability_std"),
            "score_stratified": _score_stratified_high_retention(
                records, "probability_std"
            ),
        },
        "representation_mse": {
            "exact_rank": _exact_high_retention(records, "representation_mse"),
            "score_stratified": _score_stratified_high_retention(
                records, "representation_mse"
            ),
        },
    }


def build_low_summary(records):
    score = records["score"]
    low = score < records["alpha"]
    gt_action = records["gt_action"]
    any_cross = low & (records["any_recovery_score"] >= records["alpha"])
    consensus_cross = (
        low
        & records["view_consistent"]
        & (records["consensus_recovery_score"] >= records["alpha"])
    )
    result = {
        "low_confidence": {
            "n": int(low.sum().item()),
            "gt_action_n": int(gt_action[low].sum().item()),
            "gt_action_rate": (
                float(gt_action[low].float().mean().item())
                if low.any()
                else None
            ),
        },
        "operational_threshold": {
            "any_view_crosses_alpha": _selection_summary(
                any_cross, low, gt_action, records["any_recovery_correct"]
            ),
            "all_view_consensus_crosses_alpha": _selection_summary(
                consensus_cross,
                low,
                gt_action,
                records["consensus_recovery_correct"],
            ),
        },
        "matched_budget": {},
    }
    low_n = int(low.sum().item())
    for fraction in LOW_BUDGETS:
        keep = max(1, int(math.ceil(fraction * low_n))) if low_n else 0
        baseline = _top_k(low, score, keep)
        any_view = _top_k(low, records["any_recovery_score"], keep)
        consensus = _top_k(
            low, records["consensus_recovery_ranking"], keep
        )
        representation = _top_k(
            low, records["representation_mse"], keep
        )
        key = "top_{:03d}".format(int(round(100 * fraction)))
        result["matched_budget"][key] = {
            "budget_n": keep,
            "original_confidence": _selection_summary(
                baseline, low, gt_action, records["correct"]
            ),
            "best_perturbed_view": _selection_summary(
                any_view, low, gt_action, records["any_recovery_correct"]
            ),
            "all_view_consensus": _selection_summary(
                consensus,
                low,
                gt_action,
                records["consensus_recovery_correct"],
            ),
            "largest_representation_mse": _selection_summary(
                representation, low, gt_action, records["correct"]
            ),
        }
    return result


def _forward_target(model, inputs, masks):
    strict = model._stage1_strict_masks(masks)
    features, feature_masks, _ = model.backbone_target(inputs, masks)
    fpn, fpn_masks = model.neck_target(features, feature_masks)
    logits, head_features = model.cls_head_target(fpn, fpn_masks)
    return fpn, strict, logits, head_features


def _subset(records, mask):
    selected = {
        key: value[mask]
        for key, value in records.items()
        if key != "alpha"
    }
    selected["alpha"] = records["alpha"]
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--perturbation", choices=("gaussian", "channel_dropout"), required=True
    )
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument(
        "--perturbation-seeds",
        type=int,
        nargs="+",
        default=(20260730, 20260731, 20260732, 20260733),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--levels", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.strength <= 0 or (
        args.perturbation == "channel_dropout" and args.strength >= 1
    ):
        raise ValueError("invalid perturbation strength")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": args.perturbation}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    if any(level < 0 or level >= model.n_levels for level in args.levels):
        raise ValueError("--levels contains an unavailable FPN level")
    enabled_levels = set(args.levels)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    record_keys = (
        "score",
        "consistent",
        "probability_std",
        "representation_mse",
        "gt_action",
        "correct",
        "level",
        "view_consistent",
        "consensus_recovery_score",
        "consensus_recovery_ranking",
        "consensus_recovery_correct",
        "any_recovery_score",
        "any_recovery_correct",
    )
    record_lists = {key: [] for key in record_keys}
    strict_valid_n = 0

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
            fpn0, strict0, logits0, head0 = _forward_target(
                model, inputs, masks
            )
            sample_keys = [
                "{}:{}".format(
                    video["metadata"]["video_id"],
                    video["metadata"].get("crop_start", 0),
                )
                for video in videos
            ]
            perturbed_outputs = []
            for perturbation_seed in args.perturbation_seeds:
                if args.perturbation == "gaussian":
                    perturbed = deterministic_gaussian_perturbation(
                        inputs,
                        masks,
                        sample_keys,
                        args.strength,
                        perturbation_seed,
                    )
                else:
                    perturbed = deterministic_channel_dropout_perturbation(
                        inputs,
                        masks,
                        sample_keys,
                        args.strength,
                        perturbation_seed,
                    )
                perturbed_outputs.append(
                    _forward_target(model, perturbed, masks)
                )
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
                    raise RuntimeError("target GT reconstruction mismatch")
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)

            label_start = 0
            for level, point_grid in enumerate(points):
                level_length = point_grid.shape[0]
                labels = gt_cls[:, label_start:label_start + level_length]
                label_start += level_length
                if level not in enabled_levels:
                    continue
                valid = strict0[level].squeeze(1)
                for _, strict, _, _ in perturbed_outputs:
                    if not torch.equal(valid, strict[level].squeeze(1)):
                        raise RuntimeError("feature noise changed strict masks")
                strict_valid_n += int(valid.sum().item())

                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                feature0 = head0[level].permute(0, 2, 1)
                view_probabilities = [
                    logits[level].permute(0, 2, 1).sigmoid()
                    for _, _, logits, _ in perturbed_outputs
                ]
                view_features = [
                    features[level].permute(0, 2, 1)
                    for _, _, _, features in perturbed_outputs
                ]
                score0, class0 = prob0.max(dim=-1)
                view_scores, view_classes, class0_scores = [], [], []
                for probability in view_probabilities:
                    view_score, view_class = probability.max(dim=-1)
                    view_scores.append(view_score)
                    view_classes.append(view_class)
                    class0_scores.append(
                        probability.gather(
                            -1, class0.unsqueeze(-1)
                        ).squeeze(-1)
                    )
                score_stack = torch.stack(view_scores)
                class_stack = torch.stack(view_classes)
                consistent = (class_stack == class0.unsqueeze(0)).all(dim=0)
                probability_std = population_std(
                    torch.cat(
                        (score0.unsqueeze(0), torch.stack(class0_scores)),
                        dim=0,
                    ),
                    dim=0,
                )
                representation_mse = torch.stack([
                    (feature0 - feature).square().mean(dim=-1)
                    for feature in view_features
                ]).mean(dim=0)
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1) > 0

                view_consistent = (
                    class_stack == class_stack[0].unsqueeze(0)
                ).all(dim=0)
                consensus_score = score_stack.min(dim=0).values
                consensus_class = class_stack[0]
                consensus_correct = labels.gather(
                    -1, consensus_class.unsqueeze(-1)
                ).squeeze(-1) > 0
                consensus_ranking = torch.where(
                    view_consistent,
                    consensus_score,
                    torch.full_like(consensus_score, -1.0),
                )
                any_score, any_view = score_stack.max(dim=0)
                any_class = class_stack.gather(
                    0, any_view.unsqueeze(0)
                ).squeeze(0)
                any_correct = labels.gather(
                    -1, any_class.unsqueeze(-1)
                ).squeeze(-1) > 0

                values = {
                    "score": score0,
                    "consistent": consistent,
                    "probability_std": probability_std,
                    "representation_mse": representation_mse,
                    "gt_action": gt_action,
                    "correct": correct,
                    "level": torch.full_like(class0, level),
                    "view_consistent": view_consistent,
                    "consensus_recovery_score": consensus_score,
                    "consensus_recovery_ranking": consensus_ranking,
                    "consensus_recovery_correct": consensus_correct,
                    "any_recovery_score": any_score,
                    "any_recovery_correct": any_correct,
                }
                for key, value in values.items():
                    record_lists[key].append(value[valid].detach().cpu())

    records = {key: torch.cat(value) for key, value in record_lists.items()}
    records["alpha"] = alpha
    per_level = {}
    for level in sorted(enabled_levels):
        level_records = _subset(records, records["level"] == level)
        per_level[str(level)] = {
            "high_confidence_error_filtering": build_high_summary(
                level_records
            ),
            "low_confidence_action_recovery": build_low_summary(level_records),
        }
    result = {
        "audit": "stage1_feature_noise_filtering_and_recovery",
        "perturbation": args.perturbation,
        "strength": args.strength,
        "perturbation_seeds": args.perturbation_seeds,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": args.config,
        "split": "target_train",
        "crop_seed": args.seed,
        "levels": sorted(enabled_levels),
        "alpha": alpha,
        "model_mode": "eval_frozen_student",
        "target_gt_role": "post_hoc_audit_only",
        "videos": len(dataset),
        "strict_valid_n": strict_valid_n,
        "high_confidence_error_filtering": build_high_summary(records),
        "low_confidence_action_recovery": build_low_summary(records),
        "per_level": per_level,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    brief = {
        "perturbation": args.perturbation,
        "strength": args.strength,
        "strict_valid_n": strict_valid_n,
        "high_confidence": result["high_confidence_error_filtering"][
            "high_confidence"
        ],
        "class_flip_rejected": result["high_confidence_error_filtering"][
            "class_flip_rejected"
        ],
        "low_confidence_operational": result[
            "low_confidence_action_recovery"
        ]["operational_threshold"],
    }
    print(json.dumps(brief, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
