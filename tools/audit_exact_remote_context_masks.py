"""Audit anchor-centered remote-context masks for Stage1 recovery.

Every intervention keeps the temporal grid and the anchor's complete local
receptive field. Target GT is joined only after all frozen predictions have
been generated and is never used for sampling or perturbation construction.
"""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch


MASK_MODES = ("past", "future", "bilateral")
MATCHED_BUDGET_FRACTIONS = (0.01, 0.02, 0.05, 0.10)


def deterministic_keep(seed, video_id, crop_start, level, anchor_index, rate):
    payload = "{}|{}|{}|{}|{}".format(
        seed, video_id, crop_start, level, anchor_index
    ).encode("utf-8")
    value = int(hashlib.sha1(payload).hexdigest()[:16], 16) / float(2**64)
    return value < rate


def build_remote_context_batch(inputs, masks, centers, radii, modes):
    """Create fixed-grid masked copies using the model's native valid mask."""
    if not (len(centers) == len(radii) == len(modes)):
        raise ValueError("centers, radii, and modes must have equal length")
    if inputs.shape[0] != 1 or masks.shape[0] != 1:
        raise ValueError("remote-context construction expects one source crop")
    count = len(centers)
    masked_inputs = inputs.expand(count, -1, -1).clone()
    masked_masks = masks.expand(count, -1, -1).clone()
    length = inputs.shape[-1]
    for row, (center, radius, mode) in enumerate(zip(centers, radii, modes)):
        if mode not in MASK_MODES:
            raise ValueError("unknown mask mode: {}".format(mode))
        left = max(int(center) - int(radius), 0)
        right = min(int(center) + int(radius) + 1, length)
        if mode in ("past", "bilateral"):
            masked_masks[row, :, :left] = False
        if mode in ("future", "bilateral"):
            masked_masks[row, :, right:] = False
    masked_inputs = masked_inputs * masked_masks.to(masked_inputs.dtype)
    return masked_inputs, masked_masks


def ratio(numerator, denominator):
    return float(numerator) / denominator if denominator else None


def distribution(values):
    if values.numel() == 0:
        return None
    values = values.float()
    qs = torch.tensor([0.1, 0.5, 0.9], device=values.device)
    quantile = torch.quantile(values, qs).tolist()
    return {
        "mean": float(values.mean().item()),
        "q10": quantile[0],
        "q50": quantile[1],
        "q90": quantile[2],
        "positive_rate": float((values > 0).float().mean().item()),
    }


def predicted_correct(classes, gt_labels):
    return gt_labels.gather(-1, classes.unsqueeze(-1)).squeeze(-1).bool()


def selection_summary(selected, classes, gt_labels, universe):
    selected = selected & universe
    gt_action = gt_labels.sum(dim=-1) > 0
    gt_background = ~gt_action
    correct = predicted_correct(classes, gt_labels)
    selected_count = int(selected.sum().item())
    correct_count = int((selected & correct).sum().item())
    return {
        "n": selected_count,
        "gt_action": int((selected & gt_action).sum().item()),
        "gt_background": int((selected & gt_background).sum().item()),
        "correct_action": correct_count,
        "correct_routing_precision": ratio(correct_count, selected_count),
        "low_action_correct_recall": ratio(
            correct_count, int((universe & gt_action).sum().item())
        ),
        "background_false_recovery_rate": ratio(
            int((selected & gt_background).sum().item()),
            int((universe & gt_background).sum().item()),
        ),
    }


def exact_top_fraction(universe, ranking, fraction):
    indices = torch.where(universe)[0]
    selected = torch.zeros_like(universe)
    if not indices.numel():
        return selected
    keep = max(1, int(math.ceil(fraction * indices.numel())))
    order = torch.argsort(ranking[indices], descending=True)
    selected[indices[order[:keep]]] = True
    return selected


def summarize_group(records, universe, alpha):
    gt_labels = records["gt_labels"]
    gt_action = gt_labels.sum(dim=-1) > 0
    gt_background = ~gt_action
    base_prob = records["base_probability"]
    base_score, base_class = base_prob.max(dim=-1)
    base_correct = predicted_correct(base_class, gt_labels)

    view_prob = records["view_probability"]
    view_score, view_class = view_prob.max(dim=-1)
    base_score_views = base_score.unsqueeze(0).expand_as(view_score)
    score_delta = view_score - base_score_views
    class_preserved = view_class == base_class.unsqueeze(0)
    representation_mse = records["representation_mse"]

    best_view_index = view_score.argmax(dim=0)
    gather_index = best_view_index.view(1, -1)
    best_score = view_score.gather(0, gather_index).squeeze(0)
    best_class = view_class.gather(0, gather_index).squeeze(0)
    best_delta = best_score - base_score

    stable_scores = view_score.masked_fill(~class_preserved, -1.0)
    stable_view_index = stable_scores.argmax(dim=0)
    stable_score = stable_scores.gather(
        0, stable_view_index.view(1, -1)
    ).squeeze(0)
    stable_available = class_preserved.any(dim=0)
    stable_class = base_class
    max_representation_mse = representation_mse.max(dim=0).values

    output = {
        "n": int(universe.sum().item()),
        "gt_action_rate": (
            float(gt_action[universe].float().mean().item())
            if universe.any()
            else None
        ),
        "base_top1_correct_rate": (
            float(base_correct[universe].float().mean().item())
            if universe.any()
            else None
        ),
        "modes": {},
        "operational_cross_alpha": {},
        "matched_budget": {},
    }
    for mode_index, mode in enumerate(MASK_MODES):
        delta = score_delta[mode_index]
        mode_class = view_class[mode_index]
        output["modes"][mode] = {
            "class_preserved_rate": (
                float(class_preserved[mode_index][universe].float().mean().item())
                if universe.any()
                else None
            ),
            "score_delta_gt_action": distribution(delta[universe & gt_action]),
            "score_delta_gt_background": distribution(
                delta[universe & gt_background]
            ),
            "score_delta_base_top1_correct_action": distribution(
                delta[universe & gt_action & base_correct]
            ),
            "representation_mse_gt_action": distribution(
                representation_mse[mode_index][universe & gt_action]
            ),
            "representation_mse_gt_background": distribution(
                representation_mse[mode_index][universe & gt_background]
            ),
        }
        output["operational_cross_alpha"][mode] = selection_summary(
            view_score[mode_index] >= alpha,
            mode_class,
            gt_labels,
            universe,
        )
    output["operational_cross_alpha"]["any_view"] = selection_summary(
        best_score >= alpha, best_class, gt_labels, universe
    )
    output["operational_cross_alpha"]["same_base_class_any_view"] = (
        selection_summary(
            stable_available & (stable_score >= alpha),
            stable_class,
            gt_labels,
            universe,
        )
    )

    rankings = {
        "original_confidence": (base_score, base_class),
        "best_masked_score": (best_score, best_class),
        "largest_masked_score_gain": (best_delta, best_class),
        "same_base_class_masked_score": (stable_score, stable_class),
        "largest_representation_mse": (
            max_representation_mse,
            base_class,
        ),
    }
    for fraction in MATCHED_BUDGET_FRACTIONS:
        key = "top_{:03d}".format(int(round(fraction * 100)))
        output["matched_budget"][key] = {}
        for name, (ranking, classes) in rankings.items():
            selected = exact_top_fraction(universe, ranking, fraction)
            if name == "same_base_class_masked_score":
                selected &= stable_available
            output["matched_budget"][key][name] = selection_summary(
                selected, classes, gt_labels, universe
            )
    return output


def forward_target(model, inputs, masks):
    strict = model._stage1_strict_masks(masks)
    feats, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn, fpn_masks = model.neck_target(feats, backbone_masks)
    logits, head_features = model.cls_head_target(fpn, fpn_masks)
    probabilities = [value.permute(0, 2, 1).sigmoid() for value in logits]
    representations = [value.permute(0, 2, 1) for value in head_features]
    return fpn, fpn_masks, strict, probabilities, representations


def gap_masks(base_score, alpha):
    gap = alpha - base_score
    return {
        "le_0.02": gap <= 0.02,
        "0.02_0.05": (gap > 0.02) & (gap <= 0.05),
        "0.05_0.10": (gap > 0.05) & (gap <= 0.10),
        "gt_0.10": gap > 0.10,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--guard-radii", type=int, nargs="+", default=[9, 25, 57])
    parser.add_argument("--sample-rate", type=float, default=0.01)
    parser.add_argument("--mask-batch-size", type=int, default=2)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if len(args.levels) != len(args.guard_radii):
        raise ValueError("--levels and --guard-radii must have equal length")
    if not 0.0 < args.sample_rate <= 1.0:
        raise ValueError("--sample-rate must be in (0, 1]")
    guard_by_level = dict(zip(args.levels, args.guard_radii))

    from libs.datasets import make_dataset
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_low_confidence_topk_consensus import load_models
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
    video_count = min(len(dataset), args.max_videos or len(dataset))
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    models, checkpoint_metadata = load_models(
        cfg, [("base", args.checkpoint, "state_dict")], args.device
    )
    model = models["base"]

    record_lists = {
        "base_probability": [],
        "base_representation": [],
        "view_probability": {mode: [] for mode in MASK_MODES},
        "view_representation": {mode: [] for mode in MASK_MODES},
        "gt_labels": [],
        "level": [],
    }
    counts = {
        "ordinary_valid": 0,
        "strict_valid": 0,
        "low_confidence_strict_with_bilateral_remote_context": 0,
        "sampled": 0,
    }

    with torch.inference_mode():
        for video_index in range(video_count):
            raw_video = dataset[video_index]
            video = move_video(raw_video, args.device)
            inputs, input_masks = model.preprocessing([video], mode="validation")
            (
                base_fpn,
                base_fpn_masks,
                base_strict,
                base_probabilities,
                base_representations,
            ) = forward_target(model, inputs, input_masks)
            points = model.point_generator(base_fpn)
            valid_length = int(video["feats"].shape[-1])
            video_id = video["metadata"]["video_id"]
            crop_start = int(video["metadata"].get("crop_start", 0))

            candidates = []
            label_start = 0
            label_starts = {}
            for level, point_grid in enumerate(points):
                level_length = point_grid.shape[0]
                label_starts[level] = label_start
                label_start += level_length
                if level not in guard_by_level:
                    continue
                score = base_probabilities[level][0].max(dim=-1).values
                ordinary_valid = base_fpn_masks[level][0, 0]
                strict_valid = base_strict[level][0, 0]
                radius = guard_by_level[level]
                centers = point_grid[:, 0].long()
                has_both_remote = (
                    (centers - radius > 0)
                    & (centers + radius + 1 < valid_length)
                )
                eligible = (
                    ordinary_valid
                    & strict_valid
                    & has_both_remote
                    & (score < alpha)
                )
                counts["ordinary_valid"] += int(ordinary_valid.sum().item())
                counts["strict_valid"] += int(strict_valid.sum().item())
                counts[
                    "low_confidence_strict_with_bilateral_remote_context"
                ] += int(eligible.sum().item())
                for anchor_index in torch.where(eligible)[0].tolist():
                    if deterministic_keep(
                        args.seed,
                        video_id,
                        crop_start,
                        level,
                        anchor_index,
                        args.sample_rate,
                    ):
                        candidates.append({
                            "level": level,
                            "anchor_index": anchor_index,
                            "center": int(centers[anchor_index].item()),
                            "radius": radius,
                            "base_probability": base_probabilities[level][0, anchor_index].cpu(),
                            "base_representation": base_representations[level][0, anchor_index].cpu(),
                        })
            if not candidates:
                continue

            for mode in MASK_MODES:
                for chunk_start in range(0, len(candidates), args.mask_batch_size):
                    chunk = candidates[
                        chunk_start:chunk_start + args.mask_batch_size
                    ]
                    masked_inputs, masked_masks = build_remote_context_batch(
                        inputs,
                        input_masks,
                        [item["center"] for item in chunk],
                        [item["radius"] for item in chunk],
                        [mode] * len(chunk),
                    )
                    (
                        _,
                        masked_fpn_masks,
                        masked_strict,
                        masked_probabilities,
                        masked_representations,
                    ) = forward_target(model, masked_inputs, masked_masks)
                    for row, item in enumerate(chunk):
                        level = item["level"]
                        anchor_index = item["anchor_index"]
                        if not bool(masked_fpn_masks[level][row, 0, anchor_index]):
                            raise RuntimeError("masked anchor became invalid")
                        if not bool(masked_strict[level][row, 0, anchor_index]):
                            raise RuntimeError(
                                "guard radius does not preserve local support"
                            )
                        item[mode + "_probability"] = (
                            masked_probabilities[level][row, anchor_index].cpu()
                        )
                        item[mode + "_representation"] = (
                            masked_representations[level][row, anchor_index].cpu()
                        )

            segments, labels = crop_targets(
                annotations[video_id],
                video["metadata"],
                video["feats"].shape[-1],
                cfg,
                args.device,
            )
            if not torch.equal(segments, raw_video["segments"].to(args.device)) or not torch.equal(
                labels, raw_video["labels"].to(args.device)
            ):
                raise RuntimeError("target GT reconstruction mismatch")
            gt_cls, _ = model.label_points(points, [segments], [labels])
            gt_cls = gt_cls[0]

            for item in candidates:
                level = item["level"]
                gt_index = label_starts[level] + item["anchor_index"]
                record_lists["base_probability"].append(item["base_probability"])
                record_lists["base_representation"].append(item["base_representation"])
                record_lists["gt_labels"].append(gt_cls[gt_index].bool().cpu())
                record_lists["level"].append(level)
                for mode in MASK_MODES:
                    record_lists["view_probability"][mode].append(
                        item[mode + "_probability"]
                    )
                    record_lists["view_representation"][mode].append(
                        item[mode + "_representation"]
                    )
            counts["sampled"] += len(candidates)

    if not record_lists["gt_labels"]:
        raise RuntimeError("deterministic sampling selected no anchors")
    base_probability = torch.stack(record_lists["base_probability"])
    base_representation = torch.stack(record_lists["base_representation"])
    view_probability = torch.stack([
        torch.stack(record_lists["view_probability"][mode])
        for mode in MASK_MODES
    ])
    view_representation = torch.stack([
        torch.stack(record_lists["view_representation"][mode])
        for mode in MASK_MODES
    ])
    records = {
        "base_probability": base_probability,
        "view_probability": view_probability,
        "representation_mse": (
            view_representation - base_representation.unsqueeze(0)
        ).square().mean(dim=-1),
        "gt_labels": torch.stack(record_lists["gt_labels"]),
        "level": torch.tensor(record_lists["level"], dtype=torch.long),
    }
    universe = torch.ones(records["level"].shape[0], dtype=torch.bool)
    base_score, base_class = base_probability.max(dim=-1)
    gt_action = records["gt_labels"].sum(dim=-1) > 0
    base_correct = predicted_correct(base_class, records["gt_labels"])

    result = {
        "audit": "exact_anchor_centered_remote_context_masks",
        "seed": args.seed,
        "videos": video_count,
        "alpha": alpha,
        "levels": args.levels,
        "guard_radii_input_feature_cells": guard_by_level,
        "sample_rate": args.sample_rate,
        "checkpoint": checkpoint_metadata["base"],
        "protocol": {
            "cohort": "base student score < alpha; base strict-valid; remote context exists on both sides",
            "sampling": "deterministic SHA1 anchor sampling without target GT",
            "past": "invalidate all input cells before the intact local core",
            "future": "invalidate all input cells after the intact local core",
            "bilateral": "retain only the intact local core",
            "masked_value": "zero with the model's native validity mask set false",
            "target_gt": "joined only after original and all masked predictions are frozen",
        },
        "counts": counts,
        "sample_composition": {
            "gt_action": int(gt_action.sum().item()),
            "gt_background": int((~gt_action).sum().item()),
            "base_top1_correct_action": int(base_correct.sum().item()),
        },
        "global": summarize_group(records, universe, alpha),
        "by_level": {
            str(level): summarize_group(
                records, universe & (records["level"] == level), alpha
            )
            for level in args.levels
        },
        "by_base_score_gap": {
            name: summarize_group(records, universe & mask, alpha)
            for name, mask in gap_masks(base_score, alpha).items()
        },
        "deep_low_top1_correct_action": {
            mode: {
                "n": int((gt_action & base_correct & (alpha - base_score > 0.1)).sum().item()),
                "score_delta": distribution(
                    (
                        view_probability[mode_index].max(dim=-1).values
                        - base_score
                    )[gt_action & base_correct & (alpha - base_score > 0.1)]
                ),
            }
            for mode_index, mode in enumerate(MASK_MODES)
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "counts": counts,
        "sample_composition": result["sample_composition"],
        "operational_cross_alpha": result["global"]["operational_cross_alpha"],
    }, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
