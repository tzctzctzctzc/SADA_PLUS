"""Audit target conditional-alignment gradients against target GT task loss."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def parse_levels(value):
    levels = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("levels must be a non-empty unique comma-separated list")
    return levels


def conditional_target_coefficients(
    pseudo_class,
    num_classes,
    background_weight,
    global_scale,
):
    """Return the exact class-balanced coefficient for every target anchor."""
    counts = torch.bincount(pseudo_class, minlength=num_classes)
    present_classes = int((counts > 0).sum().item())
    if present_classes == 0:
        return pseudo_class.new_zeros(pseudo_class.shape, dtype=torch.float32), 0
    class_weights = torch.ones(
        num_classes, device=pseudo_class.device, dtype=torch.float32
    )
    class_weights[-1] = float(background_weight)
    coefficients = (
        float(global_scale)
        * class_weights[pseudo_class]
        / present_classes
        / counts[pseudo_class].float()
    )
    return coefficients, present_classes


def first_order_feature_influence(
    oracle_gradient,
    domain_gradient,
    coefficients,
    beta,
):
    """Estimate GT-loss change from one target GRL feature update."""
    update = (
        float(beta) * coefficients.unsqueeze(-1) * domain_gradient
    )
    influence = (oracle_gradient * update).sum(dim=-1)
    cosine = influence / (
        oracle_gradient.norm(dim=-1)
        * update.norm(dim=-1)
    ).clamp_min(1e-12)
    return update, influence, cosine


def _mean(values):
    return float(values.mean().item()) if values.numel() else None


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize_group(records, mask, denominator=None):
    mask = mask.bool()
    count = int(mask.sum().item())
    denominator = mask.numel() if denominator is None else denominator
    if count == 0:
        return {
            "n": 0,
            "selection_rate": _ratio(0, denominator),
            "alignment_correct_rate": None,
            "harmful_rate": None,
            "domain_loss_mean": None,
            "confidence_mean": None,
            "oracle_cls_loss_mean": None,
            "oracle_gradient_norm_mean": None,
            "domain_update_norm_mean": None,
            "cosine_mean": None,
            "influence_mean": None,
            "net_predicted_delta": 0.0,
            "positive_harm": 0.0,
            "negative_benefit": 0.0,
            "positive_magnitude_fraction": None,
        }

    influence = records["influence"][mask]
    positive = float(influence.clamp_min(0).sum().item())
    negative = float((-influence.clamp_max(0)).sum().item())
    return {
        "n": count,
        "selection_rate": _ratio(count, denominator),
        "alignment_correct_rate": _mean(
            records["alignment_correct"][mask].float()
        ),
        "harmful_rate": _mean((influence > 0).float()),
        "domain_loss_mean": _mean(records["domain_loss"][mask]),
        "confidence_mean": _mean(records["confidence"][mask]),
        "oracle_cls_loss_mean": _mean(records["oracle_cls_loss"][mask]),
        "oracle_gradient_norm_mean": _mean(
            records["oracle_gradient_norm"][mask]
        ),
        "domain_update_norm_mean": _mean(
            records["domain_update_norm"][mask]
        ),
        "cosine_mean": _mean(records["cosine"][mask]),
        "influence_mean": _mean(influence),
        "net_predicted_delta": float(influence.sum().item()),
        "positive_harm": positive,
        "negative_benefit": negative,
        "positive_magnitude_fraction": _ratio(
            positive, positive + negative
        ),
    }


def summarize_quantiles(records, mask, key, bins=5):
    selected = torch.where(mask.bool())[0]
    if selected.numel() == 0:
        return []
    order = torch.argsort(records[key][selected])
    chunks = torch.tensor_split(selected[order], min(bins, selected.numel()))
    output = []
    for index, chunk in enumerate(chunks):
        chunk_mask = torch.zeros_like(mask, dtype=torch.bool)
        chunk_mask[chunk] = True
        values = records[key][chunk]
        summary = summarize_group(records, chunk_mask, selected.numel())
        summary.update({
            "quantile": index + 1,
            "value_min": float(values.min().item()),
            "value_mean": float(values.mean().item()),
            "value_max": float(values.max().item()),
        })
        output.append(summary)
    return output


def summarize_records(records):
    is_action = records["is_action"].bool()
    gt_action = records["gt_action"].bool()
    correct_class = records["correct_class"].bool()
    all_mask = torch.ones_like(is_action, dtype=torch.bool)
    groups = {
        "all": all_mask,
        "action": is_action,
        "background": ~is_action,
        "correct_alignment": records["alignment_correct"].bool(),
        "incorrect_alignment": ~records["alignment_correct"].bool(),
        "correct_action": is_action & correct_class,
        "wrong_class_action": is_action & gt_action & ~correct_class,
        "false_action": is_action & ~gt_action,
        "missed_action_background": ~is_action & gt_action,
        "true_background": ~is_action & ~gt_action,
    }
    denominator = len(is_action)
    return {
        "groups": {
            name: summarize_group(records, mask, denominator)
            for name, mask in groups.items()
        },
        "domain_loss_quantiles": summarize_quantiles(
            records, all_mask, "domain_loss"
        ),
        "action_confidence_quantiles": summarize_quantiles(
            records, is_action, "confidence"
        ),
        "domain_update_norm_quantiles": summarize_quantiles(
            records, all_mask, "domain_update_norm"
        ),
    }


def _oracle_task_gradients(model, fpn_feats, fpn_masks, points, videos, cfg):
    from libs.models.SADA.losses_aux import (
        ctr_diou_loss_1d,
        sigmoid_focal_loss,
    )

    if model.use_trident_head:
        raise ValueError("the influence audit currently requires use_trident_head=False")

    gt_cls_list, gt_offsets_list = model.label_points(
        points,
        [video["segments"] for video in videos],
        [video["labels"] for video in videos],
    )
    gt_cls = torch.stack(gt_cls_list)
    gt_offsets = torch.stack(gt_offsets_list)
    valid_per_level = [mask.squeeze(1).bool() for mask in fpn_masks]
    valid = torch.cat(valid_per_level, dim=1)
    positive = (gt_cls.sum(dim=-1) > 0) & valid
    normalizer = max(int(positive.sum().item()), 1)

    raw_logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
    logits_per_level = [value.permute(0, 2, 1) for value in raw_logits]
    logits = torch.cat(logits_per_level, dim=1)
    focal = sigmoid_focal_loss(
        logits,
        gt_cls,
        reduction="none",
        alpha=0.25,
        gamma=float(cfg["model"].get("focal_loss_gamma", 2.0)),
    ).sum(dim=-1)
    cls_loss = focal[valid].sum() / normalizer

    raw_offsets, _ = model.reg_head_target(fpn_feats, fpn_masks)
    offsets_per_level = [value.permute(0, 2, 1) for value in raw_offsets]
    offsets = torch.cat(offsets_per_level, dim=1)
    if positive.any():
        reg_loss = ctr_diou_loss_1d(
            offsets[positive], gt_offsets[positive], reduction="sum"
        ) / normalizer
    else:
        reg_loss = offsets.sum() * 0.0

    reg_weight = float(model.train_loss_reg_weight)
    if reg_weight <= 0:
        reg_weight = float(cls_loss.detach().item()) / max(
            float(reg_loss.detach().item()), 0.01
        )
    downstream_weight = float(
        cfg["model"]["loss_weights"].get("weight_downstream", 1.0)
    )
    oracle_loss = downstream_weight * (cls_loss + reg_weight * reg_loss)
    gradients = torch.autograd.grad(oracle_loss, fpn_feats)

    gt_per_level = []
    focal_per_level = []
    start = 0
    for point_grid in points:
        length = point_grid.shape[0]
        gt_per_level.append(gt_cls[:, start:start + length])
        focal_per_level.append(
            downstream_weight * focal[:, start:start + length] / normalizer
        )
        start += length
    return {
        "gradients": gradients,
        "logits": tuple(value.detach() for value in logits_per_level),
        "gt_cls": tuple(gt_per_level),
        "oracle_cls_loss": tuple(value.detach() for value in focal_per_level),
        "oracle_loss": float(oracle_loss.detach().item()),
        "cls_loss": float(cls_loss.detach().item()),
        "reg_loss": float(reg_loss.detach().item()),
        "positive": int(positive.sum().item()),
    }


def main():
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import move_video

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument(
        "--weights",
        choices=("state_dict", "ema_state_dict"),
        default="state_dict",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    levels = parse_levels(args.levels)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg.setdefault("diagnostic", {}).update({
        "selective_alignment": "off",
        "conditional_paired_support": False,
        "conditional_paired_fixed_denominator": False,
    })
    cfg["model"]["semantic_alignment_levels"] = list(levels)
    _update_config(cfg)

    split = (
        cfg["tgt_train_split"]
        if args.split == "train"
        else cfg["tgt_val_split"]
    )
    dataset = make_dataset(
        cfg["dataset_name"], args.split == "train", split, **cfg["dataset"]
    )
    total_videos = len(dataset)
    if args.max_videos is not None:
        total_videos = min(total_videos, args.max_videos)
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if args.weights not in checkpoint:
        raise ValueError("checkpoint has no {}".format(args.weights))
    model.load_state_dict(checkpoint[args.weights])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    record_lists = {
        key: []
        for key in (
            "level",
            "is_action",
            "gt_action",
            "correct_class",
            "alignment_correct",
            "confidence",
            "domain_loss",
            "oracle_cls_loss",
            "oracle_gradient_norm",
            "domain_update_norm",
            "influence",
            "cosine",
            "coefficient",
        )
    }
    batch_stats = []
    threshold = float(cfg["model"].get("pl_threshold", 0.7))
    num_action_classes = int(model.num_class_orig)
    num_alignment_classes = num_action_classes + int(model.align_bkg)
    if not model.align_bkg:
        raise ValueError("the audit currently requires align_bkg=True")
    beta = float(cfg["model"].get("beta", 1.0))
    weight_adv = float(cfg["model"]["loss_weights"].get("weight_adv", 1.0))
    weight_target = float(cfg["model"].get("weight_target_cond_adv", 1.0))
    background_weight = float(cfg["model"].get("weight_no_class_adv", 1.0))

    for batch_start in range(0, total_videos, batch_size):
        raw_videos = [
            dataset[index]
            for index in range(
                batch_start, min(batch_start + batch_size, total_videos)
            )
        ]
        videos = [move_video(video, args.device) for video in raw_videos]
        with torch.no_grad():
            inputs, masks = model.preprocessing(videos, mode="validation")
            backbone_feats, backbone_masks, _ = model.backbone_target(
                inputs, masks
            )
            frozen_fpn_feats, fpn_masks = model.neck_target(
                backbone_feats, backbone_masks
            )
        fpn_feats = tuple(
            value.detach().requires_grad_(True) for value in frozen_fpn_feats
        )
        points = model.point_generator(fpn_feats)
        task = _oracle_task_gradients(
            model, fpn_feats, fpn_masks, points, videos, cfg
        )
        batch_stats.append({
            "batch_start": batch_start,
            "videos": len(videos),
            "oracle_loss": task["oracle_loss"],
            "cls_loss": task["cls_loss"],
            "reg_loss": task["reg_loss"],
            "positive": task["positive"],
        })

        for level in levels:
            valid = fpn_masks[level].squeeze(1).bool().reshape(-1)
            features = fpn_feats[level].permute(0, 2, 1).reshape(
                -1, fpn_feats[level].shape[1]
            )
            valid_features = features[valid]
            probabilities = task["logits"][level].reshape(
                -1, num_action_classes
            ).sigmoid()
            confidence, pseudo_action_class = probabilities.max(dim=-1)
            confidence = confidence[valid]
            pseudo_action_class = pseudo_action_class[valid]
            is_action = confidence >= threshold
            pseudo_alignment_class = torch.where(
                is_action,
                pseudo_action_class,
                torch.full_like(pseudo_action_class, num_action_classes),
            )

            conditioned = torch.cat(
                [valid_features, model.learnable_embeddings[pseudo_alignment_class]],
                dim=-1,
            )
            discriminator = (
                model.disc_dom_conditional_adv[level]
                if cfg["model"].get("use_level_disc", False)
                else model.disc_dom_conditional_adv[0]
            )
            domain_logits = discriminator(conditioned).squeeze(-1)
            domain_loss = F.binary_cross_entropy_with_logits(
                domain_logits,
                torch.ones_like(domain_logits),
                reduction="none",
            )
            domain_gradient = torch.autograd.grad(
                domain_loss.sum(), valid_features
            )[0]

            level_scale = (
                weight_adv
                * weight_target
                * float(cfg["model"].get("weight_level_{}".format(level), 1.0))
                / len(levels)
            )
            coefficients, _ = conditional_target_coefficients(
                pseudo_alignment_class,
                num_alignment_classes,
                background_weight,
                level_scale,
            )
            oracle_gradient = task["gradients"][level].permute(
                0, 2, 1
            ).reshape(-1, fpn_feats[level].shape[1])[valid]
            update, influence, cosine = first_order_feature_influence(
                oracle_gradient,
                domain_gradient,
                coefficients,
                beta,
            )

            labels = task["gt_cls"][level].reshape(
                -1, num_action_classes
            )[valid]
            gt_action = labels.sum(dim=-1) > 0
            correct_class = labels.gather(
                -1, pseudo_action_class.unsqueeze(-1)
            ).squeeze(-1) > 0
            alignment_correct = torch.where(
                is_action, correct_class, ~gt_action
            )
            oracle_cls_loss = task["oracle_cls_loss"][level].reshape(-1)[valid]
            values = {
                "level": torch.full_like(pseudo_action_class, level),
                "is_action": is_action,
                "gt_action": gt_action,
                "correct_class": correct_class,
                "alignment_correct": alignment_correct,
                "confidence": confidence,
                "domain_loss": domain_loss.detach(),
                "oracle_cls_loss": oracle_cls_loss,
                "oracle_gradient_norm": oracle_gradient.norm(dim=-1),
                "domain_update_norm": update.norm(dim=-1),
                "influence": influence,
                "cosine": cosine,
                "coefficient": coefficients,
            }
            for key, value in values.items():
                record_lists[key].append(value.detach().cpu())

        if batch_start % max(batch_size * 25, 1) == 0:
            print("processed {}/{} videos".format(
                min(batch_start + len(videos), total_videos), total_videos
            ))

    records = {
        key: torch.cat(values) for key, values in record_lists.items()
    }
    per_level = {}
    for level in levels:
        selected = records["level"] == level
        per_level[str(level)] = summarize_records({
            key: value[selected] for key, value in records.items()
        })

    result = {
        "audit": "conditional_gradient_influence_target_gt",
        "target_gt_usage": "offline_gradient_probe_only",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "config": args.config,
        "split": "target_{}".format(args.split),
        "seed": args.seed,
        "levels": list(levels),
        "videos": total_videos,
        "batch_size": batch_size,
        "model_mode": "eval",
        "probe": {
            "oracle_task": "target GT focal classification plus DIoU regression",
            "domain_update": (
                "target conditional BCE with original class means, level weights, "
                "weight_adv, weight_target, and GRL beta"
            ),
            "harmful_definition": (
                "positive influence predicts increased target GT task loss after "
                "one local FPN-space target conditional update"
            ),
            "limitation": (
                "first-order FPN-feature approximation; shared-parameter Jacobians, "
                "source conditional gradients, and optimizer state are not modeled"
            ),
        },
        "global": summarize_records(records),
        "per_level": per_level,
        "batch_task": {
            "oracle_loss_mean": float(np.mean([
                row["oracle_loss"] for row in batch_stats
            ])),
            "cls_loss_mean": float(np.mean([
                row["cls_loss"] for row in batch_stats
            ])),
            "reg_loss_mean": float(np.mean([
                row["reg_loss"] for row in batch_stats
            ])),
            "positive_total": int(sum(row["positive"] for row in batch_stats)),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["global"]["groups"], indent=2))
    print("wrote", output_path)


if __name__ == "__main__":
    main()
