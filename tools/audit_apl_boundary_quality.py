"""APL/ALQA-style frozen-feature quality-transfer target-GT audit."""

import argparse
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage2_utils import (
    aligned_tiou_tnd,
    match_post_nms_to_pre_nms,
    matched_budget_indices,
)
from libs.utils import batched_nms
from libs.utils.utils import _update_config, read_yaml
from tools.audit_npl_boundary_reliability import (
    pearson,
    rank_values,
    reset_audit_rng,
    select_within_fpn_level,
    summarize_anchor_selection,
    summarize_selection,
)
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_stage2_causal import (
    aligned_iou,
    decode_segments,
    pairwise_iou,
    ratio,
)


class APLQualityHead(nn.Module):
    """The two final tIoU/tND branches on frozen regression features."""

    def __init__(self, feature_dim):
        super().__init__()
        self.output = nn.Conv1d(feature_dim, 2, kernel_size=3, padding=1)

    def forward(self, features):
        return self.output(features)


def parse_numbers(value, cast):
    return [cast(item) for item in value.split(",") if item]


def regression_hidden_and_offsets(model, fpn_feats, fpn_masks):
    hidden, offsets = [], []
    head = model.reg_head_target
    for level, (features, mask) in enumerate(zip(fpn_feats, fpn_masks)):
        current = features
        for index in range(len(head.head)):
            current, _ = head.head[index](current, mask)
            current = head.act(head.norm[index](current))
        level_offsets, _ = head.offset_head(current, mask)
        offsets.append(F.relu(head.scale[level](level_offsets)))
        hidden.append(current)
    return tuple(hidden), tuple(offsets)


def encode_batch(model, batch, mode):
    inputs, masks = model.preprocessing(batch, mode=mode)
    backbone_features, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(
        backbone_features, backbone_masks
    )
    hidden, offsets = regression_hidden_and_offsets(
        model, fpn_features, fpn_masks
    )
    points = model.point_generator(fpn_features)
    return hidden, offsets, fpn_masks, points


def collect_source_sequences(
        model, dataset, device, levels, batch_size, seed, mode):
    reset_audit_rng(seed)
    sequences = []
    positive_anchors = 0
    for batch_start in range(0, len(dataset), batch_size):
        raw_batch = [
            dataset[index]
            for index in range(
                batch_start, min(batch_start + batch_size, len(dataset))
            )
        ]
        batch = [move_video(video, device) for video in raw_batch]
        with torch.no_grad():
            hidden, offsets, fpn_masks, points = encode_batch(
                model, batch, mode
            )
            gt_cls, gt_offsets = model.label_points(
                points,
                [video["segments"] for video in batch],
                [video["labels"] for video in batch],
            )

        point_offset = 0
        for level, point in enumerate(points):
            next_offset = point_offset + point.shape[0]
            if level not in levels:
                point_offset = next_offset
                continue
            targets = torch.zeros(
                (len(batch), 2, point.shape[0]),
                dtype=torch.float32, device=device,
            )
            positive = torch.zeros(
                (len(batch), point.shape[0]),
                dtype=torch.bool, device=device,
            )
            centers = point[:, 0]
            strides = point[:, 3]
            for batch_index in range(len(batch)):
                gt_positive = (
                    gt_cls[batch_index][
                        point_offset:next_offset
                    ].sum(dim=-1) > 0
                ) & fpn_masks[level][batch_index, 0]
                positive[batch_index] = gt_positive
                if not gt_positive.any():
                    continue
                predicted_segments = decode_segments(
                    offsets[level][batch_index].transpose(0, 1),
                    centers, strides,
                )
                target_segments = decode_segments(
                    gt_offsets[batch_index][point_offset:next_offset],
                    centers, strides,
                )
                tiou, tnd = aligned_tiou_tnd(
                    predicted_segments[gt_positive],
                    target_segments[gt_positive],
                )
                targets[batch_index, 0, gt_positive] = tiou
                targets[batch_index, 1, gt_positive] = tnd
            if positive.any():
                sequences.append({
                    "features": hidden[level].detach().cpu(),
                    "targets": targets.cpu(),
                    "positive": positive.cpu(),
                    "level": level,
                })
                positive_anchors += int(positive.sum())
            point_offset = next_offset
    return sequences, positive_anchors


def evaluate_quality_head(head, sequences, device):
    head.eval()
    loss_sum = 0.0
    count = 0
    predictions, targets = [], []
    with torch.no_grad():
        for sequence in sequences:
            features = sequence["features"].to(device)
            target = sequence["targets"].to(device)
            positive = sequence["positive"].to(device)
            logits = head(features).permute(0, 2, 1)[positive]
            selected_target = target.permute(0, 2, 1)[positive]
            loss_sum += float(F.binary_cross_entropy_with_logits(
                logits, selected_target, reduction="sum"
            ))
            count += int(selected_target.numel())
            predictions.append(logits.sigmoid().cpu())
            targets.append(selected_target.cpu())
    prediction = torch.cat(predictions) if predictions else torch.empty((0, 2))
    target = torch.cat(targets) if targets else torch.empty((0, 2))
    predicted_diou = prediction[:, 0] - prediction[:, 1]
    target_diou = target[:, 0] - target[:, 1]
    return {
        "bce": ratio(loss_sum, count),
        "positive_anchors": int(target.shape[0]),
        "tiou_mae": (
            float((prediction[:, 0] - target[:, 0]).abs().mean())
            if target.numel() else None
        ),
        "tnd_mae": (
            float((prediction[:, 1] - target[:, 1]).abs().mean())
            if target.numel() else None
        ),
        "diou_pearson": pearson(
            predicted_diou.tolist(), target_diou.tolist()
        ),
        "diou_spearman": pearson(
            rank_values(predicted_diou.tolist()),
            rank_values(target_diou.tolist()),
        ),
    }


def train_quality_head(
        train_sequences, val_sequences, feature_dim, device,
        seed, max_epochs, patience, learning_rate):
    reset_audit_rng(seed)
    head = APLQualityHead(feature_dim).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    best_val = float("inf")
    remaining_patience = patience
    history = []

    for epoch in range(1, max_epochs + 1):
        head.train()
        order = list(range(len(train_sequences)))
        random.Random(seed + epoch).shuffle(order)
        train_loss = 0.0
        train_count = 0
        for index in order:
            sequence = train_sequences[index]
            features = sequence["features"].to(device)
            target = sequence["targets"].to(device)
            positive = sequence["positive"].to(device)
            logits = head(features).permute(0, 2, 1)[positive]
            selected_target = target.permute(0, 2, 1)[positive]
            loss = F.binary_cross_entropy_with_logits(
                logits, selected_target
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * selected_target.numel()
            train_count += int(selected_target.numel())

        val_metrics = evaluate_quality_head(head, val_sequences, device)
        history.append({
            "epoch": epoch,
            "train_bce": ratio(train_loss, train_count),
            "val_bce": val_metrics["bce"],
        })
        if val_metrics["bce"] < best_val - 1e-6:
            best_val = val_metrics["bce"]
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    head.load_state_dict(best_state)
    return head, {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_sequences": len(train_sequences),
        "validation": evaluate_quality_head(head, val_sequences, device),
        "history": history,
    }


def matched_budget_summary(
        records, ranking_values, fractions, level_key, summarize):
    output = {"global": {}, "within_fpn_level": {}}
    for fraction in fractions:
        key = "{:.2f}".format(fraction)
        output["global"][key] = {}
        output["within_fpn_level"][key] = {}
        expected = min(
            len(records), max(1, math.ceil(len(records) * fraction))
        )
        for name, values in ranking_values.items():
            indices = matched_budget_indices(values, fraction).tolist()
            if len(indices) != expected:
                raise RuntimeError("APL matched budget size drifted")
            output["global"][key][name] = summarize(indices)
            stratified = select_within_fpn_level(
                records, values, fraction, level_key=level_key
            )
            output["within_fpn_level"][key][name] = summarize(stratified)
    return output


def collect_target_records(
        model, student_model, quality_heads, dataset, annotations, cfg, device,
        levels, batch_size, seed):
    reset_audit_rng(seed)
    proposal_records, anchor_records = [], []
    total_gt_segments = 0
    total_gt_anchors = 0
    videos = 0

    for batch_start in range(0, len(dataset), batch_size):
        raw_videos = [
            dataset[index]
            for index in range(
                batch_start, min(batch_start + batch_size, len(dataset))
            )
        ]
        batch = [move_video(video, device) for video in raw_videos]
        with torch.no_grad():
            post_nms, _ = model.generate_stage2_pseudo_targets(
                batch,
                return_pre_nms=True,
                return_npl_details=True,
                npl_neighbor_radii=(1,),
            )
            hidden, offsets, fpn_masks, points = encode_batch(
                model, batch, "train"
            )
            _, student_level_offsets, _, _ = encode_batch(
                student_model, batch, "train"
            )
            quality_maps = {
                str(head_seed): [
                    head(features).sigmoid()
                    for features in hidden
                ]
                for head_seed, head in quality_heads.items()
            }

        valid = torch.cat([
            mask.squeeze(1) for mask in fpn_masks
        ], dim=1)
        enabled = torch.cat([
            torch.full(
                (len(batch), point.shape[0]), level in levels,
                dtype=torch.bool, device=device,
            )
            for level, point in enumerate(points)
        ], dim=1)
        centers = torch.cat([point[:, 0] for point in points])
        strides = torch.cat([point[:, 3] for point in points])
        anchor_levels = torch.cat([
            torch.full(
                (point.shape[0],), level,
                dtype=torch.long, device=device,
            )
            for level, point in enumerate(points)
        ])
        student_offsets = torch.cat([
            level_offsets.permute(0, 2, 1)
            for level_offsets in student_level_offsets
        ], dim=1)

        gt_segments, gt_labels = [], []
        for raw_video, video in zip(raw_videos, batch):
            video_id = raw_video["metadata"]["video_id"]
            segments, labels = crop_targets(
                annotations[video_id], video["metadata"],
                video["feats"].shape[-1], cfg, device,
            )
            gt_segments.append(segments)
            gt_labels.append(labels)
        gt_cls, gt_offsets = model.label_points(
            points, gt_segments, gt_labels
        )
        pseudo_cls, pseudo_offsets, assignments = model.label_points(
            points,
            [item["segments"] for item in post_nms],
            [item["labels"] for item in post_nms],
            return_assignments=True,
        )

        for batch_index, (gt, gt_label, post) in enumerate(zip(
                gt_segments, gt_labels, post_nms)):
            video_key = batch_start + batch_index
            videos += 1
            total_gt_segments += int(gt.shape[0])
            gt_positive = (
                (gt_cls[batch_index].sum(dim=-1) > 0)
                & valid[batch_index] & enabled[batch_index]
            )
            total_gt_anchors += int(gt_positive.sum())
            pseudo_positive = (
                (pseudo_cls[batch_index].sum(dim=-1) > 0)
                & valid[batch_index] & enabled[batch_index]
            )
            pseudo_segments = decode_segments(
                pseudo_offsets[batch_index], centers, strides
            )
            gt_anchor_segments = decode_segments(
                gt_offsets[batch_index], centers, strides
            )
            student_segments = decode_segments(
                student_offsets[batch_index], centers, strides
            )
            overlap = pseudo_positive & gt_positive
            anchor_iou = torch.zeros_like(centers)
            anchor_tnd = torch.zeros_like(centers)
            if overlap.any():
                overlap_tiou, overlap_tnd = aligned_tiou_tnd(
                    pseudo_segments[overlap],
                    gt_anchor_segments[overlap],
                )
                anchor_iou[overlap] = overlap_tiou
                anchor_tnd[overlap] = overlap_tnd
            proposal_iou = pairwise_iou(post["segments"], gt)
            assignment = assignments[batch_index]

            for proposal_index in range(post["segments"].shape[0]):
                if gt.numel():
                    max_iou, matched_index = proposal_iou[
                        proposal_index
                    ].max(dim=0)
                    max_iou = float(max_iou)
                    matched_index = int(matched_index)
                    proposal_tiou, proposal_tnd = aligned_tiou_tnd(
                        post["segments"][proposal_index:proposal_index + 1],
                        gt[matched_index:matched_index + 1],
                    )
                    actual_tiou = float(proposal_tiou)
                    actual_tnd = float(proposal_tnd)
                    class_correct = (
                        max_iou >= 0.5
                        and int(post["labels"][proposal_index])
                        == int(gt_label[matched_index])
                    )
                    matched_gt_keys = [
                        (video_key, gt_index)
                        for gt_index in torch.where(
                            proposal_iou[proposal_index] >= 0.5
                        )[0].tolist()
                    ]
                else:
                    max_iou = 0.0
                    actual_tiou = 0.0
                    actual_tnd = 1.0
                    class_correct = False
                    matched_gt_keys = []

                level = int(post["levels"][proposal_index])
                temporal_index = int(post["indices"][proposal_index])
                predictions = {}
                for head_seed, maps in quality_maps.items():
                    values = maps[level][
                        batch_index, :, temporal_index
                    ]
                    predictions[head_seed] = {
                        "tiou": float(values[0]),
                        "tnd": float(values[1]),
                    }

                assigned = (
                    pseudo_positive & (assignment == proposal_index)
                )
                assigned_foreground = assigned & gt_positive
                delta_by_anchor = torch.full(
                    pseudo_positive.shape, float("nan"), device=device
                )
                l1_by_anchor = torch.full(
                    pseudo_positive.shape, float("nan"), device=device
                )
                delta_values, l1_values = [], []
                if assigned_foreground.any():
                    teacher_iou = aligned_iou(
                        pseudo_segments[assigned_foreground],
                        gt_anchor_segments[assigned_foreground],
                    )
                    current_iou = aligned_iou(
                        student_segments[assigned_foreground],
                        gt_anchor_segments[assigned_foreground],
                    )
                    delta = teacher_iou - current_iou
                    l1 = (
                        pseudo_segments[assigned_foreground]
                        - student_segments[assigned_foreground]
                    ).abs().mean(dim=-1)
                    delta_by_anchor[assigned_foreground] = delta
                    l1_by_anchor[assigned_foreground] = l1
                    delta_values = delta.cpu().tolist()
                    l1_values = l1.cpu().tolist()

                proposal_records.append({
                    "score": float(post["scores"][proposal_index]),
                    "level": level,
                    "quality": predictions,
                    "actual_tiou": actual_tiou,
                    "actual_tnd": actual_tnd,
                    "proposal_max_gt_iou": max_iou,
                    "proposal_foreground": max_iou >= 0.5,
                    "proposal_class_correct": class_correct,
                    "matched_gt_keys": matched_gt_keys,
                    "assigned_anchors": int(assigned.sum()),
                    "foreground_anchors": int(assigned_foreground.sum()),
                    "good_boundary_anchors": int(
                        (assigned & (anchor_iou >= 0.5)).sum()
                    ),
                    "delta_iou": delta_values,
                    "offset_l1": l1_values,
                })
                for anchor_index in torch.where(assigned)[0].tolist():
                    is_foreground = bool(gt_positive[anchor_index])
                    anchor_records.append({
                        "score": float(post["scores"][proposal_index]),
                        "quality": predictions,
                        "anchor_level": int(anchor_levels[anchor_index]),
                        "foreground": is_foreground,
                        "good_boundary": bool(
                            anchor_iou[anchor_index] >= 0.5
                        ),
                        "actual_tiou": (
                            float(anchor_iou[anchor_index])
                            if is_foreground else None
                        ),
                        "actual_tnd": (
                            float(anchor_tnd[anchor_index])
                            if is_foreground else None
                        ),
                        "delta_iou": (
                            float(delta_by_anchor[anchor_index])
                            if is_foreground else None
                        ),
                        "offset_l1": (
                            float(l1_by_anchor[anchor_index])
                            if is_foreground else None
                        ),
                    })

    return (
        proposal_records,
        anchor_records,
        {
            "videos": videos,
            "gt_segments": total_gt_segments,
            "gt_foreground_anchors_fpn_0_2": total_gt_anchors,
        },
    )


def quality_predictions(records, head_seeds):
    result = {}
    confidence = torch.tensor(
        [record["score"] for record in records], dtype=torch.float64
    )
    result["confidence"] = confidence
    ensemble_tiou, ensemble_tnd = [], []
    for head_seed in head_seeds:
        key = str(head_seed)
        tiou = torch.tensor([
            record["quality"][key]["tiou"] for record in records
        ], dtype=torch.float64)
        tnd = torch.tensor([
            record["quality"][key]["tnd"] for record in records
        ], dtype=torch.float64)
        result["apl_seed_{}".format(head_seed)] = (
            confidence * (tiou - tnd).clamp_min(1e-6)
        )
        ensemble_tiou.append(tiou)
        ensemble_tnd.append(tnd)
    mean_tiou = torch.stack(ensemble_tiou).mean(dim=0)
    mean_tnd = torch.stack(ensemble_tnd).mean(dim=0)
    result["apl_ensemble"] = (
        confidence * (mean_tiou - mean_tnd).clamp_min(1e-6)
    )
    return result


def summarize_target_quality(records, head_seeds):
    if not records:
        return {"records": 0}
    actual_tiou = [record["actual_tiou"] for record in records]
    actual_tnd = [record["actual_tnd"] for record in records]
    actual_diou = [
        tiou - tnd for tiou, tnd in zip(actual_tiou, actual_tnd)
    ]
    output = {"records": len(records)}
    all_tiou, all_tnd = [], []
    for head_seed in head_seeds:
        key = str(head_seed)
        predicted_tiou = [
            record["quality"][key]["tiou"] for record in records
        ]
        predicted_tnd = [
            record["quality"][key]["tnd"] for record in records
        ]
        predicted_diou = [
            tiou - tnd
            for tiou, tnd in zip(predicted_tiou, predicted_tnd)
        ]
        output["seed_{}".format(head_seed)] = {
            "tiou_mae": float(np.mean(np.abs(
                np.asarray(predicted_tiou) - np.asarray(actual_tiou)
            ))),
            "tnd_mae": float(np.mean(np.abs(
                np.asarray(predicted_tnd) - np.asarray(actual_tnd)
            ))),
            "diou_pearson": pearson(predicted_diou, actual_diou),
            "diou_spearman": pearson(
                rank_values(predicted_diou), rank_values(actual_diou)
            ),
        }
        all_tiou.append(predicted_tiou)
        all_tnd.append(predicted_tnd)
    ensemble_tiou = np.asarray(all_tiou).mean(axis=0)
    ensemble_tnd = np.asarray(all_tnd).mean(axis=0)
    ensemble_diou = ensemble_tiou - ensemble_tnd
    output["ensemble"] = {
        "tiou_mae": float(np.mean(np.abs(
            ensemble_tiou - np.asarray(actual_tiou)
        ))),
        "tnd_mae": float(np.mean(np.abs(
            ensemble_tnd - np.asarray(actual_tnd)
        ))),
        "diou_pearson": pearson(
            ensemble_diou.tolist(), actual_diou
        ),
        "diou_spearman": pearson(
            rank_values(ensemble_diou.tolist()), rank_values(actual_diou)
        ),
    }
    return output


def summarize_quality_groups(proposal_records, anchor_records, head_seeds):
    """Separate foreground detection from within-foreground quality ranking."""
    groups = {
        "all_proposals": proposal_records,
        "proposal_gt_iou_ge_0.1": [
            record for record in proposal_records
            if record["proposal_max_gt_iou"] >= 0.1
        ],
        "proposal_gt_iou_ge_0.5": [
            record for record in proposal_records
            if record["proposal_max_gt_iou"] >= 0.5
        ],
        "gt_positive_assigned_anchors": [
            record for record in anchor_records
            if record["foreground"]
        ],
    }
    return {
        name: summarize_target_quality(records, head_seeds)
        for name, records in groups.items()
    }


def encode_dense_batch(model, batch):
    """Return the frozen dense predictions needed before alpha and NMS."""
    inputs, masks = model.preprocessing(batch, mode="train")
    backbone_features, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(
        backbone_features, backbone_masks
    )
    cls_logits, _ = model.cls_head_target(fpn_features, fpn_masks)
    hidden, offsets = regression_hidden_and_offsets(
        model, fpn_features, fpn_masks
    )
    points = model.point_generator(fpn_features)
    return hidden, offsets, cls_logits, fpn_masks, points


def top_k_indices(values, usable, count):
    candidates = torch.where(usable)[0]
    if count == 0:
        return candidates[:0]
    if count > candidates.numel():
        raise RuntimeError("early-quality budget exceeds usable candidates")
    order = torch.argsort(
        values[candidates], descending=True, stable=True
    )
    return candidates[order[:count]]


def generate_early_quality_targets(
        model, batch, cls_logits, offsets, fpn_masks, points,
        quality_maps, head_seeds):
    """Use ALQA before alpha/NMS at the original per-level dense budget."""
    method_specs = {
        "confidence": ("confidence", "confidence"),
    }
    for head_seed in head_seeds:
        method_specs["alqa_seed_{}".format(head_seed)] = (
            "alqa_seed_{}".format(head_seed),
            "alqa_seed_{}".format(head_seed),
        )
    method_specs["alqa_ensemble_select_confidence_nms"] = (
        "alqa_ensemble", "confidence"
    )
    method_specs["alqa_ensemble"] = (
        "alqa_ensemble", "alqa_ensemble"
    )
    pre_nms = {name: [] for name in method_specs}
    post_nms = {name: [] for name in method_specs}

    for batch_index, video in enumerate(batch):
        selected = {
            name: {
                "segments": [], "labels": [], "scores": [],
                "classification_scores": [], "levels": [], "indices": [],
            }
            for name in method_specs
        }
        for level in model.stage2_levels:
            probabilities = cls_logits[level][
                batch_index
            ].transpose(0, 1).sigmoid()
            confidence, labels = probabilities.max(dim=-1)
            point_grid = points[level]
            level_offsets = offsets[level][
                batch_index
            ].transpose(0, 1)
            segments = torch.stack((
                point_grid[:, 0] - level_offsets[:, 0] * point_grid[:, 3],
                point_grid[:, 0] + level_offsets[:, 1] * point_grid[:, 3],
            ), dim=-1)
            valid_length = float(video["feats"].shape[-1])
            segments[:, 0].clamp_(min=0.0, max=valid_length)
            segments[:, 1].clamp_(min=0.0, max=valid_length)
            duration = segments[:, 1] - segments[:, 0]
            usable = (
                fpn_masks[level][batch_index, 0]
                & torch.isfinite(segments).all(dim=-1)
                & (duration > model.test_duration_thresh)
            )
            original_budget = int(
                (usable & (confidence >= model.stage2_alpha)).sum()
            )
            ranking = {"confidence": confidence}
            predicted_tiou, predicted_tnd = [], []
            for head_seed in head_seeds:
                values = quality_maps[str(head_seed)][level][batch_index]
                tiou, tnd = values[0], values[1]
                ranking["alqa_seed_{}".format(head_seed)] = (
                    confidence * (tiou - tnd).clamp_min(1e-6)
                )
                predicted_tiou.append(tiou)
                predicted_tnd.append(tnd)
            ensemble_tiou = torch.stack(predicted_tiou).mean(dim=0)
            ensemble_tnd = torch.stack(predicted_tnd).mean(dim=0)
            ranking["alqa_ensemble"] = (
                confidence
                * (ensemble_tiou - ensemble_tnd).clamp_min(1e-6)
            )

            for name, (selection_key, nms_key) in method_specs.items():
                indices = top_k_indices(
                    ranking[selection_key], usable, original_budget
                )
                target = selected[name]
                target["segments"].append(segments[indices])
                target["labels"].append(labels[indices])
                target["scores"].append(ranking[nms_key][indices])
                target["classification_scores"].append(confidence[indices])
                target["levels"].append(torch.full_like(indices, level))
                target["indices"].append(indices)

        for name in method_specs:
            parts = selected[name]
            if not parts["segments"] or not any(
                    item.numel() for item in parts["segments"]):
                empty = {
                    "segments": torch.empty((0, 2), device=model.device),
                    "labels": torch.empty(
                        (0,), dtype=torch.long, device=model.device
                    ),
                    "scores": torch.empty((0,), device=model.device),
                    "classification_scores": torch.empty(
                        (0,), device=model.device
                    ),
                    "levels": torch.empty(
                        (0,), dtype=torch.long, device=model.device
                    ),
                    "indices": torch.empty(
                        (0,), dtype=torch.long, device=model.device
                    ),
                }
                pre_nms[name].append(empty)
                post_nms[name].append(empty)
                continue
            dense_target = {
                key: torch.cat(value)
                for key, value in parts.items()
            }
            pre_nms[name].append(dense_target)
            cpu_segments = dense_target["segments"].detach().cpu()
            cpu_scores = dense_target["scores"].detach().cpu()
            cpu_labels = dense_target["labels"].detach().cpu()
            nms_segments, nms_scores, nms_labels = batched_nms(
                cpu_segments, cpu_scores, cpu_labels,
                model.test_iou_threshold, 0.0, model.test_max_seg_num,
                use_soft_nms=False, multiclass=False,
                sigma=model.test_nms_sigma, voting_thresh=0.0,
            )
            source_indices = match_post_nms_to_pre_nms(
                nms_segments, nms_scores, nms_labels,
                cpu_segments, cpu_scores, cpu_labels,
            )
            post_nms[name].append({
                "segments": nms_segments.to(model.device),
                "labels": nms_labels.to(model.device),
                "scores": nms_scores.to(model.device),
                "classification_scores": dense_target[
                    "classification_scores"
                ][source_indices.to(model.device)],
                "levels": dense_target["levels"][
                    source_indices.to(model.device)
                ],
                "indices": dense_target["indices"][
                    source_indices.to(model.device)
                ],
            })
    return pre_nms, post_nms


def assert_original_baseline(expected, actual):
    """Prove that the confidence arm still reproduces current Stage 2."""
    if len(expected) != len(actual):
        raise RuntimeError("early-quality baseline batch size drifted")
    for expected_item, actual_item in zip(expected, actual):
        if (
            expected_item["segments"].shape != actual_item["segments"].shape
            or not torch.equal(expected_item["labels"], actual_item["labels"])
            or not torch.allclose(
                expected_item["segments"], actual_item["segments"], atol=1e-6
            )
            or not torch.allclose(
                expected_item["scores"], actual_item["scores"], atol=1e-6
            )
        ):
            raise RuntimeError(
                "pre-alpha confidence arm no longer matches Stage 2 baseline"
            )


def early_proposal_record(
        segment, label, score, classification_score, level, temporal_index,
        gt, gt_label, video_key):
    if gt.numel():
        overlap = pairwise_iou(segment[None], gt)[0]
        max_iou, matched_index = overlap.max(dim=0)
        max_iou = float(max_iou)
        matched_index = int(matched_index)
        class_correct = (
            max_iou >= 0.5 and int(label) == int(gt_label[matched_index])
        )
        matched_gt = [
            (video_key, index)
            for index in torch.where(overlap >= 0.5)[0].tolist()
        ]
    else:
        max_iou = 0.0
        class_correct = False
        matched_gt = []
    return {
        "score": float(score),
        "classification_score": float(classification_score),
        "level": int(level),
        "temporal_index": int(temporal_index),
        "video_key": int(video_key),
        "max_iou": max_iou,
        "foreground": max_iou >= 0.5,
        "class_correct": class_correct,
        "matched_gt": matched_gt,
    }


def summarize_early_proposals(records, total_gt_segments):
    matched_gt = {
        key for record in records for key in record["matched_gt"]
    }
    level_counts = {}
    for record in records:
        key = str(record["level"])
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        "proposals": len(records),
        "fpn_level_counts": level_counts,
        "foreground_precision_iou50": ratio(
            sum(record["foreground"] for record in records), len(records)
        ),
        "class_correct_precision": ratio(
            sum(record["class_correct"] for record in records), len(records)
        ),
        "mean_max_gt_iou": ratio(
            sum(record["max_iou"] for record in records), len(records)
        ),
        "iou70_rate": ratio(
            sum(record["max_iou"] >= 0.7 for record in records), len(records)
        ),
        "gt_segment_recall_iou50": ratio(
            len(matched_gt), total_gt_segments
        ),
    }


def summarize_early_anchors(records, total_gt_anchors):
    level_counts = {}
    for record in records:
        key = str(record["anchor_level"])
        level_counts[key] = level_counts.get(key, 0) + 1
    foreground = sum(record["foreground"] for record in records)
    good = sum(record["good_boundary"] for record in records)
    return {
        "pseudo_positive_anchors": len(records),
        "student_anchor_level_counts": level_counts,
        "foreground_precision": ratio(foreground, len(records)),
        "good_boundary_precision_iou50": ratio(good, len(records)),
        "gt_foreground_anchor_coverage": ratio(
            foreground, total_gt_anchors
        ),
        "good_boundary_anchor_coverage": ratio(good, total_gt_anchors),
    }


def top_count_records(records, count):
    return sorted(records, key=lambda record: record["score"], reverse=True)[
        :count
    ]


def collect_early_quality_records(
        model, quality_heads, dataset, annotations, cfg, device,
        levels, batch_size, seed, head_seeds):
    reset_audit_rng(seed)
    method_names = [
        "confidence",
        *["alqa_seed_{}".format(value) for value in head_seeds],
        "alqa_ensemble_select_confidence_nms",
        "alqa_ensemble",
    ]
    pre_records = {name: [] for name in method_names}
    post_records = {name: [] for name in method_names}
    anchor_records = {name: [] for name in method_names}
    total_gt_segments = 0
    total_gt_anchors = 0
    videos = 0

    for batch_start in range(0, len(dataset), batch_size):
        raw_videos = [
            dataset[index]
            for index in range(
                batch_start, min(batch_start + batch_size, len(dataset))
            )
        ]
        batch = [move_video(video, device) for video in raw_videos]
        with torch.no_grad():
            original_targets = model.generate_stage2_pseudo_targets(batch)
            hidden, offsets, cls_logits, fpn_masks, points = (
                encode_dense_batch(model, batch)
            )
            quality_maps = {
                str(head_seed): [
                    head(features).sigmoid()
                    for features in hidden
                ]
                for head_seed, head in quality_heads.items()
            }
            pre_nms, post_nms = generate_early_quality_targets(
                model, batch, cls_logits, offsets, fpn_masks, points,
                quality_maps, head_seeds,
            )
            assert_original_baseline(
                original_targets, post_nms["confidence"]
            )

        gt_segments, gt_labels = [], []
        for raw_video, video in zip(raw_videos, batch):
            video_id = raw_video["metadata"]["video_id"]
            segments, labels = crop_targets(
                annotations[video_id], video["metadata"],
                video["feats"].shape[-1], cfg, device,
            )
            gt_segments.append(segments)
            gt_labels.append(labels)
            total_gt_segments += int(segments.shape[0])
            videos += 1
        gt_cls, gt_offsets = model.label_points(
            points, gt_segments, gt_labels
        )
        valid = torch.cat([
            mask.squeeze(1) for mask in fpn_masks
        ], dim=1)
        enabled = torch.cat([
            torch.full(
                (len(batch), point.shape[0]), level in levels,
                dtype=torch.bool, device=device,
            )
            for level, point in enumerate(points)
        ], dim=1)
        centers = torch.cat([point[:, 0] for point in points])
        strides = torch.cat([point[:, 3] for point in points])
        anchor_levels = torch.cat([
            torch.full(
                (point.shape[0],), level,
                dtype=torch.long, device=device,
            )
            for level, point in enumerate(points)
        ])
        for batch_index in range(len(batch)):
            total_gt_anchors += int((
                (gt_cls[batch_index].sum(dim=-1) > 0)
                & valid[batch_index] & enabled[batch_index]
            ).sum())

        for name in method_names:
            for batch_index, target in enumerate(pre_nms[name]):
                video_key = batch_start + batch_index
                for proposal_index in range(target["segments"].shape[0]):
                    pre_records[name].append(early_proposal_record(
                        target["segments"][proposal_index],
                        target["labels"][proposal_index],
                        target["scores"][proposal_index],
                        target["classification_scores"][proposal_index],
                        target["levels"][proposal_index],
                        target["indices"][proposal_index],
                        gt_segments[batch_index],
                        gt_labels[batch_index],
                        video_key,
                    ))

            pseudo_cls, pseudo_offsets, assignments = model.label_points(
                points,
                [item["segments"] for item in post_nms[name]],
                [item["labels"] for item in post_nms[name]],
                return_assignments=True,
            )
            for batch_index, target in enumerate(post_nms[name]):
                video_key = batch_start + batch_index
                gt_positive = (
                    (gt_cls[batch_index].sum(dim=-1) > 0)
                    & valid[batch_index] & enabled[batch_index]
                )
                pseudo_positive = (
                    (pseudo_cls[batch_index].sum(dim=-1) > 0)
                    & valid[batch_index] & enabled[batch_index]
                )
                pseudo_segments = decode_segments(
                    pseudo_offsets[batch_index], centers, strides
                )
                gt_anchor_segments = decode_segments(
                    gt_offsets[batch_index], centers, strides
                )
                overlap = pseudo_positive & gt_positive
                good = torch.zeros_like(pseudo_positive)
                if overlap.any():
                    good[overlap] = aligned_iou(
                        pseudo_segments[overlap],
                        gt_anchor_segments[overlap],
                    ) >= 0.5
                assignment = assignments[batch_index]
                for proposal_index in range(target["segments"].shape[0]):
                    record = early_proposal_record(
                        target["segments"][proposal_index],
                        target["labels"][proposal_index],
                        target["scores"][proposal_index],
                        target["classification_scores"][proposal_index],
                        target["levels"][proposal_index],
                        target["indices"][proposal_index],
                        gt_segments[batch_index],
                        gt_labels[batch_index],
                        video_key,
                    )
                    assigned = (
                        pseudo_positive & (assignment == proposal_index)
                    )
                    record.update({
                        "assigned_anchors": int(assigned.sum()),
                        "foreground_anchors": int(
                            (assigned & gt_positive).sum()
                        ),
                        "good_boundary_anchors": int(
                            (assigned & good).sum()
                        ),
                    })
                    post_records[name].append(record)
                    for anchor_index in torch.where(assigned)[0].tolist():
                        anchor_records[name].append({
                            "score": float(target["scores"][proposal_index]),
                            "anchor_level": int(
                                anchor_levels[anchor_index]
                            ),
                            "foreground": bool(gt_positive[anchor_index]),
                            "good_boundary": bool(good[anchor_index]),
                        })

    pre_count = {name: len(value) for name, value in pre_records.items()}
    if len(set(pre_count.values())) != 1:
        raise RuntimeError("pre-alpha per-level matched budget drifted")
    baseline_keys = {
        (
            record["video_key"],
            record["level"],
            record["temporal_index"],
        )
        for record in pre_records["confidence"]
    }
    selection_change = {}
    for name, records in pre_records.items():
        keys = {
            (
                record["video_key"],
                record["level"],
                record["temporal_index"],
            )
            for record in records
        }
        added = keys - baseline_keys
        selection_change[name] = {
            "selected": len(keys),
            "shared_with_confidence": len(keys & baseline_keys),
            "replaced_vs_confidence": len(added),
            "replacement_rate": ratio(len(added), len(keys)),
            "rescued_below_alpha": sum(
                (
                    record["video_key"],
                    record["level"],
                    record["temporal_index"],
                ) in added
                and record["classification_score"] < model.stage2_alpha
                for record in records
            ),
        }
    common_proposal_count = min(
        len(value) for value in post_records.values()
    )
    common_anchor_count = min(
        len(value) for value in anchor_records.values()
    )
    return {
        "dataset": {
            "videos": videos,
            "gt_segments": total_gt_segments,
            "gt_foreground_anchors_fpn_0_2": total_gt_anchors,
        },
        "pre_nms_same_dense_budget": {
            name: summarize_early_proposals(
                records, total_gt_segments
            )
            for name, records in pre_records.items()
        },
        "pre_nms_selection_change": selection_change,
        "post_nms_natural_budget": {
            name: {
                "proposal": summarize_early_proposals(
                    post_records[name], total_gt_segments
                ),
                "anchor": summarize_early_anchors(
                    anchor_records[name], total_gt_anchors
                ),
            }
            for name in method_names
        },
        "post_nms_common_proposal_budget": {
            "count": common_proposal_count,
            "methods": {
                name: summarize_early_proposals(
                    top_count_records(
                        post_records[name], common_proposal_count
                    ),
                    total_gt_segments,
                )
                for name in method_names
            },
        },
        "post_assignment_common_anchor_budget": {
            "count": common_anchor_count,
            "methods": {
                name: summarize_early_anchors(
                    top_count_records(
                        anchor_records[name], common_anchor_count
                    ),
                    total_gt_anchors,
                )
                for name in method_names
            },
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--head-seeds", default="123,456,789")
    parser.add_argument("--retain-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--source-seed", type=int, default=20260730)
    parser.add_argument("--target-seed", type=int, default=20260729)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--near-same-l1", type=float, default=1e-2)
    args = parser.parse_args()

    levels = parse_numbers(args.levels, int)
    head_seeds = parse_numbers(args.head_seeds, int)
    fractions = parse_numbers(args.retain_fractions, float)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {
        "enabled": True, "start_epoch": 5,
        "weight": 1.0, "levels": levels,
    }
    _update_config(cfg)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(
        cfg["model_name"], cfg, device=args.device
    ).to(args.device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    student_model = make_meta_arch(
        cfg["model_name"], cfg, device=args.device
    ).to(args.device)
    student_model.load_state_dict(checkpoint["state_dict"])
    student_model.eval()

    source_train = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    target_train = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    batch_size = args.batch_size or int(
        cfg.get("loader", {}).get("batch_size", 1)
    )

    train_sequences, train_positives = collect_source_sequences(
        model, source_train, args.device, levels, batch_size,
        args.source_seed, "train",
    )
    val_sequences, val_positives = collect_source_sequences(
        model, source_val, args.device, levels, batch_size,
        args.source_seed, "validation",
    )
    feature_dim = train_sequences[0]["features"].shape[1]
    quality_heads, head_training = {}, {}
    for head_seed in head_seeds:
        head, metrics = train_quality_head(
            train_sequences, val_sequences, feature_dim, args.device,
            head_seed, args.max_epochs, args.patience, args.learning_rate,
        )
        head.eval()
        quality_heads[head_seed] = head
        head_training[str(head_seed)] = metrics

    proposal_records, anchor_records, dataset_summary = collect_target_records(
        model, student_model, quality_heads, target_train, annotations, cfg,
        args.device, levels, batch_size, args.target_seed,
    )
    proposal_rankings = quality_predictions(proposal_records, head_seeds)
    anchor_rankings = quality_predictions(anchor_records, head_seeds)
    proposal_summary = matched_budget_summary(
        proposal_records, proposal_rankings, fractions, "level",
        lambda indices: summarize_selection(
            proposal_records, indices,
            dataset_summary["gt_segments"],
            dataset_summary["gt_foreground_anchors_fpn_0_2"],
            args.tie_eps, args.near_same_l1,
        ),
    )
    anchor_summary = matched_budget_summary(
        anchor_records, anchor_rankings, fractions, "anchor_level",
        lambda indices: summarize_anchor_selection(
            anchor_records, indices,
            dataset_summary["gt_foreground_anchors_fpn_0_2"],
            args.tie_eps, args.near_same_l1,
        ),
    )
    early_quality = collect_early_quality_records(
        model, quality_heads, target_train, annotations, cfg,
        args.device, levels, batch_size, args.target_seed, head_seeds,
    )
    output = {
        "audit": "apl_alqa_style_frozen_feature_quality_transfer",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "protocol": {
            "quality_supervision": "source GT only",
            "target_gt_usage": "offline evaluation only",
            "scope": (
                "frozen-feature ALQA transfer probe, not end-to-end APL"
            ),
            "quality_head": (
                "two kernel-3 Conv1d outputs on frozen regression hidden features"
            ),
            "joint_score": (
                "classification confidence * max(pred_tIoU-pred_tND, 1e-6)"
            ),
            "candidate_pipeline": (
                "post-hoc arm: EMA -> alpha -> hard class-agnostic NMS; "
                "early arm: same per-video/per-level dense budget before "
                "the same hard class-agnostic NMS"
            ),
            "source_seed": args.source_seed,
            "target_seed": args.target_seed,
            "head_seeds": head_seeds,
            "retain_fractions": fractions,
        },
        "source": {
            "train_videos": len(source_train),
            "validation_videos": len(source_val),
            "train_positive_anchors": train_positives,
            "validation_positive_anchors": val_positives,
            "head_training": head_training,
        },
        "target": {
            **dataset_summary,
            "proposals": len(proposal_records),
            "pseudo_positive_anchors": len(anchor_records),
            "quality_prediction": summarize_target_quality(
                proposal_records, head_seeds
            ),
            "quality_prediction_groups": summarize_quality_groups(
                proposal_records, anchor_records, head_seeds
            ),
            "proposal_matched_budget": proposal_summary,
            "anchor_exact_budget": anchor_summary,
            "early_quality": early_quality,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
