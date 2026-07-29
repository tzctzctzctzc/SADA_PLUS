"""APL-style source-trained localization-quality target-GT audit."""

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
    matched_budget_indices,
)
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
        model, quality_heads, dataset, annotations, cfg, device,
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
            for level_offsets in offsets
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
            if overlap.any():
                anchor_iou[overlap] = aligned_iou(
                    pseudo_segments[overlap],
                    gt_anchor_segments[overlap],
                )
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
    actual_tiou = [record["actual_tiou"] for record in records]
    actual_tnd = [record["actual_tnd"] for record in records]
    actual_diou = [
        tiou - tnd for tiou, tnd in zip(actual_tiou, actual_tnd)
    ]
    output = {}
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
        model, quality_heads, target_train, annotations, cfg, args.device,
        levels, batch_size, args.target_seed,
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
    output = {
        "audit": "apl_style_source_trained_boundary_quality",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "protocol": {
            "quality_supervision": "source GT only",
            "target_gt_usage": "offline evaluation only",
            "quality_head": (
                "two kernel-3 Conv1d outputs on frozen regression hidden features"
            ),
            "joint_score": (
                "classification confidence * max(pred_tIoU-pred_tND, 1e-6)"
            ),
            "candidate_pipeline": (
                "EMA -> alpha -> hard class-agnostic NMS -> FPN 0-2"
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
            "proposal_matched_budget": proposal_summary,
            "anchor_exact_budget": anchor_summary,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
