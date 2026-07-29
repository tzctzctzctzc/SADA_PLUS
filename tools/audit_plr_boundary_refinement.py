"""PLR-style source-trained pseudo-boundary refinement target-GT audit."""

import argparse
import copy
import json
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
    apply_boundary_offsets,
    clip_segments,
    temporal_roi_interpolate,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_apl_boundary_quality import parse_numbers
from tools.audit_npl_boundary_reliability import reset_audit_rng
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_stage2_causal import (
    aligned_iou,
    decode_segments,
    pairwise_iou,
    ratio,
)


class PLSRRefiner(nn.Module):
    """Small 1D RoI classifier/regressor following PLR's PLSR block."""

    def __init__(self, feature_dim, hidden_dim, num_classes):
        super().__init__()
        middle_dim = max(64, hidden_dim // 2)
        self.encoder = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, middle_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(middle_dim, middle_dim),
            nn.ReLU(inplace=True),
            nn.Linear(middle_dim, num_classes),
        )
        self.regressor = nn.Sequential(
            nn.Linear(middle_dim, middle_dim),
            nn.ReLU(inplace=True),
            nn.Linear(middle_dim, 2),
            nn.Tanh(),
        )

    def forward(self, roi_features):
        encoded = self.encoder(roi_features).mean(dim=-1)
        return self.classifier(encoded), self.regressor(encoded)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    probability = logits.sigmoid()
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    probability_target = probability * targets + (1 - probability) * (
        1 - targets
    )
    alpha_target = alpha * targets + (1 - alpha) * (1 - targets)
    return (
        alpha_target * (1 - probability_target).pow(gamma) * cross_entropy
    ).mean()


def encode_latent_batch(model, batch):
    inputs, masks = model.preprocessing(batch, mode="train")
    backbone_features, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(
        backbone_features, backbone_masks
    )
    points = model.point_generator(fpn_features)
    return fpn_features, fpn_masks, points


def proposal_rois(
        fpn_features, fpn_masks, point_centers, proposals, roi_size):
    rois = []
    for batch_index, segments in enumerate(proposals):
        valid_length = int(fpn_masks[batch_index, 0].sum())
        for segment in segments:
            rois.append(temporal_roi_interpolate(
                fpn_features[batch_index],
                segment,
                point_centers,
                valid_length,
                roi_size,
            ))
    if not rois:
        return fpn_features.new_empty(
            (0, fpn_features.shape[1], roi_size)
        )
    return torch.stack(rois)


def collect_source_samples(
        model, dataset, device, batch_size, seed, roi_size, foreground_iou):
    reset_audit_rng(seed)
    all_rois, all_segments, all_gt = [], [], []
    all_classes, all_foreground, all_iou = [], [], []
    videos = 0
    proposals = 0
    num_classes = model.num_class_orig

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_batch]
            pseudo_targets = model.generate_stage2_pseudo_targets(batch)
            fpn_features, fpn_masks, points = encode_latent_batch(model, batch)
            rois = proposal_rois(
                fpn_features[0], fpn_masks[0], points[0][:, 0],
                [target["segments"] for target in pseudo_targets],
                roi_size,
            )
            roi_offset = 0
            for video, target in zip(batch, pseudo_targets):
                count = int(target["segments"].shape[0])
                videos += 1
                proposals += count
                if count == 0:
                    continue
                gt_segments = video["segments"]
                gt_labels = video["labels"]
                overlap = pairwise_iou(target["segments"], gt_segments)
                if gt_segments.numel():
                    max_iou, matches = overlap.max(dim=1)
                    matched_gt = gt_segments[matches]
                    matched_labels = gt_labels[matches]
                else:
                    max_iou = target["scores"].new_zeros(count)
                    matched_gt = target["segments"].clone()
                    matched_labels = torch.zeros(
                        count, dtype=torch.long, device=device
                    )
                foreground = max_iou >= foreground_iou
                class_targets = target["segments"].new_zeros(
                    (count, num_classes)
                )
                if foreground.any():
                    class_targets[
                        foreground, matched_labels[foreground].long()
                    ] = 1.0
                all_rois.append(rois[roi_offset:roi_offset + count].cpu())
                all_segments.append(target["segments"].cpu())
                all_gt.append(matched_gt.cpu())
                all_classes.append(class_targets.cpu())
                all_foreground.append(foreground.cpu())
                all_iou.append(max_iou.cpu())
                roi_offset += count

    if not all_rois:
        raise RuntimeError("source split produced no Stage 2 proposals")
    return {
        "rois": torch.cat(all_rois),
        "segments": torch.cat(all_segments),
        "gt_segments": torch.cat(all_gt),
        "class_targets": torch.cat(all_classes),
        "foreground": torch.cat(all_foreground),
        "baseline_iou": torch.cat(all_iou),
        "videos": videos,
        "proposals": proposals,
    }


def evaluate_refiner(
        refiner, samples, device, batch_size, max_delta_scale,
        class_weight, regression_weight):
    refiner.eval()
    total_loss = 0.0
    total_count = 0
    baseline_iou, refined_iou = [], []
    with torch.no_grad():
        for start in range(0, samples["rois"].shape[0], batch_size):
            stop = min(start + batch_size, samples["rois"].shape[0])
            rois = samples["rois"][start:stop].to(device)
            segments = samples["segments"][start:stop].to(device)
            gt_segments = samples["gt_segments"][start:stop].to(device)
            targets = samples["class_targets"][start:stop].to(device)
            foreground = samples["foreground"][start:stop].to(device)
            logits, offsets = refiner(rois)
            refined = apply_boundary_offsets(
                segments, offsets, max_delta_scale
            )
            class_loss = sigmoid_focal_loss(logits, targets)
            if foreground.any():
                tiou, _ = aligned_tiou_tnd(
                    refined[foreground], gt_segments[foreground]
                )
                regression_loss = 1.0 - tiou.mean()
                baseline_iou.extend(
                    samples["baseline_iou"][start:stop][foreground.cpu()]
                    .tolist()
                )
                refined_iou.extend(tiou.cpu().tolist())
            else:
                regression_loss = 0.0 * refined.sum()
            loss = (
                class_weight * class_loss
                + regression_weight * regression_loss
            )
            total_loss += float(loss) * (stop - start)
            total_count += stop - start

    delta = np.asarray(refined_iou) - np.asarray(baseline_iou)
    return {
        "loss": ratio(total_loss, total_count),
        "foreground_proposals": len(refined_iou),
        "baseline_mean_iou": (
            float(np.mean(baseline_iou)) if baseline_iou else None
        ),
        "refined_mean_iou": (
            float(np.mean(refined_iou)) if refined_iou else None
        ),
        "mean_delta_iou": float(delta.mean()) if delta.size else None,
        "win_rate": float((delta > 1e-4).mean()) if delta.size else None,
        "loss_rate": float((delta < -1e-4).mean()) if delta.size else None,
    }


def train_refiner(
        train_samples, val_samples, feature_dim, hidden_dim, num_classes,
        device, seed, max_epochs, patience, learning_rate, train_batch_size,
        max_delta_scale, class_weight, regression_weight):
    reset_audit_rng(seed)
    refiner = PLSRRefiner(
        feature_dim, hidden_dim, num_classes
    ).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    best_state = copy.deepcopy(refiner.state_dict())
    best_epoch = 0
    best_val = float("inf")
    remaining_patience = patience
    history = []
    sample_count = train_samples["rois"].shape[0]

    for epoch in range(1, max_epochs + 1):
        refiner.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(sample_count, generator=generator)
        loss_sum = 0.0
        seen = 0
        for start in range(0, sample_count, train_batch_size):
            indices = order[start:start + train_batch_size]
            rois = train_samples["rois"][indices].to(device)
            segments = train_samples["segments"][indices].to(device)
            gt_segments = train_samples["gt_segments"][indices].to(device)
            class_targets = train_samples["class_targets"][indices].to(device)
            foreground = train_samples["foreground"][indices].to(device)
            logits, offsets = refiner(rois)
            refined = apply_boundary_offsets(
                segments, offsets, max_delta_scale
            )
            class_loss = sigmoid_focal_loss(logits, class_targets)
            if foreground.any():
                tiou, _ = aligned_tiou_tnd(
                    refined[foreground], gt_segments[foreground]
                )
                regression_loss = 1.0 - tiou.mean()
            else:
                regression_loss = 0.0 * refined.sum()
            loss = (
                class_weight * class_loss
                + regression_weight * regression_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss) * indices.numel()
            seen += indices.numel()

        validation = evaluate_refiner(
            refiner, val_samples, device, train_batch_size,
            max_delta_scale, class_weight, regression_weight,
        )
        history.append({
            "epoch": epoch,
            "train_loss": ratio(loss_sum, seen),
            "validation_loss": validation["loss"],
            "validation_mean_delta_iou": validation["mean_delta_iou"],
        })
        if validation["loss"] < best_val - 1e-6:
            best_val = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(refiner.state_dict())
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    refiner.load_state_dict(best_state)
    return refiner, {
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": evaluate_refiner(
            refiner, val_samples, device, train_batch_size,
            max_delta_scale, class_weight, regression_weight,
        ),
        "history": history,
    }


def infer_refinement(
        refiner, rois, segments, labels, valid_lengths, max_delta_scale):
    logits, offsets = refiner(rois)
    refined = apply_boundary_offsets(segments, offsets, max_delta_scale)
    refined = clip_segments(refined, valid_lengths, fallback=segments)
    confidence = logits.sigmoid().gather(
        1, labels[:, None]
    ).squeeze(1)
    return refined, confidence


def synthesize_boundaries(
        refiner, direct_segments, feature_maps, masks, centers,
        video_indices, labels, valid_lengths, roi_size, max_delta_scale,
        repeats, boundary_sigma, feature_sigma, random_seed):
    generator = torch.Generator().manual_seed(random_seed)
    count = direct_segments.shape[0]
    duration = direct_segments[:, 1] - direct_segments[:, 0]
    start_noise = torch.randn(
        (count, repeats), generator=generator
    ).to(direct_segments.device) * boundary_sigma
    end_noise = torch.randn(
        (count, repeats), generator=generator
    ).to(direct_segments.device) * boundary_sigma
    expanded = torch.stack((
        direct_segments[:, 0, None]
        + (start_noise - 0.5) * duration[:, None],
        direct_segments[:, 1, None]
        + (end_noise + 0.5) * duration[:, None],
    ), dim=-1)
    expanded = expanded.reshape(-1, 2)
    repeated_lengths = valid_lengths.repeat_interleave(repeats)
    fallback = direct_segments.repeat_interleave(repeats, dim=0)
    expanded = clip_segments(expanded, repeated_lengths, fallback=fallback)

    expanded_by_video = [[] for _ in range(feature_maps.shape[0])]
    flat_order = []
    for proposal_index in range(count):
        video_index = int(video_indices[proposal_index])
        for repeat_index in range(repeats):
            flat_index = proposal_index * repeats + repeat_index
            expanded_by_video[video_index].append(expanded[flat_index])
            flat_order.append((video_index, len(expanded_by_video[video_index]) - 1))
    grouped = [
        (
            torch.stack(items)
            if items else expanded.new_empty((0, 2))
        )
        for items in expanded_by_video
    ]
    grouped_rois = proposal_rois(
        feature_maps, masks, centers, grouped, roi_size
    )
    # proposal_rois groups by video; restore proposal-major ordering.
    group_offsets = []
    running = 0
    for items in grouped:
        group_offsets.append(running)
        running += items.shape[0]
    per_video_seen = [0 for _ in grouped]
    reorder = []
    for video_index, _ in flat_order:
        reorder.append(group_offsets[video_index] + per_video_seen[video_index])
        per_video_seen[video_index] += 1
    rois = grouped_rois[torch.tensor(
        reorder, dtype=torch.long, device=grouped_rois.device
    )]
    noise_generator = torch.Generator().manual_seed(random_seed + 1)
    roi_noise = torch.randn(
        rois.shape, generator=noise_generator,
        dtype=rois.dtype,
    ).to(rois.device)
    rois = rois + feature_sigma * roi_noise
    repeated_labels = labels.repeat_interleave(repeats)
    refined, confidence = infer_refinement(
        refiner, rois, expanded, repeated_labels,
        repeated_lengths, max_delta_scale,
    )
    refined = refined.reshape(count, repeats, 2)
    confidence = confidence.reshape(count, repeats).clamp_min(1e-6)
    return (
        refined * confidence[:, :, None]
    ).sum(dim=1) / confidence.sum(dim=1, keepdim=True)


def update_anchor_statistics(
        statistics, variant, model, points, segments, labels,
        gt_cls, gt_offsets, valid, enabled, centers, strides, levels):
    pseudo_cls, pseudo_offsets = model.label_points(points, segments, labels)
    for batch_index in range(len(segments)):
        gt_positive = (
            (gt_cls[batch_index].sum(dim=-1) > 0)
            & valid[batch_index] & enabled[batch_index]
        )
        pseudo_positive = (
            (pseudo_cls[batch_index].sum(dim=-1) > 0)
            & valid[batch_index] & enabled[batch_index]
        )
        overlap = pseudo_positive & gt_positive
        good = torch.zeros_like(pseudo_positive)
        if overlap.any():
            pseudo_decoded = decode_segments(
                pseudo_offsets[batch_index], centers, strides
            )
            gt_decoded = decode_segments(
                gt_offsets[batch_index], centers, strides
            )
            good[overlap] = aligned_iou(
                pseudo_decoded[overlap], gt_decoded[overlap]
            ) >= 0.5
        target = statistics[variant]
        target["positive"] += int(pseudo_positive.sum())
        target["foreground"] += int((pseudo_positive & gt_positive).sum())
        target["good"] += int(good.sum())
        for level in torch.unique(levels[pseudo_positive]).tolist():
            level_key = str(int(level))
            level_mask = pseudo_positive & (levels == level)
            level_stats = target["per_level"].setdefault(
                level_key, {"positive": 0, "foreground": 0, "good": 0}
            )
            level_stats["positive"] += int(level_mask.sum())
            level_stats["foreground"] += int(
                (level_mask & gt_positive).sum()
            )
            level_stats["good"] += int((level_mask & good).sum())


def proposal_variant_record(
        segment, label, gt_segments, gt_labels, video_key):
    if gt_segments.numel():
        overlap = pairwise_iou(segment[None], gt_segments)[0]
        max_iou, match = overlap.max(dim=0)
        max_iou = float(max_iou)
        match = int(match)
        matched_gt = [
            (video_key, index)
            for index in torch.where(overlap >= 0.5)[0].tolist()
        ]
        class_correct = (
            max_iou >= 0.5 and int(label) == int(gt_labels[match])
        )
    else:
        max_iou = 0.0
        matched_gt = []
        class_correct = False
    return {
        "max_iou": max_iou,
        "matched_gt": matched_gt,
        "class_correct": class_correct,
        "segment": segment.detach().cpu().tolist(),
    }


def collect_target_results(
        model, refiners, dataset, annotations, cfg, device, levels,
        batch_size, target_seed, roi_size, max_delta_scale,
        repeats, boundary_sigma, feature_sigma):
    reset_audit_rng(target_seed)
    variants = ["baseline"]
    for seed in refiners:
        variants.extend([
            "direct_seed_{}".format(seed),
            "synthesis_seed_{}".format(seed),
        ])
    variants.extend(["direct_ensemble", "synthesis_ensemble"])
    proposal_records = []
    anchor_statistics = {
        variant: {
            "positive": 0, "foreground": 0, "good": 0, "per_level": {}
        }
        for variant in variants
    }
    total_gt_segments = 0
    total_gt_anchors = 0
    videos = 0
    proposal_count = 0

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_batch]
            pseudo_targets = model.generate_stage2_pseudo_targets(batch)
            fpn_features, fpn_masks, points = encode_latent_batch(model, batch)
            baseline_segments = [
                target["segments"] for target in pseudo_targets
            ]
            baseline_labels = [target["labels"] for target in pseudo_targets]
            rois = proposal_rois(
                fpn_features[0], fpn_masks[0], points[0][:, 0],
                baseline_segments, roi_size,
            )
            flat_segments = torch.cat(baseline_segments)
            flat_labels = torch.cat(baseline_labels)
            video_indices = torch.cat([
                torch.full(
                    (segments.shape[0],), index,
                    dtype=torch.long, device=device,
                )
                for index, segments in enumerate(baseline_segments)
            ])
            valid_lengths = torch.cat([
                torch.full(
                    (segments.shape[0],), float(batch[index]["feats"].shape[-1]),
                    dtype=segments.dtype, device=device,
                )
                for index, segments in enumerate(baseline_segments)
            ])

            flat_variants = {"baseline": flat_segments}
            direct_outputs, synthesis_outputs = [], []
            for refiner_seed, refiner in refiners.items():
                direct, _ = infer_refinement(
                    refiner, rois, flat_segments, flat_labels,
                    valid_lengths, max_delta_scale,
                )
                synthesis = synthesize_boundaries(
                    refiner, direct,
                    fpn_features[0], fpn_masks[0], points[0][:, 0],
                    video_indices, flat_labels, valid_lengths,
                    roi_size, max_delta_scale, repeats,
                    boundary_sigma, feature_sigma,
                    target_seed + batch_start * 1000,
                )
                flat_variants[
                    "direct_seed_{}".format(refiner_seed)
                ] = direct
                flat_variants[
                    "synthesis_seed_{}".format(refiner_seed)
                ] = synthesis
                direct_outputs.append(direct)
                synthesis_outputs.append(synthesis)
            flat_variants["direct_ensemble"] = torch.stack(
                direct_outputs
            ).mean(dim=0)
            flat_variants["synthesis_ensemble"] = torch.stack(
                synthesis_outputs
            ).mean(dim=0)

            grouped_variants = {}
            for variant, flat in flat_variants.items():
                grouped_variants[variant] = [
                    flat[video_indices == index]
                    for index in range(len(batch))
                ]

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_batch, batch):
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

            for variant in variants:
                update_anchor_statistics(
                    anchor_statistics, variant, model, points,
                    grouped_variants[variant], baseline_labels,
                    gt_cls, gt_offsets, valid, enabled,
                    centers, strides, anchor_levels,
                )

            flat_offset = 0
            for batch_index, baseline in enumerate(baseline_segments):
                video_key = batch_start + batch_index
                for proposal_index in range(baseline.shape[0]):
                    record = {
                        "score": float(
                            pseudo_targets[batch_index]["scores"][
                                proposal_index
                            ]
                        ),
                        "label": int(baseline_labels[batch_index][proposal_index]),
                        "variants": {},
                    }
                    for variant in variants:
                        record["variants"][variant] = proposal_variant_record(
                            grouped_variants[variant][batch_index][
                                proposal_index
                            ],
                            baseline_labels[batch_index][proposal_index],
                            gt_segments[batch_index],
                            gt_labels[batch_index],
                            video_key,
                        )
                    proposal_records.append(record)
                    proposal_count += 1
                flat_offset += baseline.shape[0]

    return {
        "variants": variants,
        "proposal_records": proposal_records,
        "anchor_statistics": anchor_statistics,
        "dataset": {
            "videos": videos,
            "proposals": proposal_count,
            "gt_segments": total_gt_segments,
            "gt_foreground_anchors_fpn_0_2": total_gt_anchors,
        },
    }


def summarize_proposal_variant(records, variant, total_gt_segments):
    current = [record["variants"][variant] for record in records]
    baseline = [record["variants"]["baseline"] for record in records]
    iou = np.asarray([item["max_iou"] for item in current])
    baseline_iou = np.asarray([item["max_iou"] for item in baseline])
    delta = iou - baseline_iou
    boundary_l1 = np.asarray([
        np.abs(
            np.asarray(item["segment"])
            - np.asarray(base["segment"])
        ).mean()
        for item, base in zip(current, baseline)
    ])
    baseline_foreground = baseline_iou >= 0.5
    baseline_background = ~baseline_foreground
    matched_gt = {
        tuple(key)
        for item in current for key in item["matched_gt"]
    }

    def subgroup(mask):
        values = delta[mask]
        return {
            "n": int(mask.sum()),
            "mean_delta_iou": (
                float(values.mean()) if values.size else None
            ),
            "win_rate": (
                float((values > 1e-4).mean()) if values.size else None
            ),
            "loss_rate": (
                float((values < -1e-4).mean()) if values.size else None
            ),
        }

    return {
        "n": len(records),
        "mean_max_gt_iou": float(iou.mean()),
        "iou30_rate": float((iou >= 0.3).mean()),
        "iou50_rate": float((iou >= 0.5).mean()),
        "iou70_rate": float((iou >= 0.7).mean()),
        "class_correct_precision": ratio(
            sum(item["class_correct"] for item in current), len(current)
        ),
        "gt_segment_recall_iou50": ratio(
            len(matched_gt), total_gt_segments
        ),
        "mean_delta_iou_vs_baseline": float(delta.mean()),
        "win_rate_vs_baseline": float((delta > 1e-4).mean()),
        "loss_rate_vs_baseline": float((delta < -1e-4).mean()),
        "tie_rate_vs_baseline": float((np.abs(delta) <= 1e-4).mean()),
        "mean_boundary_l1_vs_baseline": float(boundary_l1.mean()),
        "baseline_foreground_subgroup": subgroup(baseline_foreground),
        "baseline_background_subgroup": subgroup(baseline_background),
    }


def summarize_anchor_variant(
        statistics, total_gt_anchors, proposal_count):
    output = {
        "pseudo_positive_anchors": statistics["positive"],
        "foreground_precision": ratio(
            statistics["foreground"], statistics["positive"]
        ),
        "good_boundary_precision_iou50": ratio(
            statistics["good"], statistics["positive"]
        ),
        "gt_foreground_anchor_coverage": ratio(
            statistics["foreground"], total_gt_anchors
        ),
        "good_boundary_anchor_coverage": ratio(
            statistics["good"], total_gt_anchors
        ),
        "proposal_to_anchor_amplification": ratio(
            statistics["positive"], proposal_count
        ),
        "per_level": {},
    }
    for level, values in statistics["per_level"].items():
        output["per_level"][level] = {
            **values,
            "foreground_precision": ratio(
                values["foreground"], values["positive"]
            ),
            "good_boundary_precision_iou50": ratio(
                values["good"], values["positive"]
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
    parser.add_argument("--refiner-seeds", default="123,456,789")
    parser.add_argument("--source-seed", type=int, default=20260730)
    parser.add_argument("--target-seed", type=int, default=20260729)
    parser.add_argument("--roi-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--foreground-iou", type=float, default=0.5)
    parser.add_argument("--max-delta-scale", type=float, default=0.45)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--regression-weight", type=float, default=1.0)
    parser.add_argument("--synthesis-repeats", type=int, default=4)
    parser.add_argument("--boundary-sigma", type=float, default=0.05)
    parser.add_argument("--feature-sigma", type=float, default=0.01)
    args = parser.parse_args()

    levels = parse_numbers(args.levels, int)
    refiner_seeds = parse_numbers(args.refiner_seeds, int)
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
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
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

    train_samples = collect_source_samples(
        model, source_train, args.device, batch_size,
        args.source_seed, args.roi_size, args.foreground_iou,
    )
    val_samples = collect_source_samples(
        model, source_val, args.device, batch_size,
        args.source_seed, args.roi_size, args.foreground_iou,
    )
    feature_dim = train_samples["rois"].shape[1]
    refiners, training = {}, {}
    for seed in refiner_seeds:
        refiner, metrics = train_refiner(
            train_samples, val_samples, feature_dim, args.hidden_dim,
            model.num_class_orig, args.device, seed,
            args.max_epochs, args.patience, args.learning_rate,
            args.train_batch_size, args.max_delta_scale,
            args.class_weight, args.regression_weight,
        )
        refiner.eval()
        refiners[seed] = refiner
        training[str(seed)] = metrics

    target = collect_target_results(
        model, refiners, target_train, annotations, cfg, args.device,
        levels, batch_size, args.target_seed, args.roi_size,
        args.max_delta_scale, args.synthesis_repeats,
        args.boundary_sigma, args.feature_sigma,
    )
    proposal_summary = {
        variant: summarize_proposal_variant(
            target["proposal_records"], variant,
            target["dataset"]["gt_segments"],
        )
        for variant in target["variants"]
    }
    anchor_summary = {
        variant: summarize_anchor_variant(
            target["anchor_statistics"][variant],
            target["dataset"]["gt_foreground_anchors_fpn_0_2"],
            target["dataset"]["proposals"],
        )
        for variant in target["variants"]
    }
    output = {
        "audit": "plr_style_source_trained_boundary_refinement",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "protocol": {
            "refiner_supervision": "source GT only",
            "target_gt_usage": "offline evaluation only",
            "source_validation": (
                "fixed-seed 2304-window crop for proposal-compatible early stop"
            ),
            "candidate_pipeline": (
                "EMA -> alpha -> hard class-agnostic NMS; proposal set fixed"
            ),
            "plsr": (
                "FPN0 temporal RoI interpolation -> two Conv1d -> "
                "class and normalized start/end offset branches"
            ),
            "source_foreground_iou": args.foreground_iou,
            "roi_size": args.roi_size,
            "max_delta_scale": args.max_delta_scale,
            "refiner_seeds": refiner_seeds,
            "boundary_synthesis": {
                "repeats": args.synthesis_repeats,
                "boundary_sigma": args.boundary_sigma,
                "feature_sigma": args.feature_sigma,
                "fusion": "PLSR class-confidence weighted boundary mean",
            },
            "excluded": (
                "AWL, target training, target classification, re-NMS, "
                "threshold or loss changes"
            ),
        },
        "source": {
            "train_videos": train_samples["videos"],
            "train_proposals": train_samples["proposals"],
            "train_foreground_proposals": int(
                train_samples["foreground"].sum()
            ),
            "validation_videos": val_samples["videos"],
            "validation_proposals": val_samples["proposals"],
            "validation_foreground_proposals": int(
                val_samples["foreground"].sum()
            ),
            "refiner_training": training,
        },
        "target": {
            **target["dataset"],
            "proposal_quality": proposal_summary,
            "anchor_training_chain": anchor_summary,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
