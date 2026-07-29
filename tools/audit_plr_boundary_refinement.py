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


IOU_BINS = (
    ("iou_01_03", 0.1, 0.3),
    ("iou_03_05", 0.3, 0.5),
    ("iou_05_07", 0.5, 0.7),
    ("iou_07_10", 0.7, 1.000001),
)


def _segment_iou(candidate, references):
    if references.numel() == 0:
        return 0.0
    left = torch.maximum(candidate[0], references[:, 0])
    right = torch.minimum(candidate[1], references[:, 1])
    intersection = (right - left).clamp_min(0)
    union = (
        candidate[1] - candidate[0]
        + references[:, 1] - references[:, 0]
        - intersection
    ).clamp_min(1e-6)
    return float((intersection / union).max())


def _sample_jittered_segment(gt, valid_length, low, high, rng):
    """Generate one deterministic-error-family proposal in an IoU band."""
    start, end = float(gt[0]), float(gt[1])
    duration = max(end - start, 1.0)
    center = 0.5 * (start + end)
    for attempt in range(400):
        error_type = attempt % 4
        if error_type == 0:
            shifted_center = center + rng.uniform(-1.5, 1.5) * duration
            scale = np.exp(rng.uniform(np.log(0.3), np.log(2.5)))
            candidate = [
                shifted_center - 0.5 * scale * duration,
                shifted_center + 0.5 * scale * duration,
            ]
        elif error_type == 1:
            candidate = [
                start + rng.uniform(-1.5, 1.5) * duration,
                end + rng.uniform(-0.25, 0.25) * duration,
            ]
        elif error_type == 2:
            candidate = [
                start + rng.uniform(-0.25, 0.25) * duration,
                end + rng.uniform(-1.5, 1.5) * duration,
            ]
        else:
            candidate = [
                start + rng.uniform(-1.0, 1.0) * duration,
                end + rng.uniform(-1.0, 1.0) * duration,
            ]
        candidate = torch.tensor(
            [max(0.0, candidate[0]), min(float(valid_length), candidate[1])],
            dtype=gt.dtype, device=gt.device,
        )
        if candidate[1] - candidate[0] <= 1e-3:
            continue
        iou = _segment_iou(candidate, gt[None])
        if low <= iou < high:
            return candidate
    return None


def make_balanced_source_candidates(
        gt_segments, valid_length, rng, background_per_gt=2):
    """Build hard foreground, BS-expanded and true-background source proposals."""
    candidates, categories, feature_noise = [], [], []
    for gt in gt_segments:
        for category, low, high in IOU_BINS:
            candidate = _sample_jittered_segment(
                gt, valid_length, low, high, rng
            )
            if candidate is not None:
                candidates.append(candidate)
                categories.append(category)
                feature_noise.append(False)

        duration = max(float(gt[1] - gt[0]), 1.0)
        expanded = torch.tensor(
            [
                float(gt[0]) + (rng.normal(0.0, 0.05) - 0.5) * duration,
                float(gt[1]) + (rng.normal(0.0, 0.05) + 0.5) * duration,
            ],
            dtype=gt.dtype, device=gt.device,
        )
        expanded[0].clamp_(min=0.0, max=float(valid_length))
        expanded[1].clamp_(min=0.0, max=float(valid_length))
        if expanded[1] - expanded[0] > 1e-3:
            candidates.append(expanded)
            categories.append("bs_expanded")
            feature_noise.append(True)

    wanted_background = max(4, background_per_gt * max(len(gt_segments), 1))
    durations = (
        (gt_segments[:, 1] - gt_segments[:, 0]).clamp_min(1.0).cpu().numpy()
        if gt_segments.numel() else np.asarray([max(valid_length * 0.1, 1.0)])
    )
    for _ in range(wanted_background):
        candidate = None
        for _ in range(600):
            duration = float(rng.choice(durations))
            duration *= float(np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
            duration = min(max(duration, 1.0), float(valid_length))
            start = rng.uniform(0.0, max(float(valid_length) - duration, 1e-6))
            trial = torch.tensor(
                [start, start + duration],
                dtype=gt_segments.dtype, device=gt_segments.device,
            )
            if _segment_iou(trial, gt_segments) < 0.1:
                candidate = trial
                break
        if candidate is not None:
            candidates.append(candidate)
            categories.append("background")
            feature_noise.append(False)

    if not candidates:
        return (
            gt_segments.new_empty((0, 2)),
            [],
            torch.empty(0, dtype=torch.bool, device=gt_segments.device),
        )
    return (
        torch.stack(candidates),
        categories,
        torch.tensor(feature_noise, dtype=torch.bool, device=gt_segments.device),
    )


class DeterministicWindowDataset:
    """Lazy fixed windows that retain every GT fragment visible in a crop."""

    def __init__(self, dataset, max_seq_len, stride):
        self.dataset = dataset
        self.windows = []
        for index in range(len(dataset)):
            length = int(dataset[index]["feats"].shape[-1])
            if length <= max_seq_len:
                starts = [0]
            else:
                starts = list(range(0, length - max_seq_len + 1, stride))
                last = length - max_seq_len
                if starts[-1] != last:
                    starts.append(last)
            self.windows.extend(
                (index, start, min(start + max_seq_len, length))
                for start in starts
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        video_index, start, stop = self.windows[index]
        video = self.dataset[video_index]
        length = int(video["feats"].shape[-1])
        window = {
            "feats": video["feats"][:, start:stop].clone(),
            "metadata": copy.deepcopy(video["metadata"]),
        }
        window["metadata"]["original_feat_len"] = length
        window["metadata"]["crop_start"] = start
        window["metadata"]["video_length"] = stop - start
        segments = video["segments"]
        labels = video["labels"]
        if segments is None or segments.numel() == 0:
            window["segments"] = torch.empty((0, 2))
            window["labels"] = torch.empty(0, dtype=torch.long)
        else:
            overlap = (
                torch.minimum(segments[:, 1], torch.tensor(float(stop)))
                - torch.maximum(segments[:, 0], torch.tensor(float(start)))
            ).clamp_min(0)
            keep = overlap > 0
            clipped = segments[keep].clone()
            clipped[:, 0].clamp_(min=float(start), max=float(stop))
            clipped[:, 1].clamp_(min=float(start), max=float(stop))
            window["segments"] = clipped - start
            window["labels"] = labels[keep].clone()
        return window


def collect_source_samples(
        model, dataset, device, batch_size, seed, roi_size,
        include_teacher=True, include_synthetic=True):
    reset_audit_rng(seed)
    all_rois, all_segments, all_gt = [], [], []
    all_classes, all_foreground, all_iou = [], [], []
    all_feature_noise, all_categories = [], []
    videos = 0
    proposals = 0
    num_classes = model.num_class_orig
    rng = np.random.default_rng(seed)

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_batch]
            fpn_features, fpn_masks, points = encode_latent_batch(model, batch)
            if include_teacher:
                pseudo_targets = model.generate_stage2_pseudo_targets(batch)
            else:
                pseudo_targets = [
                    {"segments": video["segments"].new_empty((0, 2))}
                    for video in batch
                ]
            candidate_segments, candidate_categories = [], []
            candidate_feature_noise = []
            for video, target in zip(batch, pseudo_targets):
                segments = [target["segments"]]
                categories = ["teacher"] * int(target["segments"].shape[0])
                noise = torch.zeros(
                    target["segments"].shape[0],
                    dtype=torch.bool, device=device,
                )
                if include_synthetic:
                    synthetic, synthetic_categories, synthetic_noise = (
                        make_balanced_source_candidates(
                            video["segments"],
                            int(video["feats"].shape[-1]),
                            rng,
                        )
                    )
                    segments.append(synthetic)
                    categories.extend(synthetic_categories)
                    noise = torch.cat((noise, synthetic_noise))
                candidate_segments.append(torch.cat(segments))
                candidate_categories.append(categories)
                candidate_feature_noise.append(noise)
            rois = proposal_rois(
                fpn_features[0], fpn_masks[0], points[0][:, 0],
                candidate_segments,
                roi_size,
            )
            roi_offset = 0
            for video, segments, categories, feature_noise in zip(
                    batch, candidate_segments, candidate_categories,
                    candidate_feature_noise):
                count = int(segments.shape[0])
                videos += 1
                proposals += count
                if count == 0:
                    continue
                gt_segments = video["segments"]
                gt_labels = video["labels"]
                overlap = pairwise_iou(segments, gt_segments)
                if gt_segments.numel():
                    max_iou, matches = overlap.max(dim=1)
                    matched_gt = gt_segments[matches]
                    matched_labels = gt_labels[matches]
                else:
                    max_iou = segments.new_zeros(count)
                    matched_gt = segments.clone()
                    matched_labels = torch.zeros(
                        count, dtype=torch.long, device=device
                    )
                foreground = max_iou >= 0.1
                class_targets = segments.new_zeros((count, num_classes))
                if foreground.any():
                    class_targets[
                        foreground, matched_labels[foreground].long()
                    ] = 1.0
                all_rois.append(rois[roi_offset:roi_offset + count].cpu())
                all_segments.append(segments.cpu())
                all_gt.append(matched_gt.cpu())
                all_classes.append(class_targets.cpu())
                all_foreground.append(foreground.cpu())
                all_iou.append(max_iou.cpu())
                all_feature_noise.append(feature_noise.cpu())
                for category, iou in zip(categories, max_iou.tolist()):
                    if category == "teacher":
                        if iou < 0.1:
                            category = "teacher_background"
                        else:
                            category = next(
                                name for name, low, high in IOU_BINS
                                if low <= iou < high
                            )
                            category = "teacher_{}".format(category)
                    elif category.startswith("iou_"):
                        category = next(
                            name for name, low, high in IOU_BINS
                            if low <= iou < high
                        )
                    all_categories.append(category)
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
        "feature_noise": torch.cat(all_feature_noise),
        "categories": all_categories,
        "videos": videos,
        "proposals": proposals,
    }


def evaluate_refiner(
        refiner, samples, device, batch_size, max_delta_scale,
        class_weight, regression_weight):
    refiner.eval()
    total_loss = 0.0
    total_count = 0
    baseline_iou, refined_iou, foreground_categories = [], [], []
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
                foreground_categories.extend([
                    category
                    for category, keep in zip(
                        samples["categories"][start:stop],
                        foreground.cpu().tolist(),
                    )
                    if keep
                ])
            else:
                regression_loss = 0.0 * refined.sum()
            loss = (
                class_weight * class_loss
                + regression_weight * regression_loss
            )
            total_loss += float(loss) * (stop - start)
            total_count += stop - start

    delta = np.asarray(refined_iou) - np.asarray(baseline_iou)
    baseline_array = np.asarray(baseline_iou)
    refined_array = np.asarray(refined_iou)
    category_metrics = {}
    for category in sorted(set(foreground_categories)):
        mask = np.asarray([
            value == category for value in foreground_categories
        ])
        category_delta = delta[mask]
        category_metrics[category] = {
            "n": int(mask.sum()),
            "baseline_mean_iou": float(baseline_array[mask].mean()),
            "refined_mean_iou": float(refined_array[mask].mean()),
            "mean_delta_iou": float(category_delta.mean()),
        }
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
        "baseline_iou50_rate": (
            float((baseline_array >= 0.5).mean()) if delta.size else None
        ),
        "refined_iou50_rate": (
            float((refined_array >= 0.5).mean()) if delta.size else None
        ),
        "iou50_rate_gain": (
            float(
                (refined_array >= 0.5).mean()
                - (baseline_array >= 0.5).mean()
            )
            if delta.size else None
        ),
        "per_category": category_metrics,
    }


def balanced_epoch_order(samples, seed):
    """Sample 50% background and equal counts from foreground categories."""
    generator = torch.Generator().manual_seed(seed)
    foreground = samples["foreground"].bool()
    background_indices = torch.where(~foreground)[0]
    positive_groups = {}
    for index, category in enumerate(samples["categories"]):
        if foreground[index]:
            positive_groups.setdefault(category, []).append(index)
    if not len(background_indices) or not positive_groups:
        return torch.randperm(len(foreground), generator=generator)

    epoch_size = len(foreground)
    positive_count = epoch_size // 2
    background_count = epoch_size - positive_count
    selected = [
        background_indices[torch.randint(
            len(background_indices), (background_count,), generator=generator
        )]
    ]
    group_names = sorted(positive_groups)
    base = positive_count // len(group_names)
    remainder = positive_count % len(group_names)
    for group_index, name in enumerate(group_names):
        group = torch.tensor(positive_groups[name], dtype=torch.long)
        count = base + int(group_index < remainder)
        selected.append(group[torch.randint(
            len(group), (count,), generator=generator
        )])
    order = torch.cat(selected)
    return order[torch.randperm(len(order), generator=generator)]


def train_refiner(
        train_samples, val_samples, feature_dim, hidden_dim, num_classes,
        device, seed, max_epochs, patience, learning_rate, train_batch_size,
        max_delta_scale, class_weight, regression_weight, feature_sigma):
    reset_audit_rng(seed)
    refiner = PLSRRefiner(
        feature_dim, hidden_dim, num_classes
    ).to(device)
    optimizer = torch.optim.AdamW(
        refiner.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    selectors = {
        "min_loss": {
            "value": float("inf"), "direction": "min",
            "epoch": 0, "state": copy.deepcopy(refiner.state_dict()),
        },
        "max_mean_delta_iou": {
            "value": -float("inf"), "direction": "max",
            "epoch": 0, "state": copy.deepcopy(refiner.state_dict()),
        },
        "max_iou50_gain": {
            "value": -float("inf"), "direction": "max",
            "epoch": 0, "state": copy.deepcopy(refiner.state_dict()),
        },
    }
    remaining_patience = patience
    history = []

    for epoch in range(1, max_epochs + 1):
        refiner.train()
        generator = torch.Generator().manual_seed(seed + epoch)
        order = balanced_epoch_order(train_samples, seed + epoch)
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), train_batch_size):
            indices = order[start:start + train_batch_size]
            rois = train_samples["rois"][indices].to(device)
            noise_mask = train_samples["feature_noise"][indices].to(device)
            if noise_mask.any() and feature_sigma > 0:
                noise = torch.randn(
                    rois.shape, generator=generator, dtype=rois.dtype
                ).to(device)
                rois = rois + (
                    feature_sigma
                    * noise
                    * noise_mask[:, None, None].to(rois.dtype)
                )
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
            "validation_iou50_gain": validation["iou50_rate_gain"],
        })
        values = {
            "min_loss": validation["loss"],
            "max_mean_delta_iou": validation["mean_delta_iou"],
            "max_iou50_gain": validation["iou50_rate_gain"],
        }
        improved = False
        for name, value in values.items():
            target = selectors[name]
            better = (
                value < target["value"] - 1e-6
                if target["direction"] == "min"
                else value > target["value"] + 1e-6
            )
            if better:
                target["value"] = value
                target["epoch"] = epoch
                target["state"] = copy.deepcopy(refiner.state_dict())
                improved = True
        if improved:
            remaining_patience = patience
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break

    selected_refiners, selected_metrics = {}, {}
    for name, selector in selectors.items():
        selected = PLSRRefiner(
            feature_dim, hidden_dim, num_classes
        ).to(device)
        selected.load_state_dict(selector["state"])
        selected.eval()
        selected_refiners[name] = selected
        selected_metrics[name] = {
            "epoch": selector["epoch"],
            "validation": evaluate_refiner(
                selected, val_samples, device, train_batch_size,
                max_delta_scale, class_weight, regression_weight,
            ),
        }
    return selected_refiners, {
        "seed": seed,
        "selectors": selected_metrics,
        "history": history,
    }


def infer_refinement(
        refiner, rois, segments, labels, valid_lengths, max_delta_scale,
        confidence_mode="pseudo_class"):
    logits, offsets = refiner(rois)
    refined = apply_boundary_offsets(segments, offsets, max_delta_scale)
    refined = clip_segments(refined, valid_lengths, fallback=segments)
    probabilities = logits.sigmoid()
    if confidence_mode == "pseudo_class":
        confidence = probabilities.gather(1, labels[:, None]).squeeze(1)
    elif confidence_mode == "max_class":
        confidence = probabilities.max(dim=1).values
    else:
        raise ValueError("unknown confidence mode: {}".format(confidence_mode))
    return refined, confidence


def synthesize_boundaries(
        refiner, direct_segments, feature_maps, masks, centers,
        video_indices, labels, valid_lengths, roi_size, max_delta_scale,
        repeats, boundary_sigma, feature_sigma, random_seed,
        confidence_mode):
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
        repeated_lengths, max_delta_scale, confidence_mode,
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
        model, refiners, selector_groups, dataset, annotations, cfg,
        device, levels,
        batch_size, target_seed, roi_size, max_delta_scale,
        repeats, boundary_sigma, feature_sigma):
    reset_audit_rng(target_seed)
    variants = ["baseline"]
    variants.extend("direct_{}".format(name) for name in refiners)
    for selector in selector_groups:
        variants.extend([
            "direct_{}_ensemble".format(selector),
            "synthesis_pseudo_class_{}_ensemble".format(selector),
            "synthesis_max_class_{}_ensemble".format(selector),
        ])
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
            batch_proposals = sum(
                int(segments.shape[0]) for segments in baseline_segments
            )
            flat_segments = (
                torch.cat(baseline_segments)
                if batch_proposals
                else fpn_features[0].new_empty((0, 2))
            )
            flat_labels = (
                torch.cat(baseline_labels)
                if batch_proposals
                else torch.empty(0, dtype=torch.long, device=device)
            )
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
            direct_outputs = {}
            synthesis_outputs = {
                "pseudo_class": {},
                "max_class": {},
            }
            if batch_proposals:
                for name, refiner in refiners.items():
                    direct, _ = infer_refinement(
                        refiner, rois, flat_segments, flat_labels,
                        valid_lengths, max_delta_scale,
                    )
                    direct_outputs[name] = direct
                    flat_variants["direct_{}".format(name)] = direct
                    for confidence_mode in synthesis_outputs:
                        synthesis_outputs[confidence_mode][name] = (
                            synthesize_boundaries(
                                refiner, direct,
                                fpn_features[0], fpn_masks[0],
                                points[0][:, 0],
                                video_indices, flat_labels, valid_lengths,
                                roi_size, max_delta_scale, repeats,
                                boundary_sigma, feature_sigma,
                                target_seed + batch_start * 1000,
                                confidence_mode,
                            )
                        )
                for selector, names in selector_groups.items():
                    flat_variants[
                        "direct_{}_ensemble".format(selector)
                    ] = torch.stack([
                        direct_outputs[name] for name in names
                    ]).mean(dim=0)
                    for confidence_mode in synthesis_outputs:
                        flat_variants[
                            "synthesis_{}_{}_ensemble".format(
                                confidence_mode, selector
                            )
                        ] = torch.stack([
                            synthesis_outputs[confidence_mode][name]
                            for name in names
                        ]).mean(dim=0)
            else:
                for variant in variants[1:]:
                    flat_variants[variant] = flat_segments.clone()

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


def summarize_source_samples(samples):
    category_counts = {}
    for category in samples["categories"]:
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "videos_or_windows": samples["videos"],
        "proposals": samples["proposals"],
        "foreground_proposals_iou_ge_0_1": int(
            samples["foreground"].sum()
        ),
        "background_proposals_iou_lt_0_1": int(
            (~samples["foreground"]).sum()
        ),
        "category_counts": category_counts,
    }


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
        cfg["dataset_name"], False, cfg["src_train_split"], **cfg["dataset"]
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

    max_seq_len = int(cfg["dataset"]["max_seq_len"])
    training_windows = DeterministicWindowDataset(
        source_train, max_seq_len, max_seq_len
    )
    validation_windows = DeterministicWindowDataset(
        source_val, max_seq_len, max_seq_len // 2
    )
    train_samples = collect_source_samples(
        model, training_windows, args.device, batch_size,
        args.source_seed, args.roi_size,
        include_teacher=True, include_synthetic=True,
    )
    val_samples = collect_source_samples(
        model, validation_windows, args.device, batch_size,
        args.source_seed, args.roi_size,
        include_teacher=True, include_synthetic=False,
    )
    feature_dim = train_samples["rois"].shape[1]
    refiners, selector_groups, training = {}, {}, {}
    for seed in refiner_seeds:
        selected, metrics = train_refiner(
            train_samples, val_samples, feature_dim, args.hidden_dim,
            model.num_class_orig, args.device, seed,
            args.max_epochs, args.patience, args.learning_rate,
            args.train_batch_size, args.max_delta_scale,
            args.class_weight, args.regression_weight,
            args.feature_sigma,
        )
        for selector, refiner in selected.items():
            name = "seed{}_{}".format(seed, selector)
            refiners[name] = refiner
            selector_groups.setdefault(selector, []).append(name)
        training[str(seed)] = metrics

    target = collect_target_results(
        model, refiners, selector_groups, target_train, annotations, cfg,
        args.device, levels, batch_size, args.target_seed, args.roi_size,
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
        "audit": "plr_style_distribution_balanced_source_refinement",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "protocol": {
            "refiner_supervision": "source GT only",
            "target_gt_usage": "offline evaluation only",
            "source_validation": (
                "deterministic full-video sliding windows with 50% overlap"
            ),
            "source_training_candidates": (
                "EMA proposals plus GT jitter in four IoU bands, "
                "BS-expanded proposals and true background proposals"
            ),
            "source_training_windows": (
                "deterministic non-overlapping full-video coverage plus "
                "one final boundary-aligned window"
            ),
            "source_epoch_sampler": (
                "50% true background; foreground categories sampled equally"
            ),
            "source_positive_rule": "max source-GT IoU >= 0.1",
            "candidate_pipeline": (
                "EMA -> alpha -> hard class-agnostic NMS; proposal set fixed"
            ),
            "plsr": (
                "FPN0 temporal RoI interpolation -> two Conv1d -> "
                "class and normalized start/end offset branches"
            ),
            "roi_size": args.roi_size,
            "max_delta_scale": args.max_delta_scale,
            "refiner_seeds": refiner_seeds,
            "checkpoint_selection": [
                "minimum source-validation total loss",
                "maximum source-validation mean delta IoU",
                "maximum source-validation IoU>=0.5 rate gain",
            ],
            "boundary_synthesis": {
                "repeats": args.synthesis_repeats,
                "boundary_sigma": args.boundary_sigma,
                "feature_sigma": args.feature_sigma,
                "fusion_modes": [
                    "original pseudo-class confidence",
                    "maximum predicted class confidence",
                ],
            },
            "excluded": (
                "AWL, target training, target classification, re-NMS, "
                "threshold or loss changes"
            ),
        },
        "source": {
            "train": summarize_source_samples(train_samples),
            "validation": summarize_source_samples(val_samples),
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
