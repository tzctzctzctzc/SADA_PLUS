"""Proposal-region features and a source-supervised localization-quality head."""

import torch
from torch import nn
from torch.nn import functional as F


PROTOCOL_VERSION = "proposal-region-iou-quality-v1"


def _sample_1d(features, coordinates):
    """Linearly sample ``[C, T]`` features at ``[N, K]`` coordinates."""
    if features.ndim != 2 or coordinates.ndim != 2:
        raise ValueError("features and coordinates must have shapes [C,T] and [N,K]")
    length = features.shape[-1]
    coordinates = coordinates.clamp(0, max(length - 1, 0))
    left = coordinates.floor().long()
    right = (left + 1).clamp(max=max(length - 1, 0))
    weight = (coordinates - left).unsqueeze(-1)
    transposed = features.transpose(0, 1)
    return transposed[left] * (1.0 - weight) + transposed[right] * weight


def proposal_region_context(features, segments, stride, samples=4):
    """Pool proposal interior and start/end state changes from one FPN level.

    ``segments`` use the base feature-grid coordinates used by SADA.  The output
    contains four C-dimensional blocks: proposal mean, center evidence, start
    change, and end change.  The last two compare inside-boundary evidence with
    equally sized outside context.
    """
    if segments.ndim != 2 or segments.shape[-1] != 2:
        raise ValueError("segments must have shape [N,2]")
    if samples < 2:
        raise ValueError("samples must be at least two")
    if not len(segments):
        return features.new_empty((0, features.shape[0] * 4))

    start = segments[:, 0:1] / float(stride)
    end = segments[:, 1:2] / float(stride)
    duration = (end - start).clamp_min(1e-3)
    device = segments.device
    dtype = segments.dtype

    inside_fraction = torch.linspace(
        0.125, 0.875, samples, device=device, dtype=dtype
    )[None]
    edge_fraction = torch.linspace(
        0.0, 0.2, samples, device=device, dtype=dtype
    )[None]
    outside_fraction = torch.linspace(
        0.05, 0.25, samples, device=device, dtype=dtype
    )[None]

    inside = _sample_1d(features, start + duration * inside_fraction).mean(1)
    center = _sample_1d(features, start + duration * 0.5).squeeze(1)
    start_inside = _sample_1d(features, start + duration * edge_fraction).mean(1)
    start_outside = _sample_1d(
        features, start - duration * outside_fraction
    ).mean(1)
    end_inside = _sample_1d(features, end - duration * edge_fraction).mean(1)
    end_outside = _sample_1d(
        features, end + duration * outside_fraction
    ).mean(1)
    return torch.cat(
        (inside, center, start_inside - start_outside, end_inside - end_outside),
        dim=-1,
    )


def proposal_scalar_features(
    scores,
    margins,
    segments,
    points,
    boundary_quality,
    class_indices,
    levels,
    num_classes,
    num_levels,
):
    """Return scale-stable detector geometry plus class/level identities."""
    duration = (segments[:, 1] - segments[:, 0]).clamp_min(1e-4)
    left = (points - segments[:, 0]).clamp_min(0)
    right = (segments[:, 1] - points).clamp_min(0)
    continuous = torch.stack(
        (
            torch.logit(scores.clamp(1e-5, 1 - 1e-5)),
            margins,
            duration.log(),
            left / duration,
            right / duration,
            boundary_quality,
        ),
        dim=-1,
    )
    class_one_hot = F.one_hot(class_indices, num_classes).to(continuous.dtype)
    level_one_hot = F.one_hot(levels, num_levels).to(continuous.dtype)
    return torch.cat((continuous, class_one_hot, level_one_hot), dim=-1)


def same_class_temporal_iou(segments, class_indices, gt_segments, gt_labels):
    """Maximum IoU with a same-class GT segment; unmatched proposals get zero."""
    result = segments.new_zeros(len(segments))
    for class_index in class_indices.unique().tolist():
        proposal_mask = class_indices == class_index
        gt_mask = gt_labels == class_index
        if not gt_mask.any():
            continue
        proposals = segments[proposal_mask]
        targets = gt_segments[gt_mask]
        intersection = (
            torch.minimum(proposals[:, None, 1], targets[None, :, 1])
            - torch.maximum(proposals[:, None, 0], targets[None, :, 0])
        ).clamp_min(0)
        union = (
            proposals[:, None, 1]
            - proposals[:, None, 0]
            + targets[None, :, 1]
            - targets[None, :, 0]
            - intersection
        ).clamp_min(1e-6)
        result[proposal_mask] = (intersection / union).amax(dim=1)
    return result


class ProposalIoUQualityHead(nn.Module):
    """Small class-agnostic MLP trained to predict continuous proposal IoU."""

    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def soft_iou_quality_loss(logits, targets):
    """Calibrate IoU while preserving useful gradients near both endpoints."""
    targets = targets.clamp(0, 1)
    calibration = F.binary_cross_entropy_with_logits(logits, targets)
    regression = F.smooth_l1_loss(logits.sigmoid(), targets)
    return calibration + regression


def quality_weighted_scores(scores, qualities, gamma):
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if gamma == 0:
        return scores
    return scores * qualities.clamp_min(1e-5).pow(float(gamma))


def video_class_rank_weighted_scores(scores, qualities, labels, strength):
    """Apply a bounded quality residual within each video/class candidate set.

    Absolute quality calibration does not transfer reliably across videos.  TAL
    NMS, however, mainly needs the ordering of competing same-class proposals.
    Percentile ranks preserve that evidence while limiting every score change to
    ``exp(-strength) .. exp(strength)``.
    """
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if strength == 0:
        return scores
    residual = torch.zeros_like(qualities)
    for class_index in labels.unique().tolist():
        selected = labels == class_index
        count = int(selected.sum())
        if count < 2:
            continue
        order = qualities[selected].argsort(stable=True)
        ranks = torch.empty_like(order, dtype=qualities.dtype)
        ranks[order] = torch.arange(count, device=qualities.device, dtype=qualities.dtype)
        residual[selected] = 2.0 * ranks / float(count - 1) - 1.0
    return scores * torch.exp(float(strength) * residual)
