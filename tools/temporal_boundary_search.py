"""Pure temporal-evidence boundary search used by the offline P0 audit."""

import torch
from torch.nn import functional as F


def aligned_segment_iou(prediction, target):
    left = torch.maximum(prediction[..., 0], target[..., 0])
    right = torch.minimum(prediction[..., 1], target[..., 1])
    intersection = (right - left).clamp_min(0)
    union = (
        (prediction[..., 1] - prediction[..., 0]).clamp_min(0)
        + (target[..., 1] - target[..., 0]).clamp_min(0)
        - intersection
    )
    return intersection / union.clamp_min(1e-6)


def _standardize(values):
    if values.numel() < 2:
        return torch.zeros_like(values)
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def _cosine(vector, prototype):
    return F.cosine_similarity(vector.unsqueeze(0), prototype.unsqueeze(0))[0]


def boundary_evidence(
        points, features, actionness, coarse_boundary, duration,
        search_radius_ratio, window_ratio, inside_prototype,
        outside_prototype, side):
    """Compute directional evidence at candidates around one coarse boundary."""
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    radius = max(float(duration) * float(search_radius_ratio), 1.0)
    width = max(float(duration) * float(window_ratio), 1.0)
    candidates = points[(points >= coarse_boundary - radius)
                        & (points <= coarse_boundary + radius)]
    candidates = torch.cat((
        candidates,
        points.new_tensor([float(coarse_boundary)]),
    )).unique(sorted=True)

    kept, action_scores, contrast_scores, semantic_scores = [], [], [], []
    for candidate in candidates:
        before = (points >= candidate - width) & (points < candidate)
        after = (points >= candidate) & (points <= candidate + width)
        if not before.any() or not after.any():
            continue
        before_feature = features[before].mean(dim=0)
        after_feature = features[after].mean(dim=0)
        before_action = actionness[before].mean()
        after_action = actionness[after].mean()
        contrast = 1.0 - _cosine(before_feature, after_feature)
        before_in = _cosine(before_feature, inside_prototype)
        after_in = _cosine(after_feature, inside_prototype)
        before_out = _cosine(before_feature, outside_prototype)
        after_out = _cosine(after_feature, outside_prototype)
        if side == "left":
            action = after_action - before_action
            semantic = 0.5 * (
                (after_in - before_in) + (before_out - after_out)
            )
        else:
            action = before_action - after_action
            semantic = 0.5 * (
                (before_in - after_in) + (after_out - before_out)
            )
        kept.append(candidate)
        action_scores.append(action)
        contrast_scores.append(contrast)
        semantic_scores.append(semantic)

    if not kept:
        empty = points.new_empty(0)
        return {
            "candidates": empty,
            "action": empty,
            "contrast": empty,
            "semantic": empty,
            "distance": empty,
        }
    kept = torch.stack(kept)
    return {
        "candidates": kept,
        "action": _standardize(torch.stack(action_scores)),
        "contrast": _standardize(torch.stack(contrast_scores)),
        "semantic": _standardize(torch.stack(semantic_scores)),
        "distance": (kept - float(coarse_boundary)).abs()
        / max(float(duration), 1e-6),
    }


def select_boundary(evidence, coarse_boundary, parameters):
    candidates = evidence["candidates"]
    if candidates.numel() < 2:
        return float(coarse_boundary), {
            "applied": False, "margin": 0.0, "shift_ratio": 0.0,
        }
    score = (
        evidence["action"]
        + parameters["feature_weight"] * evidence["contrast"]
        + parameters["semantic_weight"] * evidence["semantic"]
        - parameters["distance_weight"] * evidence["distance"]
    )
    values, indices = torch.topk(score, k=2)
    selected = candidates[indices[0]]
    margin = float(values[0] - values[1])
    duration = float(parameters["duration"])
    shift_ratio = abs(float(selected) - float(coarse_boundary)) / max(duration, 1e-6)
    applied = (
        margin >= parameters["margin_threshold"]
        and shift_ratio <= parameters["max_shift_ratio"]
        and abs(float(selected) - float(coarse_boundary)) > 1e-6
    )
    return (float(selected) if applied else float(coarse_boundary)), {
        "applied": applied,
        "margin": margin,
        "shift_ratio": shift_ratio,
    }


def refine_with_evidence(segment, left_evidence, right_evidence, parameters):
    duration = float(segment[1] - segment[0])
    if duration <= 0:
        return segment.clone(), {
            "left": {"applied": False}, "right": {"applied": False}
        }
    local_parameters = {**parameters, "duration": duration}
    left, left_details = select_boundary(
        left_evidence, float(segment[0]), local_parameters
    )
    right, right_details = select_boundary(
        right_evidence, float(segment[1]), local_parameters
    )
    if left >= right:
        return segment.clone(), {
            "left": {**left_details, "applied": False},
            "right": {**right_details, "applied": False},
        }
    return segment.new_tensor([left, right]), {
        "left": left_details,
        "right": right_details,
    }


def refine_segment(
        points, features, actionness, segment, inside_prototype,
        outside_prototype, parameters):
    duration = float(segment[1] - segment[0])
    if duration <= 0:
        return segment.clone(), {
            "left": {"applied": False}, "right": {"applied": False}
        }
    left_evidence = boundary_evidence(
        points, features, actionness, float(segment[0]), duration,
        parameters["search_radius_ratio"], parameters["window_ratio"],
        inside_prototype, outside_prototype, "left",
    )
    right_evidence = boundary_evidence(
        points, features, actionness, float(segment[1]), duration,
        parameters["search_radius_ratio"], parameters["window_ratio"],
        inside_prototype, outside_prototype, "right",
    )
    return refine_with_evidence(
        segment, left_evidence, right_evidence, parameters
    )
