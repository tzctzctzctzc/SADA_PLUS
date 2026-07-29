import math

import torch


def npl_boundary_ambiguity(segments, valid_mask, radii, eps=1e-6):
    """Compute NPL boundary variance from adjacent dense predictions.

    The returned ambiguity is
    ``(Var(start) + Var(end)) / current_segment_duration``.
    """
    if segments.ndim != 2 or segments.shape[-1] != 2:
        raise ValueError("segments must have shape [T, 2]")
    if valid_mask.shape != segments.shape[:1]:
        raise ValueError("valid_mask must have shape [T]")
    radii = tuple(int(radius) for radius in radii)
    if not radii or any(radius < 1 for radius in radii):
        raise ValueError("radii must contain positive integers")

    finite = torch.isfinite(segments).all(dim=-1)
    duration = segments[:, 1] - segments[:, 0]
    usable = valid_mask.bool() & finite & (duration > eps)
    ambiguity = segments.new_full((segments.shape[0], len(radii)), float("inf"))
    neighbor_counts = torch.zeros(
        (segments.shape[0], len(radii)),
        dtype=torch.long,
        device=segments.device,
    )

    for index in torch.where(usable)[0].tolist():
        for radius_index, radius in enumerate(radii):
            left = max(0, index - radius)
            right = min(segments.shape[0], index + radius + 1)
            neighbors = segments[left:right][usable[left:right]]
            neighbor_counts[index, radius_index] = neighbors.shape[0]
            if neighbors.shape[0] < 2:
                continue
            boundary_variance = (
                neighbors[:, 0].var(unbiased=False)
                + neighbors[:, 1].var(unbiased=False)
            )
            ambiguity[index, radius_index] = boundary_variance / duration[index]
    return ambiguity, neighbor_counts


def match_post_nms_to_pre_nms(
        post_segments, post_scores, post_labels,
        pre_segments, pre_scores, pre_labels,
        atol=1e-6):
    """Recover the pre-NMS row retained by hard NMS."""
    if post_segments.numel() == 0:
        return torch.empty(
            (0,), dtype=torch.long, device=post_segments.device
        )
    segment_match = torch.isclose(
        post_segments[:, None], pre_segments[None], atol=atol, rtol=0
    ).all(dim=-1)
    score_match = torch.isclose(
        post_scores[:, None], pre_scores[None], atol=atol, rtol=0
    )
    label_match = post_labels[:, None] == pre_labels[None]
    matches = segment_match & score_match & label_match
    if not matches.any(dim=1).all():
        raise RuntimeError("hard-NMS output could not be matched to pre-NMS input")
    indices = matches.to(torch.int64).argmax(dim=1)
    if torch.unique(indices).numel() != indices.numel():
        raise RuntimeError("multiple hard-NMS outputs matched the same input")
    return indices


def matched_budget_indices(values, fraction, largest=True):
    """Select an exact top-k budget without threshold tie inflation."""
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if values.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=values.device)
    count = min(values.numel(), max(1, math.ceil(values.numel() * fraction)))
    return torch.argsort(values, descending=largest, stable=True)[:count]


def aligned_tiou_tnd(predicted_segments, target_segments, eps=1e-6):
    """Compute aligned temporal IoU and normalized center distance."""
    if (
        predicted_segments.shape != target_segments.shape
        or predicted_segments.ndim != 2
        or predicted_segments.shape[-1] != 2
    ):
        raise ValueError("aligned segments must both have shape [N, 2]")
    left = torch.maximum(predicted_segments[:, 0], target_segments[:, 0])
    right = torch.minimum(predicted_segments[:, 1], target_segments[:, 1])
    intersection = (right - left).clamp_min(0)
    pred_length = (
        predicted_segments[:, 1] - predicted_segments[:, 0]
    ).clamp_min(0)
    target_length = (
        target_segments[:, 1] - target_segments[:, 0]
    ).clamp_min(0)
    tiou = intersection / (
        pred_length + target_length - intersection
    ).clamp_min(eps)

    pred_center = predicted_segments.mean(dim=-1)
    target_center = target_segments.mean(dim=-1)
    cover_left = torch.minimum(
        predicted_segments[:, 0], target_segments[:, 0]
    )
    cover_right = torch.maximum(
        predicted_segments[:, 1], target_segments[:, 1]
    )
    cover_length = (cover_right - cover_left).clamp_min(eps)
    tnd = (pred_center - target_center).square() / cover_length.square()
    return tiou, tnd
