import hashlib
import math

import torch


def alqa_joint_score(
        confidence, predicted_tiou, predicted_tnd, epsilon=1e-6):
    return confidence * (predicted_tiou - predicted_tnd).clamp_min(epsilon)


def select_same_budget_alqa(
        confidence, joint_score, usable, alpha):
    """Replace the original alpha set without changing its dense budget."""
    candidates = torch.where(usable)[0]
    budget = int((usable & (confidence >= alpha)).sum())
    if budget == 0:
        return candidates[:0], 0
    order = torch.argsort(
        joint_score[candidates], descending=True, stable=True
    )
    return candidates[order[:budget]], budget


def proposal_loss_terms(anchor_losses, proposal_ids, proposal_weights):
    """Return weighted proposal-mean loss terms without reducing a batch."""
    if anchor_losses.ndim != 1 or proposal_ids.shape != anchor_losses.shape:
        raise ValueError("anchor losses and proposal ids must be 1D and aligned")
    if proposal_weights.ndim != 1:
        raise ValueError("proposal weights must be 1D")
    valid = (
        (proposal_ids >= 0)
        & (proposal_ids < proposal_weights.numel())
    )
    if not valid.any():
        zero = anchor_losses.sum() * 0.0
        return zero, zero.detach(), zero.detach(), zero.detach()

    ids = proposal_ids[valid].long()
    losses = anchor_losses[valid]
    sums = losses.new_zeros(proposal_weights.numel())
    counts = losses.new_zeros(proposal_weights.numel())
    sums.scatter_add_(0, ids, losses)
    counts.scatter_add_(0, ids, torch.ones_like(losses))
    present = counts > 0
    weights = proposal_weights.detach()[present].to(losses.dtype)
    proposal_means = sums[present] / counts[present]
    numerator = (weights * proposal_means).sum()
    denominator = weights.sum()
    weight_square_sum = weights.square().sum()
    proposal_count = present.sum().to(losses.dtype)
    return numerator, denominator, weight_square_sum, proposal_count


def aligned_boundary_iou_from_offsets(first, second, eps=1e-6):
    """Compute same-anchor segment IoU from left/right boundary offsets."""
    if first.shape != second.shape or first.shape[-1] != 2:
        raise ValueError("offset tensors must have the same [..., 2] shape")
    first = first.clamp_min(0)
    second = second.clamp_min(0)
    intersection = (
        torch.minimum(first[..., 0], second[..., 0])
        + torch.minimum(first[..., 1], second[..., 1])
    )
    union = (
        torch.maximum(first[..., 0], second[..., 0])
        + torch.maximum(first[..., 1], second[..., 1])
    )
    return (intersection / union.clamp_min(eps)).clamp(0.0, 1.0)


def weighted_anchor_mse(predictions, targets, weights, mask, eps=1e-8):
    """Return detached-weight MSE normalized by the selected weight sum."""
    if predictions.shape != targets.shape or predictions.shape[-1] != 2:
        raise ValueError("prediction and target offsets must align as [..., 2]")
    if weights.shape != predictions.shape[:-1] or mask.shape != weights.shape:
        raise ValueError("weights and mask must align with anchor dimensions")
    selected = mask.bool()
    if not selected.any():
        return predictions.sum() * 0.0
    selected_weights = weights.detach()[selected].to(predictions.dtype)
    per_anchor = (
        predictions[selected] - targets.detach()[selected]
    ).square().mean(dim=-1)
    return (
        selected_weights * per_anchor
    ).sum() / selected_weights.sum().clamp_min(eps)


def top_fraction_per_sample_mask(scores, eligible, fraction):
    """Keep the highest-scoring fraction independently in each sample."""
    if scores.shape != eligible.shape or scores.ndim != 2:
        raise ValueError("scores and eligible must be aligned [B, N] tensors")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    kept = torch.zeros_like(eligible, dtype=torch.bool)
    for sample_index in range(scores.shape[0]):
        indices = torch.where(eligible[sample_index])[0]
        if not indices.numel():
            continue
        count = min(
            indices.numel(),
            max(1, math.ceil(indices.numel() * fraction)),
        )
        order = torch.argsort(
            scores[sample_index, indices],
            descending=True,
            stable=True,
        )
        kept[sample_index, indices[order[:count]]] = True
    return kept


def deterministic_gaussian_perturbation(
        inputs, masks, sample_keys, noise_std, base_seed, epoch):
    """Apply independent reproducible Gaussian noise to each valid sample."""
    if inputs.ndim != 3 or masks.shape != (inputs.shape[0], 1, inputs.shape[2]):
        raise ValueError("inputs and masks must have shapes [B,C,T] and [B,1,T]")
    if len(sample_keys) != inputs.shape[0] or noise_std <= 0:
        raise ValueError("sample keys must align and noise std must be positive")
    perturbed = inputs.clone()
    for index, sample_key in enumerate(sample_keys):
        token = '{}|{}|{}'.format(base_seed, epoch, sample_key)
        seed = int.from_bytes(
            hashlib.sha256(token.encode('utf-8')).digest()[:8], 'little'
        ) % (2 ** 31)
        generator = torch.Generator(device=inputs.device)
        generator.manual_seed(seed)
        noise = torch.randn(
            inputs[index].shape,
            dtype=inputs.dtype,
            device=inputs.device,
            generator=generator,
        )
        perturbed[index] = (
            inputs[index]
            + noise_std * noise * masks[index].to(inputs.dtype)
        )
    return perturbed


def temporal_roi_interpolate(
        features, segment, point_centers, valid_length, output_size):
    """Linearly sample a CxT feature map inside one temporal segment."""
    if features.ndim != 2:
        raise ValueError("features must have shape [C, T]")
    if segment.shape != (2,):
        raise ValueError("segment must have shape [2]")
    if output_size < 2 or valid_length < 1:
        raise ValueError("output_size must be >=2 and valid_length >=1")
    valid_length = min(int(valid_length), features.shape[-1])
    centers = point_centers[:valid_length]
    if centers.numel() == 1:
        return features[:, :1].expand(-1, output_size)
    step = (centers[-1] - centers[0]) / (centers.numel() - 1)
    positions = torch.linspace(
        0.0, 1.0, output_size,
        dtype=features.dtype, device=features.device,
    )
    positions = segment[0] + positions * (segment[1] - segment[0])
    indices = ((positions - centers[0]) / step.clamp_min(1e-6)).clamp(
        0, valid_length - 1
    )
    left = indices.floor().long()
    right = (left + 1).clamp_max(valid_length - 1)
    weight = (indices - left.to(indices.dtype)).unsqueeze(0)
    return (
        features[:, left] * (1.0 - weight)
        + features[:, right] * weight
    )


def apply_boundary_offsets(segments, normalized_offsets, max_scale):
    """Add duration-normalized start/end offsets while keeping positive length."""
    if segments.shape != normalized_offsets.shape or segments.shape[-1] != 2:
        raise ValueError("segments and offsets must both have shape [N, 2]")
    if not 0.0 <= max_scale < 0.5:
        raise ValueError("max_scale must satisfy 0 <= max_scale < 0.5")
    duration = (segments[:, 1] - segments[:, 0]).clamp_min(1e-6)
    scaled = normalized_offsets.clamp(-1.0, 1.0) * max_scale
    return torch.stack((
        segments[:, 0] + scaled[:, 0] * duration,
        segments[:, 1] + scaled[:, 1] * duration,
    ), dim=-1)


def clip_segments(segments, valid_lengths, fallback=None):
    """Clip per-video segments and fall back if clipping collapses a segment."""
    clipped = segments.clone()
    clipped[:, 0].clamp_(min=0.0)
    clipped[:, 1].clamp_(min=0.0)
    clipped[:, 0] = torch.minimum(clipped[:, 0], valid_lengths)
    clipped[:, 1] = torch.minimum(clipped[:, 1], valid_lengths)
    invalid = (
        ~torch.isfinite(clipped).all(dim=-1)
        | ((clipped[:, 1] - clipped[:, 0]) <= 1e-6)
    )
    if invalid.any():
        if fallback is None:
            raise RuntimeError("boundary refinement produced invalid segments")
        clipped[invalid] = fallback[invalid]
    return clipped


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
