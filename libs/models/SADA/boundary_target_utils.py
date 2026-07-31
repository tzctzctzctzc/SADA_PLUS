import torch
from torch.nn import functional as F


def _plain_mask(mask):
    return mask[:, 0] if mask.ndim == 3 else mask


@torch.no_grad()
def build_boundary_heatmaps(
        points, masks, gt_segments, regression_ranges, sigma=1.0):
    """Build scale-aware Gaussian start/end targets on each FPN grid."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not (len(points) == len(masks) == len(regression_ranges)):
        raise ValueError("points, masks and regression ranges must align")
    starts, ends = [], []
    for level, (level_points, level_mask, level_range) in enumerate(zip(
            points, masks, regression_ranges)):
        valid = _plain_mask(level_mask).bool()
        start_target = level_points.new_zeros(valid.shape)
        end_target = level_points.new_zeros(valid.shape)
        centers = level_points[:, 0]
        stride = float(level_points[0, 3])
        low, high = map(float, level_range)
        for batch_index, segments in enumerate(gt_segments):
            if segments.numel() == 0:
                continue
            durations = segments[:, 1] - segments[:, 0]
            scale = durations / 2.0
            eligible = (scale >= low) & (scale <= high) & (durations > 0)
            for segment in segments[eligible]:
                start_gaussian = torch.exp(
                    -0.5 * ((centers - segment[0]) / (sigma * stride)) ** 2
                )
                end_gaussian = torch.exp(
                    -0.5 * ((centers - segment[1]) / (sigma * stride)) ** 2
                )
                start_target[batch_index] = torch.maximum(
                    start_target[batch_index], start_gaussian
                )
                end_target[batch_index] = torch.maximum(
                    end_target[batch_index], end_gaussian
                )
        starts.append(start_target * valid.to(start_target.dtype))
        ends.append(end_target * valid.to(end_target.dtype))
    return starts, ends


def boundary_focal_loss(
        start_logits, end_logits, start_targets, end_targets, masks,
        alpha=0.25, gamma=2.0):
    """Focal loss normalized by Gaussian boundary mass."""
    if not (
        len(start_logits) == len(end_logits) == len(start_targets)
        == len(end_targets) == len(masks)
    ):
        raise ValueError("boundary predictions, targets and masks must align")
    numerator = start_logits[0].sum() * 0.0
    normalizer = numerator.detach()
    for start, end, start_target, end_target, mask in zip(
            start_logits, end_logits, start_targets, end_targets, masks):
        valid = _plain_mask(mask).bool()
        for logits, target in ((start, start_target), (end, end_target)):
            logits = logits.float()
            target = target.float()
            probability = logits.sigmoid()
            ce = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            p_t = probability * target + (1.0 - probability) * (1.0 - target)
            alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
            numerator = numerator + (
                ce * (1.0 - p_t).pow(gamma) * alpha_t
            )[valid].sum()
            normalizer = normalizer + target[valid].sum()
    return numerator / normalizer.clamp_min(1.0)


def select_boundary_candidate(
        points, probabilities, coarse_boundary, duration, parameters):
    """Select one boundary peak; a rejected candidate carries no supervision."""
    radius = float(parameters["search_radius_ratio"]) * float(duration)
    candidates = torch.where(
        (points >= float(coarse_boundary) - radius)
        & (points <= float(coarse_boundary) + radius)
    )[0]
    if candidates.numel() < 2 or duration <= 0:
        return float(coarse_boundary), {
            "accepted": False, "peak": 0.0, "gain": 0.0,
            "margin": 0.0, "shift_ratio": 0.0,
        }
    candidate_probabilities = probabilities[candidates]
    values, order = torch.topk(candidate_probabilities, 2)
    selected_index = candidates[order[0]]
    selected = float(points[selected_index])
    coarse_index = torch.argmin((points - float(coarse_boundary)).abs())
    peak = float(values[0])
    gain = peak - float(probabilities[coarse_index])
    margin = peak - float(values[1])
    shift_ratio = abs(selected - float(coarse_boundary)) / float(duration)
    accepted = (
        peak >= float(parameters["absolute_threshold"])
        and gain >= float(parameters["gain_threshold"])
        and margin >= float(parameters["margin_threshold"])
        and shift_ratio <= float(parameters["max_shift_ratio"])
        and abs(selected - float(coarse_boundary)) > 1e-6
    )
    return (selected if accepted else float(coarse_boundary)), {
        "accepted": accepted,
        "peak": peak,
        "gain": gain,
        "margin": margin,
        "shift_ratio": shift_ratio,
    }


def refine_segment_from_boundary_maps(
        points, start_probabilities, end_probabilities, segment, parameters):
    """Refine left/right independently while preserving a valid duration."""
    duration = float(segment[1] - segment[0])
    left, left_detail = select_boundary_candidate(
        points, start_probabilities, float(segment[0]), duration, parameters
    )
    right, right_detail = select_boundary_candidate(
        points, end_probabilities, float(segment[1]), duration, parameters
    )
    new_duration = right - left
    minimum = float(parameters.get("min_duration_ratio", 0.5)) * duration
    maximum = float(parameters.get("max_duration_ratio", 1.5)) * duration
    if left >= right or new_duration < minimum or new_duration > maximum:
        return segment.clone(), {
            "left": {**left_detail, "accepted": False},
            "right": {**right_detail, "accepted": False},
        }
    refined = segment.new_tensor([left, right])
    return refined, {"left": left_detail, "right": right_detail}


def build_same_selection_supervision(
        pseudo_labels, gt_labels, pseudo_offsets, gt_offsets,
        valid_mask, enabled_levels, mode):
    """Keep the cleaned selection fixed and change only boundary targets."""
    if mode not in ("clean_pseudo", "same_selection"):
        raise ValueError("unknown same-selection mode: {}".format(mode))
    if pseudo_labels.shape != gt_labels.shape:
        raise ValueError("pseudo labels and GT labels must align")
    if pseudo_offsets.shape != gt_offsets.shape:
        raise ValueError("pseudo offsets and GT offsets must align")
    mask = (
        valid_mask.bool()
        & enabled_levels.bool()
        & (pseudo_labels.sum(dim=-1) > 0)
        & (gt_labels.sum(dim=-1) > 0)
    )
    targets = pseudo_offsets if mode == "clean_pseudo" else gt_offsets
    return mask, targets


def masked_side_mse(predictions, targets, left_mask, right_mask):
    """MSE over independently accepted left and right boundary targets."""
    if predictions.shape != targets.shape or predictions.shape[-1] != 2:
        raise ValueError("boundary offsets must align as [..., 2]")
    if left_mask.shape != predictions.shape[:-1] or right_mask.shape != left_mask.shape:
        raise ValueError("side masks must align with boundary offsets")
    side_mask = torch.stack((left_mask, right_mask), dim=-1).bool()
    if not side_mask.any():
        return predictions.sum() * 0.0
    return (predictions - targets.detach()).square()[side_mask].mean()
