import torch
import torch.nn.functional as F


def population_std(values, dim=0):
    """Population standard deviation used by the two-view routing rule."""
    return torch.std(values, dim=dim, correction=0)


def shift_features(features, valid_mask, delta):
    """Translate CxT features without wrap-around."""
    shifted = torch.zeros_like(features)
    shifted_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    if delta > 0:
        shifted[..., delta:] = features[..., :-delta]
        shifted_mask[..., delta:] = valid_mask[..., :-delta]
    elif delta < 0:
        amount = -delta
        shifted[..., :-amount] = features[..., amount:]
        shifted_mask[..., :-amount] = valid_mask[..., amount:]
    else:
        shifted.copy_(features)
        shifted_mask.copy_(valid_mask)
    return shifted, shifted_mask


def interpolate_probabilities(probabilities, strict_mask, points, query):
    """
    Interpolate BxTxC probabilities on a shared 1-D point grid.

    Returns interpolated probabilities and a mask which is true only when both
    interpolation endpoints are strict-valid.
    """
    grid = points[:, 0]
    right = torch.searchsorted(grid.contiguous(), query.contiguous(), right=False)
    right = right.clamp(1, grid.numel() - 1)
    left = right - 1

    left_t = grid[left]
    right_t = grid[right]
    weight = ((query - left_t) / (right_t - left_t).clamp_min(1e-12))
    left_prob = probabilities[:, left, :]
    right_prob = probabilities[:, right, :]
    output = left_prob + weight[None, :, None] * (right_prob - left_prob)

    inside = (query >= grid[0]) & (query <= grid[-1])
    valid = strict_mask[:, left] & strict_mask[:, right] & inside[None, :]
    return output, valid


def erode_mask(mask, kernel_size, stride=1, padding=0):
    """Propagate strict local support through a temporal kernel."""
    pooled = F.avg_pool1d(
        mask.float(), kernel_size=kernel_size, stride=stride, padding=padding
    )
    return pooled >= (1.0 - 1e-6)


def build_routing_masks(
    original_prob,
    perturbed_prob,
    valid,
    alpha,
    beta,
    tau_act,
    tau_bg,
):
    """Create mutually exclusive reliable-action/background/ignore masks."""
    score0, cls0 = original_prob.max(dim=-1)
    score1, cls1 = perturbed_prob.max(dim=-1)

    cls1_for_original = perturbed_prob.gather(-1, cls0.unsqueeze(-1)).squeeze(-1)
    cls0_score = original_prob.gather(-1, cls0.unsqueeze(-1)).squeeze(-1)
    action_std = population_std(
        torch.stack((cls0_score, cls1_for_original), dim=0), dim=0
    )
    background_std = population_std(
        torch.stack((score0, score1), dim=0), dim=0
    )

    action_candidate = valid & (score0 >= alpha)
    class_consistent = action_candidate & (cls0 == cls1)
    background_candidate = valid & (score0 <= beta)
    all_view_low = background_candidate & (score1 <= beta)
    reliable_action = class_consistent & (action_std <= tau_act)
    stable_background = all_view_low & (background_std <= tau_bg)
    stable_background &= ~reliable_action
    ignore = ~(reliable_action | stable_background)
    return {
        "reliable_action": reliable_action,
        "stable_background": stable_background,
        "ignore": ignore,
        "action_candidate": action_candidate,
        "class_consistent": class_consistent,
        "background_candidate": background_candidate,
        "all_view_low": all_view_low,
        "pseudo_class": cls0,
        "perturbed_class": cls1,
        "valid": valid,
        "action_std": action_std,
        "background_std": background_std,
        "score_original": score0,
        "score_perturbed": score1,
    }


def weighted_quantile(values, weights, quantile):
    """Deterministic weighted quantile for source-only calibration."""
    if values.numel() == 0:
        raise ValueError("cannot calibrate a quantile from an empty tensor")
    order = torch.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = torch.cumsum(weights, dim=0)
    cutoff = quantile * cumulative[-1]
    index = torch.searchsorted(cumulative, cutoff).clamp_max(values.numel() - 1)
    return values[index]
