import torch
import torch.nn.functional as F


@torch.no_grad()
def target_routing_gt_counts(probabilities, gt_labels, alpha):
    """Count how the current target routing treats GT action anchors."""
    if probabilities.shape != gt_labels.shape:
        raise ValueError("probabilities and GT labels must have the same shape")

    score, pseudo_class = probabilities.max(dim=-1)
    selected = score >= alpha
    gt_action = gt_labels.sum(dim=-1) > 0
    class_correct = gt_labels.gather(
        -1, pseudo_class.unsqueeze(-1)
    ).squeeze(-1) > 0

    return {
        "valid": torch.tensor(
            selected.numel(), device=probabilities.device, dtype=torch.float32
        ),
        "selected": selected.sum().float(),
        "gt_action": gt_action.sum().float(),
        "selected_gt_action": (selected & gt_action).sum().float(),
        "selected_correct_action": (selected & class_correct).sum().float(),
        "ignored_gt_action": ((~selected) & gt_action).sum().float(),
        "selected_gt_background": (selected & (~gt_action)).sum().float(),
    }


def build_oracle_routing(probabilities, gt_labels, alpha, mode):
    """Build diagnostic target routes without changing the default SADA path."""
    if mode not in ("o1", "o2", "o3"):
        raise ValueError("unknown oracle routing mode: {}".format(mode))
    if probabilities.shape != gt_labels.shape:
        raise ValueError("probabilities and GT labels must have the same shape")

    score, pseudo_class = probabilities.max(dim=-1)
    high_confidence = score >= alpha
    gt_foreground = gt_labels.sum(dim=-1) > 0
    pseudo_labels = F.one_hot(
        pseudo_class, num_classes=probabilities.shape[-1]
    ).to(probabilities.dtype)
    action_labels = torch.zeros_like(probabilities).detach()

    if mode == "o1":
        action = high_confidence & gt_foreground
        background = ~gt_foreground
        action_labels[action] = pseudo_labels[action]
    elif mode == "o2":
        action = high_confidence
        background = ~high_confidence
        action_labels[action] = pseudo_labels[action]
        true_action = action & gt_foreground
        action_labels[true_action] = gt_labels[true_action]
    else:
        action = gt_foreground
        background = ~gt_foreground
        action_labels[action] = gt_labels[action]

    ignore = ~(action | background)
    return {
        "action_labels": action_labels,
        "action": action,
        "background": background,
        "ignore": ignore,
        "gt_foreground": gt_foreground,
        "high_confidence": high_confidence,
        "pseudo_class": pseudo_class,
        "score": score,
    }


def build_oracle_target_regression_mask(
    max_scores, gt_labels, valid_mask, enabled_levels, alpha, mode
):
    """Select target anchors for the explicit Oracle boundary diagnostic.

    ``selected`` keeps the original SADA confidence candidate set and only
    reveals its true foreground members. ``full`` reveals every foreground
    anchor in the requested levels. Both modes are diagnostics, never UDA.
    """
    if mode not in ("selected", "full"):
        raise ValueError("unknown Oracle target regression mode: {}".format(mode))
    if (
        max_scores.shape != valid_mask.shape
        or max_scores.shape != enabled_levels.shape
        or gt_labels.shape[:-1] != max_scores.shape
    ):
        raise ValueError(
            "Oracle target regression inputs have incompatible shapes: "
            "scores={}, labels={}, valid={}, levels={}".format(
                tuple(max_scores.shape),
                tuple(gt_labels.shape),
                tuple(valid_mask.shape),
                tuple(enabled_levels.shape),
            )
        )

    foreground = gt_labels.sum(dim=-1) > 0
    eligible = valid_mask & enabled_levels & foreground
    if mode == "selected":
        eligible = eligible & (max_scores >= alpha)
    return eligible


def validate_stage1_resume(
    checkpoint,
    current_enabled,
    current_perturbation,
    current_alpha,
    calibration_epoch,
):
    """Reject checkpoint resumes which would silently change Stage 1 mode."""
    stage1 = checkpoint.get("stage1")
    checkpoint_config = checkpoint.get("config", {})
    if hasattr(checkpoint_config, "config"):
        checkpoint_config = checkpoint_config.config
    checkpoint_stage1_cfg = checkpoint_config.get("stage1", {})
    checkpoint_model_cfg = checkpoint_config.get("model", {})

    if stage1 is None:
        if current_enabled and checkpoint.get("epoch", 0) >= calibration_epoch:
            raise ValueError(
                "Stage 1 is enabled, but this checkpoint has no Stage 1 state "
                "and has already reached the calibration epoch"
            )
        return None

    if not current_enabled:
        raise ValueError(
            "checkpoint contains Stage 1 state, but current run uses "
            "--stage1 off; pass the matching --stage1 mode"
        )

    saved_enabled = stage1.get(
        "enabled", checkpoint_stage1_cfg.get("enabled", True)
    )
    if not saved_enabled:
        raise ValueError(
            "checkpoint Stage 1 state is marked disabled and cannot be resumed"
        )

    saved_perturbation = stage1.get(
        "perturbation", checkpoint_stage1_cfg.get("perturbation")
    )
    if saved_perturbation is None:
        raise ValueError(
            "cannot determine the Stage 1 perturbation mode from checkpoint"
        )
    if saved_perturbation != current_perturbation:
        raise ValueError(
            "Stage 1 perturbation mismatch: checkpoint={} current={}".format(
                saved_perturbation, current_perturbation
            )
        )

    saved_alpha = stage1.get(
        "alpha", checkpoint_model_cfg.get("pl_threshold")
    )
    if saved_alpha is None:
        raise ValueError("cannot determine Stage 1 alpha from checkpoint")
    if not torch.isclose(
        torch.as_tensor(float(saved_alpha)),
        torch.as_tensor(float(current_alpha)),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Stage 1 alpha mismatch: checkpoint={} current={}".format(
                saved_alpha, current_alpha
            )
        )

    if (
        checkpoint.get("epoch", 0) >= calibration_epoch
        and not stage1.get("calibrated", False)
    ):
        raise ValueError(
            "checkpoint passed the Stage 1 calibration epoch but contains "
            "uncalibrated Stage 1 state"
        )
    return stage1


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


def context_view_on_cpu(original, full, start, length):
    """Build one Context view on CPU before batch padding/concatenation."""
    if full is not None:
        context = full[:, start:start + length]
    else:
        context = original
    return context.detach().cpu()


def interpolate_probabilities(probabilities, strict_mask, points, query):
    """
    Interpolate BxTxC probabilities on a shared 1-D point grid.

    Returns interpolated probabilities and a mask which is true only when both
    interpolation endpoints are strict-valid.
    """
    grid = points[:, 0]
    insertion = torch.searchsorted(
        grid.contiguous(), query.contiguous(), right=False
    )
    exact_index = insertion.clamp_max(grid.numel() - 1)
    exact = (
        (insertion < grid.numel())
        & torch.isclose(
            grid[exact_index], query, rtol=0.0, atol=1e-6
        )
    )

    right = insertion.clamp(1, grid.numel() - 1)
    left = right - 1

    left_t = grid[left]
    right_t = grid[right]
    weight = ((query - left_t) / (right_t - left_t).clamp_min(1e-12))
    left_prob = probabilities[:, left, :]
    right_prob = probabilities[:, right, :]
    interpolated = left_prob + weight[None, :, None] * (
        right_prob - left_prob
    )
    exact_prob = probabilities[:, exact_index, :]
    output = torch.where(exact[None, :, None], exact_prob, interpolated)

    inside = (query >= grid[0]) & (query <= grid[-1])
    interpolated_valid = strict_mask[:, left] & strict_mask[:, right]
    exact_valid = strict_mask[:, exact_index]
    valid = torch.where(
        exact[None, :], exact_valid, interpolated_valid
    ) & inside[None, :]
    return output, valid


def source_action_eligible(
    labels,
    pseudo_class,
    perturbed_class,
    score_original,
    valid,
    alpha,
):
    """Check source action correctness without collapsing multi-hot labels."""
    predicted_is_positive = labels.gather(
        -1, pseudo_class.unsqueeze(-1)
    ).squeeze(-1) > 0
    return (
        valid
        & (labels.sum(dim=-1) > 0)
        & predicted_is_positive
        & (perturbed_class == pseudo_class)
        & (score_original >= alpha)
    )


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
