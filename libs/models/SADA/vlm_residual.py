import torch


def select_supported_residual(
    candidate_mask,
    base_target_mask,
    source_samples,
    level,
    residual_levels,
):
    """Accept rescue anchors only when source and ordinary target support exist."""
    if (
        int(level) not in set(residual_levels)
        or int(source_samples) <= 0
        or not bool(base_target_mask.any())
    ):
        return torch.zeros_like(candidate_mask)
    return candidate_mask


def bounded_residual_loss(base_losses, residual_losses, budget_fraction):
    """Add a detached-budget residual without letting it dominate the base loss."""
    if not 0.0 < float(budget_fraction) <= 1.0:
        raise ValueError("residual budget fraction must be in (0, 1]")

    zero = base_losses.sum() * 0.0 + residual_losses.sum() * 0.0
    if base_losses.numel() == 0:
        return zero, zero, zero

    base_loss = base_losses.mean()
    if residual_losses.numel() == 0:
        return base_loss, zero, zero

    raw_residual = residual_losses.mean()
    budget = float(budget_fraction) * base_loss.detach()
    scale = torch.clamp(
        budget / raw_residual.detach().clamp_min(torch.finfo(raw_residual.dtype).eps),
        max=1.0,
    )
    residual = scale * raw_residual
    budget_excess = torch.clamp(residual.detach() - budget, min=0.0)
    return base_loss + residual, residual, budget_excess


def bounded_extra_loss(reference_losses, extra_losses, budget_fraction):
    """Scale an auxiliary loss by a detached budget from trusted losses."""
    if not 0.0 < float(budget_fraction) <= 1.0:
        raise ValueError("extra budget fraction must be in (0, 1]")

    zero = reference_losses.sum() * 0.0 + extra_losses.sum() * 0.0
    if reference_losses.numel() == 0 or extra_losses.numel() == 0:
        return zero, zero

    raw_extra = extra_losses.mean()
    reference = reference_losses.mean().detach()
    budget = float(budget_fraction) * reference
    scale = torch.clamp(
        budget / raw_extra.detach().clamp_min(torch.finfo(raw_extra.dtype).eps),
        max=1.0,
    )
    extra = scale * raw_extra
    budget_excess = torch.clamp(extra.detach() - budget, min=0.0)
    return extra, budget_excess
