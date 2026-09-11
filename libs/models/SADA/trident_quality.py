"""Localization-quality estimates derived from Trident boundary distributions."""

import torch


def boundary_peak_quality(center_logits, start_neighbours, end_neighbours):
    """Return geometric-mean start/end peak probability for every class and point."""
    center = center_logits.view(center_logits.shape[0], 2, -1)
    start = torch.softmax(start_neighbours + center[None, :, 0, :], dim=-1)
    end = torch.softmax(end_neighbours + center[None, :, 1, :], dim=-1)
    quality = (start.amax(dim=-1) * end.amax(dim=-1)).clamp_min(0).sqrt()
    return torch.nan_to_num(quality, nan=0.0, posinf=0.0, neginf=0.0)


def quality_weighted_scores(scores, qualities, gamma):
    if gamma < 0.0:
        raise ValueError("gamma must be non-negative")
    return scores * qualities.clamp_min(1e-8).pow(float(gamma))
