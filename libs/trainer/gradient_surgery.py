import torch


def project_conflicting_gradients(reference_gradients, auxiliary_gradients):
    """Project an auxiliary gradient away from a conflicting reference one."""
    if len(reference_gradients) != len(auxiliary_gradients):
        raise ValueError('gradient lists must have equal length')

    shared = [
        (reference, auxiliary)
        for reference, auxiliary in zip(reference_gradients, auxiliary_gradients)
        if reference is not None and auxiliary is not None
    ]
    if not shared:
        device = next(
            (gradient.device for gradient in auxiliary_gradients
             if gradient is not None),
            torch.device('cpu'),
        )
        zero = torch.zeros((), device=device)
        return list(auxiliary_gradients), zero, zero, zero

    dot = sum((reference * auxiliary).sum() for reference, auxiliary in shared)
    reference_norm_sq = sum(reference.square().sum() for reference, _ in shared)
    auxiliary_norm_sq = sum(auxiliary.square().sum() for _, auxiliary in shared)
    cosine = dot / (reference_norm_sq * auxiliary_norm_sq).sqrt().clamp_min(1e-12)
    conflict = (dot < 0).to(dot.dtype)
    coefficient = torch.where(
        dot < 0,
        dot / reference_norm_sq.clamp_min(1e-12),
        torch.zeros_like(dot),
    )
    projected = [
        auxiliary - coefficient * reference
        if reference is not None and auxiliary is not None else auxiliary
        for reference, auxiliary in zip(reference_gradients, auxiliary_gradients)
    ]
    return projected, cosine, conflict, coefficient.abs()
