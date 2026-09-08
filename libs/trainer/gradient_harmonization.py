import torch


def task_protected_gradients(task_gradients, alignment_gradients, eps=1e-12):
    """Remove only the alignment component that opposes the source task.

    The projection is global across the shared feature extractor. Parameters
    reached by only one loss keep their original gradient.
    """
    if len(task_gradients) != len(alignment_gradients):
        raise ValueError("gradient lists must have the same length")

    paired = [
        (task, alignment)
        for task, alignment in zip(task_gradients, alignment_gradients)
        if task is not None and alignment is not None
    ]
    if not paired:
        zero = torch.tensor(0.0)
        return [
            task if alignment is None else alignment if task is None else task + alignment
            for task, alignment in zip(task_gradients, alignment_gradients)
        ], {
            "task_norm": zero,
            "alignment_norm": zero,
            "projected_alignment_norm": zero,
            "cosine": zero,
            "conflict": zero,
            "paired_parameters": zero,
        }

    device = paired[0][0].device
    task_norm_sq = sum((task.float() ** 2).sum() for task, _ in paired)
    alignment_norm_sq = sum(
        (alignment.float() ** 2).sum() for _, alignment in paired
    )
    dot = sum(
        (task.float() * alignment.float()).sum()
        for task, alignment in paired
    )
    denominator = (task_norm_sq * alignment_norm_sq).sqrt().clamp_min(eps)
    cosine = dot / denominator
    conflict = bool(dot.detach().item() < 0.0 and task_norm_sq.detach().item() > eps)
    coefficient = dot / task_norm_sq.clamp_min(eps) if conflict else dot * 0.0

    combined = []
    projected_alignment_norm_sq = task_norm_sq * 0.0
    for task, alignment in zip(task_gradients, alignment_gradients):
        if task is None:
            combined.append(alignment)
            continue
        if alignment is None:
            combined.append(task)
            continue
        projected_alignment = alignment - coefficient.to(alignment.dtype) * task
        combined.append(task + projected_alignment)
        projected_alignment_norm_sq = (
            projected_alignment_norm_sq
            + (projected_alignment.float() ** 2).sum()
        )

    return combined, {
        "task_norm": task_norm_sq.sqrt().detach().to(device),
        "alignment_norm": alignment_norm_sq.sqrt().detach().to(device),
        "projected_alignment_norm": projected_alignment_norm_sq.sqrt().detach().to(device),
        "cosine": cosine.detach().to(device),
        "conflict": torch.tensor(float(conflict), device=device),
        "paired_parameters": torch.tensor(float(len(paired)), device=device),
    }
