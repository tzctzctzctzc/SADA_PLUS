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


def blockwise_task_protected_gradients(
    task_gradients, alignment_gradients, block_ids, eps=1e-12
):
    """Apply task protection independently inside each network block.

    Global projection can hide local conflicts when agreeing and opposing
    blocks cancel. This variant preserves the same one-sided rule while
    preventing cancellation across backbone/FPN blocks.
    """
    if not (
        len(task_gradients) == len(alignment_gradients) == len(block_ids)
    ):
        raise ValueError("gradient and block-id lists must have the same length")

    indices_by_block = {}
    for index, block_id in enumerate(block_ids):
        indices_by_block.setdefault(block_id, []).append(index)

    combined = [None] * len(task_gradients)
    block_stats = []
    for indices in indices_by_block.values():
        block_combined, stats = task_protected_gradients(
            [task_gradients[index] for index in indices],
            [alignment_gradients[index] for index in indices],
            eps=eps,
        )
        for index, gradient in zip(indices, block_combined):
            combined[index] = gradient
        if stats["paired_parameters"].item() > 0:
            block_stats.append(stats)

    paired = [
        (task, alignment)
        for task, alignment in zip(task_gradients, alignment_gradients)
        if task is not None and alignment is not None
    ]
    if not paired:
        zero = torch.tensor(0.0)
        return combined, {
            "task_norm": zero,
            "alignment_norm": zero,
            "projected_alignment_norm": zero,
            "cosine": zero,
            "conflict": zero,
            "paired_parameters": zero,
            "block_conflict_fraction": zero,
            "paired_blocks": zero,
        }

    device = paired[0][0].device
    task_norm_sq = sum((task.float() ** 2).sum() for task, _ in paired)
    alignment_norm_sq = sum(
        (alignment.float() ** 2).sum() for _, alignment in paired
    )
    dot = sum(
        (task.float() * alignment.float()).sum() for task, alignment in paired
    )
    projected_alignment_norm_sq = task_norm_sq * 0.0
    for task, gradient in zip(task_gradients, combined):
        if task is not None and gradient is not None:
            projected_alignment_norm_sq += ((gradient - task).float() ** 2).sum()

    denominator = (task_norm_sq * alignment_norm_sq).sqrt().clamp_min(eps)
    conflicts = sum(stats["conflict"] for stats in block_stats)
    paired_blocks = torch.tensor(float(len(block_stats)), device=device)
    return combined, {
        "task_norm": task_norm_sq.sqrt().detach().to(device),
        "alignment_norm": alignment_norm_sq.sqrt().detach().to(device),
        "projected_alignment_norm": projected_alignment_norm_sq.sqrt().detach().to(device),
        "cosine": (dot / denominator).detach().to(device),
        "conflict": torch.tensor(float(dot.detach().item() < 0.0), device=device),
        "paired_parameters": torch.tensor(float(len(paired)), device=device),
        "block_conflict_fraction": (conflicts / paired_blocks.clamp_min(1.0)).detach(),
        "paired_blocks": paired_blocks,
    }
