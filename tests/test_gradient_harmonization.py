import torch

from libs.trainer.gradient_harmonization import (
    blockwise_task_protected_gradients,
    task_protected_gradients,
)


def test_task_protection_projects_conflicting_alignment_gradient():
    task = [torch.tensor([1.0, 0.0])]
    alignment = [torch.tensor([-2.0, 1.0])]

    combined, stats = task_protected_gradients(task, alignment)

    projected_alignment = combined[0] - task[0]
    assert torch.allclose(projected_alignment, torch.tensor([0.0, 1.0]))
    assert torch.dot(task[0], projected_alignment).item() == 0.0
    assert stats['conflict'].item() == 1.0
    assert stats['cosine'].item() < 0.0


def test_task_protection_keeps_agreeing_gradients_unchanged():
    task = [torch.tensor([1.0, 0.0])]
    alignment = [torch.tensor([2.0, 1.0])]

    combined, stats = task_protected_gradients(task, alignment)

    assert torch.equal(combined[0], task[0] + alignment[0])
    assert stats['conflict'].item() == 0.0


def test_task_protection_preserves_single_loss_parameters():
    task = [torch.tensor([1.0]), None]
    alignment = [None, torch.tensor([2.0])]

    combined, stats = task_protected_gradients(task, alignment)

    assert torch.equal(combined[0], task[0])
    assert torch.equal(combined[1], alignment[1])
    assert stats['paired_parameters'].item() == 0.0


def test_blockwise_protection_prevents_cross_block_conflict_cancellation():
    task = [torch.tensor([1.0]), torch.tensor([1.0])]
    alignment = [torch.tensor([-2.0]), torch.tensor([2.0])]

    global_combined, _ = task_protected_gradients(task, alignment)
    block_combined, stats = blockwise_task_protected_gradients(
        task, alignment, ['early', 'late']
    )

    assert torch.equal(global_combined[0], torch.tensor([-1.0]))
    assert torch.equal(block_combined[0], torch.tensor([1.0]))
    assert torch.equal(block_combined[1], torch.tensor([3.0]))
    assert stats['block_conflict_fraction'].item() == 0.5
    assert stats['paired_blocks'].item() == 2.0
