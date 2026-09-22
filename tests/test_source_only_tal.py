from libs.trainer.training_regimes import (
    build_training_generators,
    consumes_target_train,
    training_epoch_length,
)
import torch


def test_source_only_uses_only_source_epoch_length():
    assert training_epoch_length("source_only", 4, 7) == 4
    assert training_epoch_length("source_only", 11, 3) == 11
    assert not consumes_target_train("source_only")


def test_other_regimes_keep_existing_epoch_length_rules():
    assert training_epoch_length("target_only", 4, 7) == 7
    assert training_epoch_length("off", 4, 7) == 7
    assert training_epoch_length("source_target", 9, 7) == 9
    assert consumes_target_train("off")


def test_formal_so_and_uda_use_the_same_source_step_budget():
    assert training_epoch_length("source_only", 25, 53, "so") == 25
    assert training_epoch_length("off", 25, 53, "uda") == 25
    assert training_epoch_length("off", 62, 110, "uda") == 62


def test_target_iterator_does_not_change_source_order():
    source_a, target_a = build_training_generators(6)
    source_b, _ = build_training_generators(6)
    torch.randperm(100, generator=target_a)
    assert torch.equal(
        torch.randperm(100, generator=source_a),
        torch.randperm(100, generator=source_b),
    )
