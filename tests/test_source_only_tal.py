from libs.trainer.training_regimes import (
    consumes_target_train,
    training_epoch_length,
)


def test_source_only_uses_only_source_epoch_length():
    assert training_epoch_length("source_only", 4, 7) == 4
    assert training_epoch_length("source_only", 11, 3) == 11
    assert not consumes_target_train("source_only")


def test_other_regimes_keep_existing_epoch_length_rules():
    assert training_epoch_length("target_only", 4, 7) == 7
    assert training_epoch_length("off", 4, 7) == 7
    assert training_epoch_length("source_target", 9, 7) == 9
    assert consumes_target_train("off")
