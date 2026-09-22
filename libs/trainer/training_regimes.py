"""Small helpers defining which data a controlled training regime may use."""

import torch


RQ2_TARGET_SEED_OFFSET = 1_000_003


def consumes_target_train(supervised_mode):
    return supervised_mode != "source_only"


def training_epoch_length(
    supervised_mode, source_batches, target_batches, formal_regime=None
):
    if formal_regime in ("so", "uda"):
        return source_batches
    if supervised_mode == "source_only":
        return source_batches
    if supervised_mode == "target_only":
        return target_batches
    return max(source_batches, target_batches)


def build_training_generators(seed):
    source = torch.Generator(device="cpu")
    target = torch.Generator(device="cpu")
    source.manual_seed(int(seed))
    target.manual_seed(int(seed) + RQ2_TARGET_SEED_OFFSET)
    return source, target
