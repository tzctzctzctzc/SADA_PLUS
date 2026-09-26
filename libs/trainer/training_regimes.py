"""Small helpers defining which data a controlled training regime may use."""

import torch


RQ2_TARGET_SEED_OFFSET = 1_000_003


def consumes_target_train(supervised_mode):
    return supervised_mode != "source_only"


def training_epoch_length(
    supervised_mode, source_batches, target_batches, formal_regime=None
):
    if formal_regime in ("so", "uda"):
        return max(source_batches, target_batches)
    if supervised_mode == "source_only":
        return source_batches
    if supervised_mode == "target_only":
        return target_batches
    return max(source_batches, target_batches)


def epoch_length_from_config(cfg, source_batches, target_batches):
    mode = cfg.get("diagnostic", {}).get("supervised_upper_bound", "off")
    formal = cfg.get("rq2_formal", {})
    regime = formal.get("regime")
    if regime in ("so", "uda"):
        budget = formal["epoch_budget"]
        if source_batches != budget["source_batches"]:
            raise ValueError("source loader differs from the frozen epoch budget")
        if regime == "uda" and target_batches != budget["target_batches"]:
            raise ValueError("target loader differs from the frozen epoch budget")
        # SO aliases its target loader to source. Only the frozen video count,
        # not target features/labels, supplies its matched update budget.
        target_batches = budget["target_batches"]
    return training_epoch_length(mode, source_batches, target_batches, regime)


def uses_alignment_optimizer(supervised_mode, epoch, warmup_adv, formal_regime=None):
    matched_source = supervised_mode == "source_only" and formal_regime == "so"
    return epoch >= warmup_adv and (supervised_mode == "off" or matched_source)


def build_training_generators(seed):
    source = torch.Generator(device="cpu")
    target = torch.Generator(device="cpu")
    source.manual_seed(int(seed))
    target.manual_seed(int(seed) + RQ2_TARGET_SEED_OFFSET)
    return source, target
