"""Small helpers defining which data a controlled training regime may use."""


def consumes_target_train(supervised_mode):
    return supervised_mode != "source_only"


def training_epoch_length(supervised_mode, source_batches, target_batches):
    if supervised_mode == "source_only":
        return source_batches
    if supervised_mode == "target_only":
        return target_batches
    return max(source_batches, target_batches)
