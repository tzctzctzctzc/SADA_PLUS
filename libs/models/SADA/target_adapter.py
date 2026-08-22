import torch


ORACLE_TARGET_DISTILLATION_MODES = ("off", "selected", "full")


def build_oracle_target_distillation(
    scores,
    gt_labels,
    valid_mask,
    enabled_levels,
    alpha,
    mode,
):
    """Select target anchors for diagnostic direct task supervision."""
    if mode not in ORACLE_TARGET_DISTILLATION_MODES[1:]:
        raise ValueError(
            "unknown Oracle target distillation mode: {}".format(mode)
        )
    if (
        scores.shape != valid_mask.shape
        or scores.shape != enabled_levels.shape
        or gt_labels.shape[:-1] != scores.shape
    ):
        raise ValueError(
            "Oracle target distillation inputs have incompatible shapes"
        )

    selected = valid_mask & enabled_levels
    if mode == "selected":
        selected = selected & (scores >= float(alpha))

    foreground = selected & (gt_labels.sum(dim=-1) > 0)
    background = selected & ~foreground
    return {
        "selected": selected,
        "foreground": foreground,
        "background": background,
        "targets": gt_labels.detach().clone(),
    }
