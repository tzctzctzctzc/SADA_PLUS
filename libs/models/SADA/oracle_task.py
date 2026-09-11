ORACLE_TARGET_TASK_MODES = ("off", "class", "boundary", "joint")


def compose_oracle_target_task_loss(
    source_task_loss,
    target_class_loss,
    target_boundary_loss,
    boundary_weight,
    mode,
):
    """Compose an explicitly target-supervised diagnostic loss.

    This helper is intentionally small: target GT may enter only through the
    selected task component.  The resulting run is an Oracle diagnostic, not
    an unsupervised domain-adaptation method.
    """
    if mode not in ORACLE_TARGET_TASK_MODES:
        raise ValueError("unknown Oracle target task mode: {}".format(mode))

    total = source_task_loss
    if mode in ("class", "joint"):
        total = total + target_class_loss
    if mode in ("boundary", "joint"):
        total = total + boundary_weight * target_boundary_loss
    return total
