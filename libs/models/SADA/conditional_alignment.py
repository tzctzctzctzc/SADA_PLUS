def aggregate_conditional_class_losses(rows, zero, paired_support=False):
    """Aggregate class-wise domain losses with optional same-batch support."""
    zero = zero * 0.0
    source_rows = [row for row in rows if row["source_samples"] > 0]
    target_rows = [row for row in rows if row["target_samples"] > 0]
    paired_rows = [
        row
        for row in rows
        if row["source_samples"] > 0 and row["target_samples"] > 0
    ]

    active_source_rows = paired_rows if paired_support else source_rows
    active_target_rows = paired_rows if paired_support else target_rows

    def mean_loss(active_rows, key):
        if not active_rows:
            return zero
        return sum(
            row["class_weight"] * row[key] for row in active_rows
        ) / len(active_rows)

    forgotten_target_rows = [
        row
        for row in target_rows
        if row["source_samples"] == 0
    ]
    original_target_denominator = max(len(target_rows), 1)
    forgotten_target_objective = sum(
        (
            row["class_weight"]
            * row["target_loss"]
            / original_target_denominator
        )
        for row in forgotten_target_rows
    ) if forgotten_target_rows else zero

    return {
        "source_loss": mean_loss(active_source_rows, "source_loss"),
        "target_loss": mean_loss(active_target_rows, "target_loss"),
        "source_classes": len(active_source_rows),
        "target_classes": len(active_target_rows),
        "source_present_classes": len(source_rows),
        "target_present_classes": len(target_rows),
        "paired_classes": len(paired_rows),
        "forgotten_target_classes": len(forgotten_target_rows),
        "forgotten_target_samples": sum(
            row["target_samples"] for row in forgotten_target_rows
        ),
        "forgotten_target_objective": forgotten_target_objective,
    }
