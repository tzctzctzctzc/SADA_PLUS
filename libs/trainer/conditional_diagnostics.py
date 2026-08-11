import re


_CLASS_METRIC = re.compile(
    r"^conditional_class_l(?P<level>\d+)_c(?P<class_id>\d+)_(?P<metric>.+)$"
)


def build_conditional_class_report(accumulator, batch_count, epoch):
    """Convert per-batch class metrics into one compact epoch report."""
    if batch_count <= 0:
        raise ValueError("conditional class diagnostics require at least one batch")

    grouped = {}
    for key, value in accumulator.items():
        match = _CLASS_METRIC.match(key)
        if match is None:
            continue
        row_key = (int(match.group("level")), int(match.group("class_id")))
        grouped.setdefault(row_key, {})[match.group("metric")] = float(value)

    rows = []
    for (level, class_id), metrics in sorted(grouped.items()):
        source_samples = metrics.get("source_samples", 0.0)
        target_samples = metrics.get("target_samples", 0.0)
        rescue_samples = metrics.get("vlm_rescue_samples", 0.0)
        unpaired_target = metrics.get("target_samples_without_source", 0.0)
        unpaired_rescue = metrics.get(
            "vlm_rescue_samples_without_source", 0.0
        )
        source_present = metrics.get("source_present", 0.0)
        target_present = metrics.get("target_present", 0.0)
        target_loss = metrics.get("target_loss", 0.0)
        target_objective = metrics.get("target_objective", 0.0)
        unpaired_objective = metrics.get(
            "target_objective_without_source", 0.0
        )
        rows.append({
            "level": level,
            "class_id": class_id,
            "source_samples_total": source_samples,
            "target_samples_total": target_samples,
            "vlm_rescue_samples_total": rescue_samples,
            "target_samples_without_source_total": unpaired_target,
            "vlm_rescue_samples_without_source_total": unpaired_rescue,
            "source_present_rate": source_present / batch_count,
            "target_present_rate": target_present / batch_count,
            "target_loss_mean_when_present": (
                target_loss / target_present if target_present else 0.0
            ),
            "target_objective_mean_per_batch": target_objective / batch_count,
            "target_objective_without_source_mean_per_batch": (
                unpaired_objective / batch_count
            ),
            "target_without_source_rate": (
                unpaired_target / target_samples if target_samples else 0.0
            ),
            "vlm_rescue_without_source_rate": (
                unpaired_rescue / rescue_samples if rescue_samples else 0.0
            ),
        })

    target_total = sum(row["target_samples_total"] for row in rows)
    unpaired_target_total = sum(
        row["target_samples_without_source_total"] for row in rows
    )
    rescue_total = sum(row["vlm_rescue_samples_total"] for row in rows)
    unpaired_rescue_total = sum(
        row["vlm_rescue_samples_without_source_total"] for row in rows
    )
    target_objective_total = sum(
        row["target_objective_mean_per_batch"] for row in rows
    )
    unpaired_objective_total = sum(
        row["target_objective_without_source_mean_per_batch"] for row in rows
    )
    summary = {
        "target_samples_total": target_total,
        "target_samples_without_source_total": unpaired_target_total,
        "target_without_source_rate": (
            unpaired_target_total / target_total if target_total else 0.0
        ),
        "vlm_rescue_samples_total": rescue_total,
        "vlm_rescue_samples_without_source_total": unpaired_rescue_total,
        "vlm_rescue_without_source_rate": (
            unpaired_rescue_total / rescue_total if rescue_total else 0.0
        ),
        "target_objective_mean_per_batch": target_objective_total,
        "target_objective_without_source_mean_per_batch": unpaired_objective_total,
        "target_objective_without_source_fraction": (
            unpaired_objective_total / target_objective_total
            if target_objective_total else 0.0
        ),
    }
    return {
        "schema_version": "conditional-class-support-diagnostic-v1",
        "epoch": int(epoch),
        "batch_count": int(batch_count),
        "summary": summary,
        "rows": rows,
    }
