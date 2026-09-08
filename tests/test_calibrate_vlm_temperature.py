import pytest

from tools.calibrate_vlm_temperature import (
    calibration_metrics,
    calibration_rows,
    fit_temperature,
    row_class_logits,
    temperature_scaled_probabilities,
)


def test_temperature_scaled_probabilities_preserve_argmax_and_sum():
    logits = [8.0, 2.0] + [0.0] * 8
    probabilities = temperature_scaled_probabilities(logits, 4.0)

    assert sum(probabilities) == pytest.approx(1.0)
    assert max(range(10), key=probabilities.__getitem__) == 0
    assert probabilities[0] < temperature_scaled_probabilities(logits, 1.0)[0]


def test_fit_temperature_reduces_nll_for_overconfident_predictions():
    rows = [
        {"class_logits": [12.0, 0.0] + [-2.0] * 8, "labels": [0]},
        {"class_logits": [12.0, 0.0] + [-2.0] * 8, "labels": [1]},
        {"class_logits": [12.0, 0.0] + [-2.0] * 8, "labels": [1]},
    ]

    temperature = fit_temperature(rows)

    assert temperature > 1.0
    assert calibration_metrics(rows, temperature)["nll"] < calibration_metrics(rows, 1.0)["nll"]


def test_calibration_rows_use_only_known_source_actions():
    rows = [
        {
            "status": "ok",
            "gt_assigned_class_ids": [2],
            "class_logits": [0.0] * 10,
        },
        {
            "status": "ok",
            "gt_assigned_class_ids": [],
            "class_logits": [0.0] * 10,
        },
    ]

    selected = calibration_rows(rows)

    assert selected == [{"class_logits": [0.0] * 10, "labels": [2]}]


def test_clip_probabilities_can_be_used_as_temperature_logits():
    probabilities = [0.55, 0.45] + [0.0] * 8

    logits = row_class_logits({"class_probabilities": probabilities})

    assert max(range(10), key=logits.__getitem__) == 0
    assert temperature_scaled_probabilities(logits, 1.0) == pytest.approx(
        probabilities, abs=1e-10
    )
