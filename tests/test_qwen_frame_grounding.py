import pytest

from tools.run_qwen3_vl_frame_grounding import (
    digit_to_frame,
    grounding_audit,
    normalize_frame_logits,
)


def test_frame_mapping_is_deterministic_and_complete():
    first = digit_to_frame("candidate-1", 8)
    second = digit_to_frame("candidate-1", 8)

    assert first == second
    assert sorted(first) == list(range(8))


def test_frame_logits_are_returned_in_chronological_order():
    result = normalize_frame_logits([5.0, 0.0], mapping=(1, 0))

    assert result["peak_frame_index"] == 1
    assert result["frame_probabilities"][1] > result["frame_probabilities"][0]


def test_grounding_audit_compares_peak_with_fixed_midpoint():
    rows = [{
        "video_id": "video",
        "prediction": "TAKE",
        "proposal_start_sec": 0.0,
        "proposal_end_sec": 8.0,
        "frame_count": 8,
        "peak_frame_index": 6,
        "gt_assigned_class_ids": [0],
    }]
    database = {"video": {"annotations": [{
        "label_id": 0,
        "segment": [6.0, 7.0],
    }]}}

    result = grounding_audit(rows, database)

    assert result["peak_hit_rate"] == pytest.approx(1.0)
    assert result["midpoint_hit_rate"] == pytest.approx(0.0)
    assert result["peak_gain_vs_midpoint"] == pytest.approx(1.0)
    assert result["class_correct"]["peak_hit_rate"] == pytest.approx(1.0)
