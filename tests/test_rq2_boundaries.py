import numpy as np

from tools.build_rq2_boundary_manifest import context_ratios, overlaps
from tools.compare_rq1_frame_counts import paired_bootstrap
from tools.evaluate_rq2_boundaries import position_prior, prediction_interval, temporal_iou
from tools.run_rq2_clip import localize_peak
from tools.run_rq2_qwen import parse_interval
from tools.run_rq2_slowfast import timestamp_features


def test_context_ratios_are_deterministic_independent_and_in_range():
    first = context_ratios("candidate-a")
    second = context_ratios("candidate-a")
    assert first == second
    assert all(0.25 <= value <= 2.0 for value in first)
    assert first[0] != first[1]


def test_overlap_uses_positive_duration_only():
    assert overlaps((1.0, 3.0), (2.0, 4.0))
    assert not overlaps((1.0, 2.0), (2.0, 3.0))


def test_qwen_interval_parser_requires_ordered_in_range_gaps():
    assert parse_interval("4,12") == (4, 12)
    assert parse_interval("answer: 2; 16") == (2, 16)
    assert parse_interval("12,4") is None
    assert parse_interval("0,17") is None


def test_clip_peak_localizer_returns_component_around_global_peak():
    start, end, threshold = localize_peak([0.0, 1.0, 3.0, 2.0, 0.0])
    assert (start, end) == (2, 4)
    assert threshold == 1.5


def test_boundary_positions_convert_to_seconds_and_tiou():
    row = {
        "frame_count": 16,
        "proposal_start_sec": 10.0,
        "proposal_end_sec": 26.0,
    }
    interval = prediction_interval(
        row, {"pred_start_position": 4, "pred_end_position": 12}
    )
    assert interval == (14.0, 22.0)
    assert temporal_iou(interval, (16.0, 24.0)) == 0.6


def test_position_prior_uses_registered_context_range():
    row = {
        "frame_count": 16,
        "context_ratio_min": 0.25,
        "context_ratio_max": 2.0,
    }
    assert position_prior(row) == {
        "status": "ok",
        "pred_start_position": 16 * 1.125 / 3.25,
        "pred_end_position": 16 * 2.125 / 3.25,
    }


def test_slowfast_timestamp_sampling_uses_nearest_feature_center():
    features = np.eye(3, dtype=np.float32)
    sampled = timestamp_features(
        features, [0.0, 1.0, 2.0], fps=1.0, feat_stride=1.0, num_frames=0.0
    )
    assert np.allclose(sampled, features)


def test_paired_video_bootstrap_detects_uniform_improvement():
    rows = [
        {"candidate_id": "a", "video_id": "v1", "gt_label_id": 0},
        {"candidate_id": "b", "video_id": "v2", "gt_label_id": 1},
    ]
    first = {
        ("a", None): {"class_id": 1},
        ("b", None): {"class_id": 0},
    }
    second = {
        ("a", None): {"class_id": 0},
        ("b", None): {"class_id": 1},
    }
    result = paired_bootstrap(rows, first, second, "S1", repeats=20, seed=0)
    assert np.allclose(result["accuracy_delta_95ci"], [1.0, 1.0])
