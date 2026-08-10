import torch

from tools.vlm_boundary_utils import (
    centered_candidates,
    grid_to_seconds,
    hard_class_agnostic_nms_indices,
    transition_pair_grid,
)


def test_hard_nms_preserves_exact_origin_and_deterministic_ties():
    segments = torch.tensor([[0.0, 2.0], [0.1, 2.1], [3.0, 4.0], [5.0, 6.0]])
    scores = torch.tensor([0.9, 0.9, 0.8, 0.7])
    kept = hard_class_agnostic_nms_indices(segments, scores, 0.5, 3)
    assert kept.tolist() == [0, 2, 3]


def test_grid_coordinate_conversion_matches_sada_postprocessing():
    assert grid_to_seconds(10, feat_stride=16, num_frames=32, fps=30) == 176 / 30


def test_centered_candidates_clip_and_deduplicate_video_edges():
    assert centered_candidates(0.1, 0.2, 7, 1.0) == [0.0, 0.1, 0.3, 0.5, 0.7]
    assert centered_candidates(0.9, 0.2, 7, 1.0) == [0.3, 0.5, 0.7, 0.9, 1.0]


def test_transition_pair_grid_uses_short_real_frame_pairs_and_deduplicates():
    pairs = transition_pair_grid(1.0, 0.2, 7, 2.0, 30.0, gap_frames=2)
    assert len(pairs) == 7
    assert len({(row["before_frame_index"], row["after_frame_index"]) for row in pairs}) == 7
    assert all(row["after_frame_index"] - row["before_frame_index"] == 2 for row in pairs)
    assert all(
        abs(row["timestamp_sec"] - (
            row["before_frame_index"] + row["after_frame_index"]
        ) / 60.0) < 1e-9
        for row in pairs
    )
