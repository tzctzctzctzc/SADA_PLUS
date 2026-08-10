import torch

from tools.vlm_boundary_utils import (
    centered_candidates,
    grid_to_seconds,
    hard_class_agnostic_nms_indices,
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
