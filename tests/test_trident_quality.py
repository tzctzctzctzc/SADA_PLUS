import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1] / "libs" / "models" / "SADA" / "trident_quality.py"
)
SPEC = importlib.util.spec_from_file_location("trident_quality_standalone", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
boundary_peak_quality = MODULE.boundary_peak_quality
center_only_offsets = MODULE.center_only_offsets
quality_weighted_scores = MODULE.quality_weighted_scores


def test_boundary_peak_quality_prefers_sharp_distributions():
    center = torch.tensor([
        [8.0, 0.0, 0.0, 8.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    neighbours = torch.zeros(1, 2, 3)
    quality = boundary_peak_quality(center, neighbours, neighbours)
    assert quality.shape == (1, 2)
    assert quality[0, 0] > quality[0, 1]


def test_quality_gamma_zero_preserves_scores():
    scores = torch.tensor([0.3, 0.8])
    quality = torch.tensor([0.1, 0.9])
    assert torch.equal(quality_weighted_scores(scores, quality, 0.0), scores)


def test_quality_gamma_must_be_non_negative():
    with pytest.raises(ValueError):
        quality_weighted_scores(torch.ones(1), torch.ones(1), -1.0)


def test_center_only_offsets_ignore_class_neighbours():
    logits = torch.full((1, 6), -20.0)
    logits[0, 0] = 20.0
    logits[0, 5] = 20.0
    offsets = center_only_offsets(logits)
    assert torch.allclose(offsets, torch.tensor([[2.0, 2.0]]), atol=1e-4)
