import json
import importlib.util
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/anchor_validity.py"
SPEC = importlib.util.spec_from_file_location("anchor_validity", MODULE_PATH)
anchor_validity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchor_validity)
FEATURE_NAMES = anchor_validity.FEATURE_NAMES
STRUCTURE_PROBE_SCHEMA_VERSION = anchor_validity.STRUCTURE_PROBE_SCHEMA_VERSION
anchor_validity_features = anchor_validity.anchor_validity_features
high_precision_threshold = anchor_validity.high_precision_threshold
load_structure_probe = anchor_validity.load_structure_probe
probe_probabilities = anchor_validity.probe_probabilities


def test_anchor_features_use_absolute_regression_distance_and_level():
    features = anchor_validity_features(
        torch.tensor([[0.8, 0.2, 0.1]]),
        torch.tensor([0.4]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        stride=2.0,
        level=2,
    )
    assert features.shape == (1, len(FEATURE_NAMES))
    assert torch.isclose(features[0, 4], torch.log1p(torch.tensor(8.0)))
    assert torch.isclose(features[0, 5], torch.log1p(torch.tensor(6.0)))
    assert features[0, 9].item() == 1.0
    assert features[0, 7:13].sum().item() == 1.0


def test_high_precision_calibration_uses_source_val_labels_only():
    scores = torch.tensor([0.95, 0.9, 0.8, 0.7])
    labels = torch.tensor([1, 1, 0, 1])
    result = high_precision_threshold(scores, labels, minimum_precision=0.9)
    assert result["threshold"] == pytest_approx(0.9)
    assert result["precision"] == 1.0
    assert result["recall"] == pytest_approx(2 / 3)


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_structure_probe_artifact_is_strict_and_hashes_exact_bytes(tmp_path):
    payload = {
        "schema_version": STRUCTURE_PROBE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": [0.0] * len(FEATURE_NAMES),
        "std": [1.0] * len(FEATURE_NAMES),
        "weight": [0.0] * len(FEATURE_NAMES),
        "bias": 0.0,
        "threshold": 0.8,
        "target_gt_usage": "forbidden",
        "supervision": "source_gt_official_label_points_conditioned_on_target_event",
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, probe, digest = load_structure_probe(path)
    assert loaded["target_gt_usage"] == "forbidden"
    assert len(digest) == 64
    assert probe_probabilities(torch.zeros(1, len(FEATURE_NAMES)), probe).item() == 0.5
