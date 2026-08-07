import json
import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = Path(__file__).parents[1] / "libs/models/SADA/anchor_validity.py"
SPEC = importlib.util.spec_from_file_location("anchor_validity", MODULE_PATH)
anchor_validity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(anchor_validity)


def sample_data(count=40):
    torch.manual_seed(7)
    features = torch.randn(count, 12)
    features[:, 6:12] = 0
    features[torch.arange(count), 6 + torch.arange(count) % 6] = 1
    labels = (features[:, 3] + 0.5 * features[:, 5] > 0).float()
    fpn = torch.randn(count, 16)
    return features, fpn, labels


def routing_context():
    return {
        "routing_teacher_state_sha256": "a" * 64,
        "routing_teacher_weights": "ema",
        "alpha": 0.7,
        "beta": 0.1,
        "crop_alignment": 1,
        "fpn_dim": 16,
        "feat_stride": 16.0,
        "feat_num_frames": 32,
        "num_classes": 10,
    }


def test_anchor_features_exclude_actionness_and_keep_geometry_and_level():
    features = anchor_validity.anchor_validity_features(
        torch.tensor([[0.8, 0.2, 0.1]]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
        stride=2.0,
        level=2,
    )
    assert features.shape == (1, 12)
    assert "teacher_actionness" not in anchor_validity.FEATURE_NAMES
    assert torch.isclose(features[0, 3], torch.log1p(torch.tensor(8.0)))
    assert torch.isclose(features[0, 4], torch.log1p(torch.tensor(6.0)))
    assert features[0, 8].item() == 1.0
    assert features[0, 6:12].sum().item() == 1.0


def test_feature_groups_are_the_declared_three_way_ablation():
    assert len(anchor_validity.FEATURE_GROUPS["confidence_only"]) == 3
    assert len(anchor_validity.FEATURE_GROUPS["structure_pure"]) == 9
    assert len(anchor_validity.FEATURE_GROUPS["reliability_full"]) == 12


@pytest.mark.parametrize("model_type", ["logistic", "mlp", "gbdt"])
def test_tabular_models_share_probability_interface(model_type):
    features, _, labels = sample_data()
    if model_type == "logistic":
        model = anchor_validity.fit_logistic_probe(features, labels, max_iter=10)
    elif model_type == "mlp":
        model = anchor_validity.fit_mlp_probe(features, labels, epochs=5)
    else:
        model = anchor_validity.fit_gbdt_probe(
            features, labels, n_estimators=5, max_depth=2
        )
    scores = anchor_validity.probe_probabilities(features, model)
    assert scores.shape == labels.shape
    assert bool(((scores >= 0) & (scores <= 1)).all())


def test_frozen_fpn_validity_head_uses_fpn_and_structural_inputs():
    features, fpn, labels = sample_data()
    model = anchor_validity.fit_fpn_mlp_probe(
        features, fpn, labels, epochs=5
    )
    scores = anchor_validity.probe_probabilities(features, model, fpn)
    assert scores.shape == labels.shape
    payload = anchor_validity.serialize_probe(model, 0.8, {})
    assert payload["evidence_family"] == "representation_validity"


def test_high_precision_calibration_uses_source_val_labels_only():
    result = anchor_validity.high_precision_threshold(
        torch.tensor([0.95, 0.9, 0.8, 0.7]),
        torch.tensor([1, 1, 0, 1]),
        minimum_precision=0.9,
    )
    assert result["threshold"] == pytest.approx(0.9)
    assert result["precision"] == 1.0
    assert result["recall"] == pytest.approx(2 / 3)


def test_validity_artifact_is_source_only_and_hashes_exact_bytes(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            "routing_context": routing_context(),
            "routing_context_sha256": anchor_validity.routing_context_sha256(
                routing_context()
            ),
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, restored, digest = anchor_validity.load_validity_model(path)
    assert loaded["schema_version"] == "source-supervised-anchor-validity-v3"
    assert loaded["evidence_family"] == "reliability_full"
    assert len(digest) == 64
    assert anchor_validity.probe_probabilities(features, restored).shape == labels.shape


def test_validity_artifact_fails_on_teacher_context_mismatch(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    context = routing_context()
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            "routing_context": context,
            "routing_context_sha256": anchor_validity.routing_context_sha256(
                context
            ),
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, _, _ = anchor_validity.load_validity_model(path)
    expected = dict(context)
    expected["routing_teacher_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="differs from frozen routing teacher"):
        anchor_validity.validate_validity_routing_context(loaded, expected)
