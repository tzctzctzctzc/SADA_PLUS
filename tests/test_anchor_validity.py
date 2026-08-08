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


def teacher_provenance():
    return {
        "routing_teacher_checkpoint_sha256": "f" * 64,
        "routing_teacher_state_sha256": "a" * 64,
        "routing_teacher_weights": "ema",
    }


def assignment_protocol():
    return {
        "regression_range": [[0, 4], [2, 8]],
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "reg_distance": True,
        "fpn_strides": [1, 2],
        "n_levels": 2,
    }


def cohort_protocol():
    return {
        "target_class_ids": list(range(10)),
        "action_subsampling": "10_majority_classes",
        "semantic_condition": "center_inside_target_event_strict",
        "confidence_condition": "beta < teacher_score < alpha",
        "alpha": 0.7,
        "beta": 0.1,
        "source_train_split": ["train"],
        "source_calibration_split": ["val"],
    }


def sampling_protocol():
    return {
        "max_seq_len": 2304,
        "crop_ratio": [0.9, 1.0],
        "trunc_thresh": 0.3,
        "crop_alignment": 1,
        "sampling_seed": 7,
        "max_train_anchors": 200000,
        "max_videos": None,
        "train_crops_per_video": 8,
        "calibration_crops_per_video": 4,
        "stratification_policy": "none_random_without_replacement",
    }


def fit_protocol():
    return {
        "model_type": "logistic",
        "feature_group": "reliability_full",
        "max_iter": 50,
        "epochs": 200,
        "learning_rate": 0.001,
        "l2": 0.01,
        "gbdt_estimators": 100,
        "gbdt_learning_rate": 0.05,
        "gbdt_max_depth": 3,
        "fit_batch_size": 4096,
        "seed": 7,
    }


def validation_partition():
    return {
        "policy": "deterministic_video_disjoint_holdout",
        "seed": 11,
        "calibration_fraction": 0.6,
        "calibration_video_ids": ["v1", "v3"],
        "report_video_ids": ["v2"],
    }


def provenance_metadata():
    metadata = {}
    for name, protocol in (
        ("teacher_provenance", teacher_provenance()),
        ("assignment_protocol", assignment_protocol()),
        ("cohort_protocol", cohort_protocol()),
        ("sampling_protocol", sampling_protocol()),
        ("fit_protocol", fit_protocol()),
        ("validation_partition", validation_partition()),
    ):
        metadata[name] = protocol
        metadata[name + "_sha256"] = anchor_validity.protocol_sha256(protocol)
    return metadata


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


def test_integrated_fpn_validity_head_uses_current_features_and_geometry():
    torch.manual_seed(3)
    head = anchor_validity.FPNValidityHead(
        fpn_dim=16,
        n_levels=2,
        hidden_dim=12,
        fusion_dim=7,
        level_embedding_dim=4,
    )
    features = (
        torch.randn(2, 16, 8, requires_grad=True),
        torch.randn(2, 16, 4, requires_grad=True),
    )
    offsets = (
        torch.rand(2, 2, 8, requires_grad=True),
        torch.rand(2, 2, 4, requires_grad=True),
    )
    masks = (
        torch.ones(2, 1, 8, dtype=torch.bool),
        torch.ones(2, 1, 4, dtype=torch.bool),
    )
    logits = head(features, offsets, masks, strides=(1, 2))
    assert [tuple(value.shape) for value in logits] == [(2, 8), (2, 4)]
    sum(value.sum() for value in logits).backward()
    assert features[0].grad is not None
    assert offsets[0].grad is None


def test_frozen_sada_leaves_only_validity_head_trainable():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 2)
            self.validity_head = torch.nn.Linear(2, 1)

    model = ToyModel()
    stats = anchor_validity.freeze_sada_except_validity_head(model)
    assert all(not value.requires_grad for value in model.base.parameters())
    assert all(value.requires_grad for value in model.validity_head.parameters())
    assert stats["trainable_head_parameters"] == 3


def test_frozen_head_source_partition_is_reproducible_and_disjoint():
    rows = tuple({"id": "v{:02d}".format(index)} for index in range(20))
    first = anchor_validity.partition_frozen_head_source_videos(
        rows, selection_videos=3, calibration_videos=4, report_videos=5, seed=17
    )
    second = anchor_validity.partition_frozen_head_source_videos(
        rows, selection_videos=3, calibration_videos=4, report_videos=5, seed=17
    )
    assert first == second
    groups = [{item["id"] for item in group} for group in first[:4]]
    assert [len(group) for group in groups] == [8, 3, 4, 5]
    assert sum(len(group) for group in groups) == len(set().union(*groups))
    assert set().union(*groups) == {item["id"] for item in rows}


def test_validity_head_gradient_does_not_change_base_gradient_clipping():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Linear(2, 1, bias=False)
            self.validity_head = torch.nn.Linear(1, 1, bias=False)

    clipped_base_gradients = []
    for head_gradient in (0.0, 1000.0):
        model = ToyModel()
        model.base.weight.grad = torch.tensor([[3.0, 4.0]])
        model.validity_head.weight.grad = torch.tensor([[head_gradient]])
        anchor_validity.clip_grad_norm_with_isolated_validity_head(model, 1.0)
        clipped_base_gradients.append(model.base.weight.grad.clone())
        assert model.validity_head.weight.grad.norm() <= 1.0
    assert torch.allclose(clipped_base_gradients[0], clipped_base_gradients[1])
    assert torch.allclose(
        clipped_base_gradients[0], torch.tensor([[0.6, 0.8]]), atol=1e-6
    )


def test_source_validity_loss_uses_only_cohort_and_official_labels():
    logits = (torch.tensor([[2.0, -2.0, 9.0]], requires_grad=True),)
    labels = (torch.tensor([[1, 0, 1]], dtype=torch.bool),)
    cohorts = (torch.tensor([[1, 1, 0]], dtype=torch.bool),)
    masks = (torch.ones(1, 3, dtype=torch.bool),)
    loss, stats = anchor_validity.source_validity_loss(
        logits, labels, cohorts, masks
    )
    assert torch.isfinite(loss)
    assert stats == {"count": 2, "positive": 1, "negative": 1}
    loss.backward()
    assert logits[0].grad[0, 2].item() == 0.0


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
        model = anchor_validity.fit_mlp_probe(
            features, labels, epochs=5, batch_size=8
        )
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
        features, fpn, labels, epochs=5, batch_size=8
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


def test_high_precision_failure_reports_observed_ceiling():
    with pytest.raises(ValueError, match="maximum observed precision=0.500000"):
        anchor_validity.high_precision_threshold(
            torch.tensor([0.9, 0.8]),
            torch.tensor([0, 1]),
            minimum_precision=0.9,
        )


def test_source_validation_partition_is_deterministic_and_video_disjoint():
    rows = [{"id": "v{}".format(index)} for index in range(7)]
    first = anchor_validity.partition_source_validation_videos(rows, 0.6, 11)
    second = anchor_validity.partition_source_validation_videos(rows, 0.6, 11)
    assert first == second
    calibration, report, protocol = first
    calibration_ids = {row["id"] for row in calibration}
    report_ids = {row["id"] for row in report}
    assert len(calibration_ids) == 4
    assert len(report_ids) == 3
    assert calibration_ids.isdisjoint(report_ids)
    assert calibration_ids | report_ids == {row["id"] for row in rows}
    assert protocol["calibration_video_ids"] == [row["id"] for row in calibration]
    assert protocol["report_video_ids"] == [row["id"] for row in report]


def test_validity_artifact_is_source_only_and_hashes_exact_bytes(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            **provenance_metadata(),
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, restored, digest = anchor_validity.load_validity_model(path)
    assert loaded["schema_version"] == "source-supervised-anchor-validity-v5"
    assert loaded["evidence_family"] == "reliability_full"
    assert loaded["model_provenance"]["model_type"] == "logistic"
    assert len(digest) == 64
    assert anchor_validity.probe_probabilities(features, restored).shape == labels.shape


def test_validity_artifact_fails_on_teacher_context_mismatch(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            **provenance_metadata(),
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, _, _ = anchor_validity.load_validity_model(path)
    expected = dict(teacher_provenance())
    expected["routing_teacher_state_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="provenance differs"):
        anchor_validity.validate_validity_provenance(
            loaded, expected, assignment_protocol(), cohort_protocol()
        )


def test_validity_artifact_fails_when_tal_assignment_definition_changes(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            **provenance_metadata(),
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded, _, _ = anchor_validity.load_validity_model(path)
    expected_assignment = dict(assignment_protocol())
    expected_assignment["center_sample_radius"] = 2.0
    with pytest.raises(ValueError, match="assignment_protocol"):
        anchor_validity.validate_validity_provenance(
            loaded,
            teacher_provenance(),
            expected_assignment,
            cohort_protocol(),
        )


def test_validity_artifact_rejects_tampered_sampling_protocol(tmp_path):
    features, _, labels = sample_data()
    model = anchor_validity.fit_logistic_probe(features, labels, max_iter=5)
    metadata = provenance_metadata()
    metadata["sampling_protocol"]["sampling_seed"] = 99
    payload = anchor_validity.serialize_probe(
        model,
        0.8,
        {
            "target_gt_usage": "forbidden",
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            **metadata,
        },
    )
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sampling_protocol hash mismatch"):
        anchor_validity.load_validity_model(path)
