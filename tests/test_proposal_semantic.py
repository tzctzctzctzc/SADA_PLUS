import importlib.util
import json
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "libs" / "models" / "SADA" / "proposal_semantic.py"
)
SPEC = importlib.util.spec_from_file_location("proposal_semantic", MODULE_PATH)
proposal_semantic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proposal_semantic)


def test_target_adapter_is_identity_at_initialization():
    adapter = proposal_semantic.ProposalTargetAdapter(4, bottleneck=2)
    features = torch.randn(2, 4, 8)

    output = adapter(features)

    assert torch.equal(output, features)


def test_logit_adapter_is_zero_at_initialization_and_detaches_features():
    adapter = proposal_semantic.ProposalLogitAdapter(4, 2, max_delta=1.5)
    features = torch.randn(2, 4, 8, requires_grad=True)

    residual = adapter(features)
    residual.sum().backward()

    assert torch.equal(residual, torch.zeros_like(residual))
    assert features.grad is None
    assert adapter.weight.grad is not None


def test_logit_adapter_construction_does_not_consume_rng_state():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)

    proposal_semantic.ProposalLogitAdapter(4, 2)
    actual = torch.rand(4)

    assert torch.equal(actual, expected)


def test_logit_adapter_temporal_kernel_preserves_length_and_uses_context():
    adapter = proposal_semantic.ProposalLogitAdapter(
        1, 1, max_delta=1.0, kernel_size=3
    )
    with torch.no_grad():
        adapter.weight[0, 0, 0] = 1.0
    features = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])

    residual = adapter(features)

    assert residual.shape == (1, 1, 4)
    assert residual[0, 0, 2] > 0.0
    assert torch.count_nonzero(residual) == 1


def test_logit_adapter_can_learn_confusion_correction_from_base_logits():
    adapter = proposal_semantic.ProposalLogitAdapter(3, 2, max_delta=1.0)
    with torch.no_grad():
        adapter.weight[1, 2, 0] = 1.0
    base_logits = torch.tensor([[[0.0], [0.0], [2.0]]], requires_grad=True)

    residual = adapter(base_logits)
    residual.sum().backward()

    assert residual[0, 1, 0] > 0.0
    assert base_logits.grad is None


def test_logit_adapter_can_disable_global_class_bias():
    adapter = proposal_semantic.ProposalLogitAdapter(
        4, 2, max_delta=1.0, bias=False
    )

    assert adapter.bias is None
    assert torch.equal(
        adapter(torch.randn(2, 4, 5)),
        torch.zeros(2, 2, 5),
    )


def test_mlp_logit_adapter_is_zero_and_preserves_rng_state():
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    adapter = proposal_semantic.ProposalMLPLogitAdapter(
        4, 2, max_delta=1.0, bottleneck=3
    )
    actual = torch.rand(4)
    features = torch.randn(2, 4, 5, requires_grad=True)

    residual = adapter(features)
    residual.sum().backward()

    assert torch.equal(actual, expected)
    assert torch.equal(residual, torch.zeros_like(residual))
    assert features.grad is None
    assert adapter.output_weight.grad is not None


def test_proposal_detection_soft_ce_prefers_matching_distribution():
    logits = [torch.tensor([[[4.0, -2.0], [-2.0, 4.0]]])]
    proposals = [{
        "segments": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        "levels": torch.tensor([0, 0]),
        "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }]

    matching, count = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="soft_ce"
    )
    reversed_loss, _ = proposal_semantic.proposal_detection_loss(
        [logits[0].flip(-1)], proposals, (1,), loss_type="soft_ce"
    )

    assert count == 2
    assert matching < reversed_loss


def test_known_class_ce_does_not_suppress_unqueried_classes():
    class_count = len(proposal_semantic.KNOWN_LABELS) + 1
    logits = [torch.zeros(1, 1, class_count, requires_grad=True)]
    targets = torch.zeros(1, class_count)
    targets[0, 0] = 0.8
    targets[0, 1] = 0.2
    proposals = [{
        "segments": torch.tensor([[0.0, 0.0]]),
        "levels": torch.tensor([0]),
        "targets": targets,
    }]

    loss, _ = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="known_soft_ce"
    )
    loss.backward()

    assert torch.count_nonzero(logits[0].grad[..., :-1]) > 0
    assert logits[0].grad[..., -1].item() == 0.0


def test_positive_proposal_loss_does_not_update_unselected_classes():
    logits = [torch.zeros(1, 1, 3, requires_grad=True)]
    proposals = [{
        "segments": torch.tensor([[0.0, 0.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[0.8, 0.1, 0.1]]),
    }]

    loss, _ = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="positive_bce"
    )
    loss.backward()

    assert logits[0].grad[0, 0, 0] < 0
    assert torch.equal(logits[0].grad[0, 0, 1:], torch.zeros(2))


def test_top2_positive_loss_keeps_uncertainty_without_negative_gradients():
    logits = [torch.zeros(1, 1, 3, requires_grad=True)]
    proposals = [{
        "segments": torch.tensor([[0.0, 0.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[0.6, 0.3, 0.1]]),
    }]

    loss, _ = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="top2_positive_bce"
    )
    loss.backward()

    assert torch.all(logits[0].grad[0, 0, :2] < 0)
    assert logits[0].grad[0, 0, 2] == 0


def test_top2_pooling_treats_a_proposal_as_a_temporal_bag():
    logits = [torch.tensor([[[-4.0], [4.0], [3.0], [-4.0], [-4.0]]])]
    proposals = [{
        "segments": torch.tensor([[0.0, 4.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
    }]

    mean_loss, _ = proposal_semantic.proposal_detection_loss(
        logits,
        proposals,
        (1,),
        loss_type="positive_bce",
        pooling="mean",
    )
    top2_loss, _ = proposal_semantic.proposal_detection_loss(
        logits,
        proposals,
        (1,),
        loss_type="positive_bce",
        pooling="top2",
    )

    assert top2_loss < mean_loss


def test_topk_pooling_controls_number_of_positive_instances():
    logits = [torch.tensor([[[-4.0], [4.0], [3.0], [2.0], [-4.0]]])]
    proposals = [{
        "segments": torch.tensor([[0.0, 4.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
    }]

    top1, _ = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="positive_bce", pooling="top1"
    )
    top3, _ = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,), loss_type="positive_bce", pooling="top3"
    )

    assert top1 < top3


def test_inner_outer_ranking_penalizes_temporal_leakage():
    localized = [torch.tensor([[[-4.0], [4.0], [4.0], [-4.0], [-4.0]]])]
    leaked = [torch.tensor([[[4.0], [4.0], [4.0], [4.0], [4.0]]])]
    proposals = [{
        "segments": torch.tensor([[1.0, 2.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
    }]

    localized_loss, _ = proposal_semantic.proposal_detection_loss(
        localized,
        proposals,
        (1,),
        loss_type="positive_bce_rank",
        pooling="top2",
    )
    leaked_loss, _ = proposal_semantic.proposal_detection_loss(
        leaked,
        proposals,
        (1,),
        loss_type="positive_bce_rank",
        pooling="top2",
    )

    assert localized_loss < leaked_loss


def test_language_advantage_focuses_gradient_on_base_model_blind_spot():
    adapted = [torch.zeros(2, 5, 1, requires_grad=True)]
    base = [torch.tensor([
        [[-4.0], [-4.0], [-4.0], [-4.0], [-4.0]],
        [[4.0], [4.0], [4.0], [4.0], [4.0]],
    ])]
    proposals = [{
        "segments": torch.tensor([[1.0, 2.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
    } for _ in range(2)]

    loss, _ = proposal_semantic.proposal_detection_loss(
        adapted,
        proposals,
        (1,),
        loss_type="positive_bce_rank_advantage",
        pooling="top1",
        base_logits=base,
    )
    loss.backward()

    blind_spot_gradient = adapted[0].grad[0].abs().sum()
    already_solved_gradient = adapted[0].grad[1].abs().sum()
    assert blind_spot_gradient > 0
    assert blind_spot_gradient > 20 * already_solved_gradient


def test_teacher_score_continuously_reweights_without_dropping_proposals():
    adapted = [torch.zeros(2, 5, 1, requires_grad=True)]
    proposals = [{
        "segments": torch.tensor([[1.0, 2.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
        "teacher_scores": torch.tensor([score]),
    } for score in (0.2, 0.8)]

    loss, count = proposal_semantic.proposal_detection_loss(
        adapted,
        proposals,
        (1,),
        loss_type="positive_bce_rank_teacher_score",
        pooling="top1",
    )
    loss.backward()

    low_gradient = adapted[0].grad[0].abs().sum()
    high_gradient = adapted[0].grad[1].abs().sum()
    assert count == 2
    assert high_gradient > 3 * low_gradient


def test_outer_anchor_penalizes_only_residual_leakage():
    base = [torch.zeros(1, 5, 1)]
    localized = [base[0].clone()]
    localized[0][0, 1:3, 0] = 4.0
    leaked = [localized[0].clone()]
    leaked[0][0, (0, 3, 4), 0] = 1.0
    proposals = [{
        "segments": torch.tensor([[1.0, 2.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0]]),
    }]

    localized_loss, _ = proposal_semantic.proposal_detection_loss(
        localized,
        proposals,
        (1,),
        loss_type="positive_bce_outer_anchor",
        pooling="top2",
        base_logits=base,
    )
    leaked_loss, _ = proposal_semantic.proposal_detection_loss(
        leaked,
        proposals,
        (1,),
        loss_type="positive_bce_outer_anchor",
        pooling="top2",
        base_logits=base,
    )

    assert localized_loss < leaked_loss


def test_background_anchor_excludes_vlm_positive_bag_and_invalid_time():
    base = [torch.full((1, 4, 2), -10.0)]
    adapted = [base[0] + 1.0]
    masks = [torch.tensor([[[True, True, True, False]]])]
    proposals = [{
        "segments": torch.tensor([[1.0, 2.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0, 0.0]]),
    }]

    loss, count = proposal_semantic.proposal_background_anchor_loss(
        adapted,
        base,
        masks,
        proposals,
        (1,),
        confidence_threshold=0.05,
    )

    assert count == 1
    assert loss.item() == pytest.approx(1.0)


def test_class_balance_upweights_rare_labels_and_preserves_mean_scale():
    entries = [
        {
            "semantic_label": label,
            "class_probabilities": probabilities,
        }
        for label, probabilities in (
            [("TAKE", [0.8, 0.2])] * 4
            + [("PUT", [0.1, 0.9])]
            + [("OPEN", [0.6, 0.4])]
        )
    ]

    weights, counts = proposal_semantic.proposal_class_balance_weights(
        entries,
        num_classes=2,
        confidence_threshold=0.7,
        power=0.5,
        max_weight=2.0,
    )

    assert counts.tolist() == [4.0, 1.0]
    assert weights[1] > weights[0]
    assert float((weights * counts).sum() / counts.sum()) == pytest.approx(1.0)


def test_source_class_reliability_preserves_target_weight_scale(tmp_path):
    calibration = tmp_path / "reliability.json"
    calibration.write_text(json.dumps({
        "schema_version": "proposal-class-reliability-v1",
        "target_gt_usage": "forbidden",
        "gt_view": "proposal_interval_any",
        "smoothed_reliabilities": [0.8, 0.4] + [0.6] * 8,
    }))
    entries = [
        {"semantic_label": "TAKE", "class_probabilities": [0.9, 0.1]},
        {"semantic_label": "PUT", "class_probabilities": [0.1, 0.9]},
    ]

    weights, counts, _, _ = (
        proposal_semantic.load_proposal_class_reliability(
            calibration, 2, entries, confidence_threshold=0.7
        )
    )

    assert counts.tolist() == [1.0, 1.0]
    assert weights[0] > weights[1]
    assert float((weights * counts).sum() / counts.sum()) == pytest.approx(1.0)


def test_base_ambiguity_gate_protects_confident_foreground_and_background():
    probabilities = torch.tensor([0.01, 0.5, 0.99])
    logits = torch.logit(probabilities).reshape(1, 1, 3)

    gate = proposal_semantic.proposal_logit_gate(
        logits, "base_ambiguity"
    ).flatten()

    assert gate[1] == pytest.approx(1.0)
    assert gate[0] == pytest.approx(float(gate[2]), abs=1e-6)
    assert gate[0] < 0.05


def test_vlm_proposal_agreement_gate_uses_frozen_base_logits():
    logits = [torch.tensor([[
        [4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]
    ]])]
    proposals = [{
        "segments": torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        "levels": torch.tensor([0, 0]),
        "evidence": torch.tensor([
            [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        ]),
        "evidence_eligible": torch.tensor([True, True]),
    }]

    stats = proposal_semantic.prepare_vlm_proposal_targets(
        logits, proposals, (1,), policy="agree", confidence_threshold=0.5
    )

    assert stats["eligible_count"] == 2
    assert stats["selected_count"] == 1
    assert torch.allclose(
        proposals[0]["targets"],
        torch.tensor([[0.8, 0.2], [0.0, 0.0]]),
    )


def test_proposal_detection_loss_supervises_detector_logits_only():
    logits = [torch.zeros(1, 8, 2, requires_grad=True)]
    targets = torch.tensor([[0.0, 1.0]], requires_grad=True)
    proposals = [{
        "segments": torch.tensor([[2.0, 6.0]]),
        "levels": torch.tensor([0]),
        "targets": targets,
    }]

    loss, count = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,)
    )
    loss.backward()

    assert count == 1
    assert logits[0].grad is not None
    assert logits[0].grad.abs().sum() > 0
    assert targets.grad is None


def test_observable_oracle_labels_feed_proposal_detection_loss():
    proposals = [{
        "segments": torch.tensor([[0.0, 4.0], [6.0, 8.0]]),
        "levels": torch.tensor([0, 0]),
        "evidence": torch.zeros(2, 6),
        "evidence_eligible": torch.ones(2, dtype=torch.bool),
    }]
    targets = proposal_semantic.labels_for_proposals(
        proposals,
        [torch.tensor([[1.0, 5.0]])],
        [torch.tensor([1])],
        2,
        "soft",
    )
    proposals[0]["targets"] = targets[0]
    logits = [torch.zeros(1, 10, 2, requires_grad=True)]

    loss, count = proposal_semantic.proposal_detection_loss(
        logits, proposals, (1,)
    )

    assert count == 1
    assert loss.item() > 0


def test_teacher_proposals_decode_geometry_and_apply_class_agnostic_nms():
    points = [torch.tensor([
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [8.0, 0.0, 0.0, 1.0],
    ])]
    masks = [torch.tensor([[True, True, True]])]
    logits = [torch.tensor([[[4.0, 0.0], [3.0, 0.0], [0.0, 4.0]]])]
    offsets = [torch.tensor([[[1.0, 2.0], [2.0, 1.0], [1.0, 1.0]]])]

    result = proposal_semantic.build_teacher_proposals(
        points, masks, logits, offsets, (0,), 0.8, 4, 0.5
    )[0]

    assert result["segments"].tolist() == [[0.0, 3.0], [7.0, 9.0]]
    assert result["levels"].tolist() == [0, 0]
    assert not result["segments"].requires_grad
    assert torch.allclose(result["evidence"][:, :2].sum(dim=-1), torch.ones(2))


def test_oracle_labels_distinguish_center_dominant_and_soft_interval():
    proposals = [{
        "segments": torch.tensor([[0.0, 10.0]]),
        "levels": torch.tensor([0]),
        "evidence": torch.zeros(1, 6),
        "evidence_eligible": torch.ones(1, dtype=torch.bool),
    }]
    segments = [torch.tensor([[0.0, 4.0], [4.0, 10.0]])]
    labels = [torch.tensor([0, 1])]

    center = proposal_semantic.labels_for_proposals(
        proposals, segments, labels, 2, "center"
    )[0]
    dominant = proposal_semantic.labels_for_proposals(
        proposals, segments, labels, 2, "dominant"
    )[0]
    soft = proposal_semantic.labels_for_proposals(
        proposals, segments, labels, 2, "soft"
    )[0]

    assert center.tolist() == [[0.0, 1.0]]
    assert dominant.tolist() == [[0.0, 1.0]]
    assert torch.allclose(soft, torch.tensor([[0.4, 0.6]]))


def test_cached_vlm_special_labels_abstain_and_convert_to_crop_grid():
    cache = {"video": [
        {
            "candidate_id": "known",
            "absolute_feature_index": 10,
            "proposal_start_sec": 1.0,
            "proposal_end_sec": 2.0,
            "absolute_center_time": 1.5,
            "semantic_label": "TAKE",
            "source_fpn_levels": (0,),
        },
        {
            "candidate_id": "unknown",
            "absolute_feature_index": 12,
            "proposal_start_sec": 2.0,
            "proposal_end_sec": 3.0,
            "absolute_center_time": 2.5,
            "semantic_label": "ABSTAIN",
            "source_fpn_levels": (0,),
        },
    ]}
    videos = [{
        "feats": torch.zeros(4, 40),
        "metadata": {
            "video_id": "video",
            "crop_start": 2,
            "crop_feat_len": 40,
            "feat_stride": 10,
            "feat_num_frames": 0,
            "fps": 10,
        },
    }]

    result = proposal_semantic.cached_proposals_for_videos(
        cache, videos, 2, 1, "cpu", torch.float32
    )[0]

    assert result["segments"].tolist() == [[0.0, 1.0]]
    # The first interval ends at the crop boundary and is excluded.
    assert result["evidence_eligible"].tolist() == [False]
    assert result["evidence"][0, 4].item() == 1.0


def test_cached_vlm_probability_vector_is_used_as_semantic_embedding():
    probabilities = (0.7, 0.3)
    cache = {"video": [{
        "candidate_id": "known",
        "absolute_feature_index": 10,
        "proposal_start_sec": 3.0,
        "proposal_end_sec": 5.0,
        "absolute_center_time": 4.0,
        "semantic_label": "TAKE",
        "source_fpn_levels": (0,),
        "class_probabilities": probabilities,
    }]}
    videos = [{
        "feats": torch.zeros(4, 10),
        "metadata": {
            "video_id": "video", "crop_start": 0, "crop_feat_len": 10,
            "feat_stride": 10, "feat_num_frames": 0, "fps": 10,
        },
    }]

    result = proposal_semantic.cached_proposals_for_videos(
        cache, videos, 2, 1, "cpu", torch.float32
    )[0]

    assert result["evidence"][0, :2].tolist() == pytest.approx(probabilities)
    assert result["evidence_eligible"].tolist() == [True]


@pytest.mark.parametrize("source", [
    "offline_vlm_proposal_semantics",
    "offline_vlm_consensus_proposal_semantics",
])
def test_loader_accepts_ten_vlm_classes_with_original_tal_class_count(
    tmp_path, source
):
    path = tmp_path / "proposal_semantic.json"
    path.write_text(json.dumps({
        "schema_version": "proposal-semantic-evidence-v1",
        "source": source,
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {},
        "prompt_protocol": {},
        "entries": [{
            "candidate_id": "video-p1",
            "video_id": "video",
            "absolute_feature_index": 10,
            "absolute_center_time": 1.5,
            "proposal_start_sec": 1.0,
            "proposal_end_sec": 2.0,
            "semantic_label": "TAKE",
            "source_fpn_levels": [0],
            "class_probabilities": [0.1] * 10,
        }],
    }), encoding="utf-8")

    cache, count, _, _ = proposal_semantic.load_proposal_semantic_cache(
        path, 97
    )

    assert count == 1
    assert cache["video"][0]["class_probabilities"] == pytest.approx(
        (0.1,) * 10
    )
    vector = proposal_semantic.evidence_vector(
        "TAKE", 97, "cpu", torch.float32, (0.1,) * 10
    )
    assert vector.shape == (101,)
    assert vector[:10].tolist() == pytest.approx([0.1] * 10)


def test_branch_preserves_temporal_order_and_detaches_target_q():
    branch = proposal_semantic.ProposalSemanticBranch(
        fpn_dim=2,
        num_classes=2,
        hidden_dim=12,
        embedding_dim=4,
        confidence_threshold=0.0,
        entropy_threshold=1.0,
    )
    features = [torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)]
    logits = [torch.zeros(1, 8, 2, requires_grad=True)]
    proposals = [{
        "segments": torch.tensor([[2.0, 6.0]]),
        "levels": torch.tensor([0]),
        "evidence": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "evidence_eligible": torch.tensor([True]),
        "targets": torch.tensor([[1.0, 0.0]]),
    }]

    encoded = branch.encode(features, logits, proposals, (1,))
    source_loss = branch.source_loss(encoded, proposals)
    selection = branch.select_target(encoded, proposals)
    source_loss.backward()

    assert encoded["embedding"].shape == (1, 4)
    assert selection["selected"].tolist() == [True]
    assert selection["q"].requires_grad is False
    assert logits[0].grad is not None


def test_cotrain_branch_updates_visual_features_without_vlm_shortcut():
    branch = proposal_semantic.VLMProposalCoAdaptationBranch(
        fpn_dim=4,
        num_classes=2,
        hidden_dim=16,
        embedding_dim=8,
        confidence_threshold=0.7,
    )
    assert branch.classifier.bias is None
    source_features = [torch.randn(1, 4, 8, requires_grad=True)]
    target_features = [torch.randn(1, 4, 8, requires_grad=True)]
    source = [{
        "segments": torch.tensor([[2.0, 6.0]]),
        "levels": torch.tensor([0]),
        "targets": torch.tensor([[1.0, 0.0]]),
    }]
    target_evidence = torch.tensor(
        [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    target = [{
        "segments": torch.tensor([[2.0, 6.0]]),
        "levels": torch.tensor([0]),
        "evidence": target_evidence,
        "evidence_eligible": torch.tensor([True]),
    }]

    encoded_source = branch.encode(source_features, source, (1,))
    encoded_target = branch.encode(target_features, target, (1,))
    source_targets = branch.flatten_field(source, "targets")
    selection = branch.select_vlm_target(target)
    source_loss = branch.source_loss(encoded_source, source)
    positive_loss, positive_count = branch.target_positive_loss(
        encoded_target, selection
    )
    alignment_loss, aligned_classes, _ = branch.prototype_alignment(
        encoded_source,
        source_targets,
        encoded_target,
        selection,
    )
    (source_loss + positive_loss + alignment_loss).backward()

    assert positive_count == 1
    assert aligned_classes == 1
    assert source_features[0].grad is not None
    assert source_features[0].grad.abs().sum() > 0
    assert target_features[0].grad is not None
    assert target_features[0].grad.abs().sum() > 0
    assert target_evidence.grad is None


def test_cotrain_target_selection_comes_from_vlm_not_student_logits():
    branch = proposal_semantic.VLMProposalCoAdaptationBranch(
        fpn_dim=4,
        num_classes=2,
        hidden_dim=16,
        embedding_dim=8,
        confidence_threshold=0.7,
    )
    proposals = [{
        "segments": torch.tensor([[1.0, 3.0], [4.0, 6.0]]),
        "levels": torch.tensor([0, 0]),
        "evidence": torch.tensor([
            [0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.4, 0.6, 0.0, 0.0, 0.0, 0.0],
        ]),
        "evidence_eligible": torch.tensor([True, True]),
    }]

    selection = branch.select_vlm_target(proposals)

    assert selection["labels"].tolist() == [0, 1]
    assert selection["selected"].tolist() == [True, False]
    assert selection["q"].requires_grad is False


def test_dual_stream_is_zero_initialized_and_detached_from_detector():
    branch = proposal_semantic.VLMDualStreamSemanticBranch(
        fpn_dim=4,
        num_classes=3,
        n_levels=2,
        hidden_dim=8,
        global_tokens=3,
        max_delta=1.0,
    )
    features = [
        torch.randn(2, 4, 7, requires_grad=True),
        torch.randn(2, 4, 4, requires_grad=True),
    ]
    masks = [
        torch.ones(2, 1, 7, dtype=torch.bool),
        torch.ones(2, 1, 4, dtype=torch.bool),
    ]
    foreground = [
        torch.full((2, 1, 7), 0.5, requires_grad=True),
        torch.full((2, 1, 4), 0.5, requires_grad=True),
    ]

    output = branch(features, masks, foreground_gates=foreground)
    assert [value.shape for value in output["semantic_logits"]] == [
        (2, 7, 3), (2, 4, 3)
    ]
    assert all(torch.count_nonzero(value) == 0 for value in output["residuals"])
    loss = sum(value.sum() for value in output["semantic_logits"])
    loss = loss + sum(value.sum() for value in output["residuals"])
    loss.backward()

    assert branch.classifier.weight.grad is not None
    assert all(value.grad is None for value in features)
    assert all(value.grad is None for value in foreground)


def test_dual_stream_foreground_confidence_and_validity_gate_dense_residual():
    branch = proposal_semantic.VLMDualStreamSemanticBranch(
        fpn_dim=4,
        num_classes=2,
        n_levels=1,
        hidden_dim=8,
        global_tokens=2,
        max_delta=0.75,
    )
    with torch.no_grad():
        branch.classifier.weight.fill_(0.1)
    features = [torch.randn(1, 4, 5)]
    masks = [torch.tensor([[[True, True, True, True, False]]])]
    foreground = [torch.tensor([[[0.0, 1.0, 1.0, 0.0, 1.0]]])]

    residual = branch(
        features, masks, foreground_gates=foreground
    )["residuals"][0]

    assert residual[:, 1:3].abs().sum() > 0
    assert residual[:, [0, 3]].abs().max() < 1e-6
    assert torch.count_nonzero(residual[:, 4]) == 0
    assert residual.abs().max() <= 0.75


def test_prototype_alignment_uses_soft_q_and_source_memory_without_q_gradient():
    branch = proposal_semantic.ProposalSemanticBranch(
        fpn_dim=1,
        num_classes=2,
        hidden_dim=4,
        embedding_dim=2,
    )
    source_embedding = torch.tensor([[1.0, 0.0]], requires_grad=True)
    source = {"embedding": source_embedding}
    source_targets = torch.tensor([[1.0, 0.0]])
    target_embedding = torch.tensor([[0.0, 1.0]], requires_grad=True)
    target = {"embedding": target_embedding}
    q = torch.tensor([[0.8, 0.2]])
    selection = {"q": q, "selected": torch.tensor([True])}

    loss, classes = branch.prototype_alignment(
        source, source_targets, target, selection
    )
    loss.backward()

    assert classes == 1
    assert source_embedding.grad is None
    assert target_embedding.grad is not None
    assert q.grad is None
    assert branch.source_prototype_ready.tolist() == [True, False]


def test_prototype_memory_is_stable_across_batches_until_backward():
    branch = proposal_semantic.ProposalSemanticBranch(
        fpn_dim=1,
        num_classes=2,
        hidden_dim=4,
        embedding_dim=2,
    )
    branch.prototype_alignment(
        {"embedding": torch.tensor([[1.0, 0.0]])},
        torch.tensor([[1.0, 0.0]]),
        {"embedding": torch.tensor([[0.0, 1.0]])},
        {"q": torch.tensor([[1.0, 0.0]]), "selected": torch.tensor([False])},
    )

    target_embedding = torch.tensor([[0.0, 1.0]], requires_grad=True)
    loss, classes = branch.prototype_alignment(
        {"embedding": torch.tensor([[0.0, 1.0]], requires_grad=True)},
        torch.tensor([[0.0, 1.0]]),
        {"embedding": target_embedding},
        {"q": torch.tensor([[1.0, 0.0]]), "selected": torch.tensor([True])},
    )
    loss.backward()

    assert classes == 1
    assert target_embedding.grad is not None
    assert branch.source_prototype_ready.tolist() == [True, True]


def test_prototype_alignment_excludes_softmax_tail_classes():
    branch = proposal_semantic.ProposalSemanticBranch(
        fpn_dim=1,
        num_classes=2,
        hidden_dim=4,
        embedding_dim=2,
    )
    source_embedding = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    target_embedding = torch.tensor([[0.5, 0.5]], requires_grad=True)
    loss, classes = branch.prototype_alignment(
        {"embedding": source_embedding},
        torch.eye(2),
        {"embedding": target_embedding},
        {
            "q": torch.tensor([[0.95, 0.05]]),
            "selected": torch.tensor([True]),
        },
    )
    loss.backward()

    assert classes == 1
    assert target_embedding.grad is not None


def test_cache_loader_keeps_only_semantics_and_proposal_geometry(tmp_path):
    payload = {
        "schema_version": "proposal-semantic-evidence-v1",
        "source": "offline_vlm_proposal_semantics",
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {},
        "prompt_protocol": {},
        "entries": [{
            "candidate_id": "v-p1",
            "video_id": "v",
            "absolute_feature_index": 1,
            "absolute_center_time": 1.5,
            "proposal_start_sec": 1.0,
            "proposal_end_sec": 2.0,
            "source_fpn_levels": [0],
            "semantic_label": "PUT",
        }],
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cache, count, digest, loaded = proposal_semantic.load_proposal_semantic_cache(
        path, 10
    )

    assert count == 1
    assert len(digest) == 64
    assert loaded["entries"][0].keys().isdisjoint({
        "gt_assigned_class_ids", "center_all_gt_class_ids", "semantic_gt_kind"
    })
    assert cache["v"][0]["semantic_label"] == "PUT"


def test_vlm_anchor_routes_use_absolute_point_not_cached_boundaries():
    features = [torch.ones(1, 2, 9)]
    masks = [torch.ones(1, 1, 9, dtype=torch.bool)]
    videos = [{
        "feats": torch.ones(2, 9),
        "metadata": {"video_id": "v", "crop_start": 3, "crop_feat_len": 9},
    }]
    entry = {
        "absolute_feature_index": 7,
        "proposal_start_sec": 0.0,
        "proposal_end_sec": 100.0,
        "source_fpn_levels": [0],
        "semantic_label": "PUT",
        "class_probabilities": [0.05, 0.8] + [0.0] * 8,
    }

    first, stats = proposal_semantic.build_vlm_anchor_routes(
        {"v": [entry]}, videos, features, masks, (1,), ((0, 4),),
        (0,), 10, confidence_threshold=0.7, topk=3,
    )
    changed = dict(entry, proposal_start_sec=49.0, proposal_end_sec=50.0)
    second, _ = proposal_semantic.build_vlm_anchor_routes(
        {"v": [changed]}, videos, features, masks, (1,), ((0, 4),),
        (0,), 10, confidence_threshold=0.7, topk=3,
    )

    assert stats["anchor_count"] == 1
    assert stats["selected_count"] == 3
    assert first[0]["selected"][0, 4]
    assert first[0]["pseudo_class"][0, 4].item() == 1
    assert torch.equal(first[0]["selected"], second[0]["selected"])
    assert torch.equal(first[0]["targets"], second[0]["targets"])


def test_vlm_anchor_route_override_preserves_uncovered_sada_routes():
    base_targets = torch.tensor([
        [1.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ])
    base_selected = torch.tensor([True, False, True, False])
    route = {
        "selected": torch.tensor([[False, False, True, False]]),
        "targets": torch.tensor([[[0.0, 0.0], [0.0, 0.0],
                                  [0.0, 1.0], [0.0, 0.0]]]),
    }
    valid = torch.ones(1, 4, dtype=torch.bool)

    targets, selected, overridden = proposal_semantic.override_target_routes(
        base_targets, base_selected, route, valid
    )

    assert torch.equal(targets[0], base_targets[0])
    assert torch.equal(targets[1], base_targets[1])
    assert torch.equal(targets[2], torch.tensor([0.0, 1.0]))
    assert selected.tolist() == [True, False, True, False]
    assert overridden.tolist() == [False, False, True, False]


def test_semantic_transport_is_zero_initialized_and_detaches_fpn():
    branch = proposal_semantic.VLMSemanticTransportBranch(
        fpn_dim=4,
        num_classes=2,
        n_levels=1,
        hidden_dim=8,
        temperature=0.2,
        max_residual=0.1,
    )
    source = [torch.randn(1, 4, 5, requires_grad=True)]
    target = [torch.randn(1, 4, 5, requires_grad=True)]
    masks = [torch.ones(1, 1, 5, dtype=torch.bool)]
    source_labels = [torch.tensor([[
        [1.0, 0.0], [1.0, 0.0], [0.0, 1.0],
        [0.0, 1.0], [0.0, 0.0],
    ]])]
    routes = [{
        "selected": torch.tensor([[False, True, False, True, False]]),
        "scores": torch.tensor([[0.0, 0.9, 0.0, 0.8, 0.0]]),
        "pseudo_class": torch.tensor([[0, 0, 0, 1, 0]]),
        "targets": torch.tensor([[
            [0.0, 0.0], [1.0, 0.0], [0.0, 0.0],
            [0.0, 1.0], [0.0, 0.0],
        ]]),
    }]

    output = branch(
        source, target, masks, masks,
        source_labels=source_labels,
        target_routes=routes,
        enabled_levels=(0,),
        update_prototypes=True,
    )
    assert torch.equal(output["source_detection_feats"][0], source[0])
    assert torch.equal(output["target_detection_feats"][0], target[0])
    assert branch.source_prototype_ready[0].tolist() == [True, True]
    loss = output["source_loss"] + output["alignment_loss"] + output["direct_loss"]
    assert torch.isfinite(loss)
    assert output["direct_loss"] > 0.01
    loss.backward()

    assert source[0].grad is None
    assert target[0].grad is None
    assert branch.output.weight.grad is not None
    assert branch.output.weight.grad.abs().sum() > 0


def test_semantic_transport_multilevel_prototype_updates_allow_backward():
    branch = proposal_semantic.VLMSemanticTransportBranch(
        fpn_dim=4, num_classes=2, n_levels=2, hidden_dim=8
    )
    source = [torch.randn(1, 4, 5), torch.randn(1, 4, 3)]
    target = [torch.randn(1, 4, 5), torch.randn(1, 4, 3)]
    masks = [
        torch.ones(1, 1, 5, dtype=torch.bool),
        torch.ones(1, 1, 3, dtype=torch.bool),
    ]
    source_labels = [
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0],
                       [0.0, 0.0], [0.0, 0.0]]]),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]]),
    ]
    routes = [
        {
            "selected": torch.tensor([[True, False, False, False, False]]),
            "scores": torch.tensor([[0.9, 0.0, 0.0, 0.0, 0.0]]),
            "pseudo_class": torch.zeros(1, 5, dtype=torch.long),
            "targets": torch.tensor([[[1.0, 0.0], [0.0, 0.0],
                                      [0.0, 0.0], [0.0, 0.0],
                                      [0.0, 0.0]]]),
        },
        {
            "selected": torch.tensor([[True, False, False]]),
            "scores": torch.tensor([[0.9, 0.0, 0.0]]),
            "pseudo_class": torch.zeros(1, 3, dtype=torch.long),
            "targets": torch.tensor([[[1.0, 0.0], [0.0, 0.0],
                                      [0.0, 0.0]]]),
        },
    ]

    output = branch(
        source, target, masks, masks,
        source_labels=source_labels,
        target_routes=routes,
        enabled_levels=(0, 1),
        update_prototypes=True,
    )
    loss = output["source_loss"] + output["direct_loss"]
    loss.backward()

    assert branch.input_projection.weight.grad is not None
    assert branch.output.weight.grad is not None
