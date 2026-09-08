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
