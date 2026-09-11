import torch

from libs.models.SADA.proposal_quality import (
    ProposalIoUQualityHead,
    proposal_region_context,
    proposal_scalar_features,
    quality_weighted_scores,
    same_class_temporal_iou,
    soft_iou_quality_loss,
    video_class_rank_weighted_scores,
)


def test_region_context_detects_boundary_state_changes():
    features = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
    pooled = proposal_region_context(features, torch.tensor([[2.0, 5.0]]), 1)
    assert pooled.shape == (1, 4)
    assert pooled[0, 0] > 0.8
    assert pooled[0, 2] > 0
    assert pooled[0, 3] > 0


def test_same_class_iou_does_not_match_wrong_class():
    proposals = torch.tensor([[0.0, 2.0], [0.0, 2.0], [2.0, 4.0]])
    classes = torch.tensor([0, 1, 0])
    gt_segments = torch.tensor([[0.0, 2.0], [1.0, 3.0]])
    gt_labels = torch.tensor([0, 1])
    quality = same_class_temporal_iou(proposals, classes, gt_segments, gt_labels)
    assert torch.allclose(quality, torch.tensor([1.0, 1.0 / 3.0, 0.0]))


def test_scalar_features_and_head_are_finite():
    scalars = proposal_scalar_features(
        scores=torch.tensor([0.8, 0.2]),
        margins=torch.tensor([0.3, 0.1]),
        segments=torch.tensor([[1.0, 5.0], [2.0, 6.0]]),
        points=torch.tensor([3.0, 4.0]),
        boundary_quality=torch.tensor([0.7, 0.4]),
        class_indices=torch.tensor([1, 2]),
        levels=torch.tensor([0, 1]),
        num_classes=3,
        num_levels=2,
    )
    head = ProposalIoUQualityHead(scalars.shape[-1])
    logits = head(scalars)
    loss = soft_iou_quality_loss(logits, torch.tensor([0.9, 0.1]))
    assert scalars.shape == (2, 11)
    assert torch.isfinite(loss)


def test_quality_score_gamma_zero_is_exact_identity():
    scores = torch.tensor([0.2, 0.8])
    qualities = torch.tensor([0.9, 0.1])
    assert torch.equal(quality_weighted_scores(scores, qualities, 0), scores)
    assert quality_weighted_scores(scores, qualities, 1)[0] > 0


def test_video_class_rank_fusion_is_bounded_and_class_local():
    scores = torch.ones(4)
    qualities = torch.tensor([0.1, 0.9, 0.8, 0.2])
    labels = torch.tensor([0, 0, 1, 1])
    fused = video_class_rank_weighted_scores(scores, qualities, labels, 0.2)
    assert fused[1] > fused[0]
    assert fused[2] > fused[3]
    assert fused.max() <= torch.exp(torch.tensor(0.2))
    assert fused.min() >= torch.exp(torch.tensor(-0.2))
