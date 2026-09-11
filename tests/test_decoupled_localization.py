import torch

from libs.models.SADA.decoupled_localization import (
    DirectLocalizationHead,
    balanced_soft_quality_loss,
    fuse_class_and_quality,
    initialize_from_trident_regression_trunk,
)


def test_direct_localization_head_shapes_and_positive_offsets():
    head = DirectLocalizationHead(8, 16, 2, num_layers=3, kernel_size=3)
    features = [torch.randn(2, 8, 11), torch.randn(2, 8, 6)]
    masks = [torch.ones(2, 1, 11, dtype=torch.bool), torch.ones(2, 1, 6, dtype=torch.bool)]
    offsets, quality = head(features, masks)
    assert [tuple(value.shape) for value in offsets] == [(2, 2, 11), (2, 2, 6)]
    assert [tuple(value.shape) for value in quality] == [(2, 1, 11), (2, 1, 6)]
    assert all(torch.all(value > 0) for value in offsets)


def test_balanced_quality_loss_weights_groups_equally():
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0])
    positive = torch.tensor([True, False, False, False])
    valid = torch.ones(4, dtype=torch.bool)
    loss = balanced_soft_quality_loss(logits, targets, positive, valid)
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-6)


def test_quality_fusion_is_pointwise_and_bounded():
    classes = torch.tensor([[0.8, 0.4], [0.6, 0.3]])
    quality_logits = torch.tensor([[0.0], [2.0]])
    fused = fuse_class_and_quality(classes, quality_logits, gamma=1.0)
    assert torch.allclose(fused[0], classes[0] * 0.5)
    assert torch.all(fused <= classes)
    assert torch.allclose(
        fuse_class_and_quality(classes, quality_logits, gamma=0.0), classes
    )


def test_trident_trunk_initialization_copies_shared_layers_only():
    source = DirectLocalizationHead(8, 16, 2, num_layers=3, kernel_size=3)
    destination = DirectLocalizationHead(8, 16, 2, num_layers=3, kernel_size=3)

    class Parent:
        head = source.convs
        norm = source.norms
        scale = source.scales

    old_output = destination.offset_head.conv.weight.detach().clone()
    initialize_from_trident_regression_trunk(destination, Parent())
    for left, right in zip(destination.convs, source.convs):
        assert torch.equal(left.conv.weight, right.conv.weight)
    assert torch.equal(destination.offset_head.conv.weight, old_output)
