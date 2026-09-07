import pytest
import torch

pytest.importorskip('einops')

from libs.models.SADA.backbones import ActionFormerBackbone


def test_actionformer_backbone_builds_masked_feature_pyramid():
    backbone = ActionFormerBackbone(
        n_in=8,
        n_embd=16,
        n_head=4,
        n_embd_ks=3,
        max_len=16,
        arch=(1, 1, 2),
        mha_win_size=(5, 5, 5),
        scale_factor=2,
        with_ln=True,
    )
    features = torch.randn(2, 8, 16)
    mask = torch.ones(2, 1, 16, dtype=torch.bool)
    mask[1, :, 12:] = False

    pyramid, masks, embedded = backbone(features, mask)

    assert [item.shape for item in pyramid] == [
        (2, 16, 16), (2, 16, 8), (2, 16, 4)
    ]
    assert [item.shape for item in masks] == [
        (2, 1, 16), (2, 1, 8), (2, 1, 4)
    ]
    assert embedded.shape == (2, 16, 16)
    for item, item_mask in zip(pyramid, masks):
        assert torch.count_nonzero(item * ~item_mask).item() == 0


def test_actionformer_backbone_rejects_even_attention_window():
    with pytest.raises(ValueError, match='window must be odd'):
        ActionFormerBackbone(
            n_in=8,
            n_embd=16,
            n_head=4,
            n_embd_ks=3,
            max_len=16,
            arch=(1, 1, 1),
            mha_win_size=(4, 4),
        )
