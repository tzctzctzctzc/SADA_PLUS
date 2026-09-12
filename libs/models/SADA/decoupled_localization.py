"""Class-agnostic localization stream used by the source-only structure gate."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm, MaskedConv1D, Scale


class DirectLocalizationHead(nn.Module):
    """Predict direct left/right offsets and an independent IoU-quality logit."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        with_ln=True,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("direct localization head requires at least two layers")
        self.fpn_levels = int(fpn_levels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer in range(num_layers - 1):
            in_dim = input_dim if layer == 0 else hidden_dim
            self.convs.append(
                MaskedConv1D(
                    in_dim,
                    hidden_dim,
                    kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=not with_ln,
                )
            )
            self.norms.append(LayerNorm(hidden_dim) if with_ln else nn.Identity())
        self.offset_head = MaskedConv1D(
            hidden_dim, 2, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.quality_head = MaskedConv1D(
            hidden_dim, 1, kernel_size, stride=1, padding=kernel_size // 2
        )
        self.scales = nn.ModuleList([Scale() for _ in range(self.fpn_levels)])

    def forward(self, fpn_feats, fpn_masks):
        if len(fpn_feats) != self.fpn_levels or len(fpn_masks) != self.fpn_levels:
            raise ValueError("feature pyramid level count does not match the head")
        offsets = []
        quality_logits = []
        for level, (features, mask) in enumerate(zip(fpn_feats, fpn_masks)):
            hidden = features
            for conv, norm in zip(self.convs, self.norms):
                hidden, _ = conv(hidden, mask)
                hidden = F.relu(norm(hidden))
            raw_offsets, _ = self.offset_head(hidden, mask)
            # Softplus preserves a gradient for initially negative offsets while
            # enforcing the non-negative center-offset representation.
            offsets.append(F.softplus(self.scales[level](raw_offsets)))
            logits, _ = self.quality_head(hidden, mask)
            quality_logits.append(logits)
        return offsets, quality_logits


class BoundarySensitiveLocalizationNeck(nn.Module):
    """Build a localization-only representation from temporal state changes."""

    def __init__(self, channels, fpn_levels):
        super().__init__()
        self.fpn_levels = int(fpn_levels)
        self.input_norms = nn.ModuleList(
            [LayerNorm(channels) for _ in range(self.fpn_levels)]
        )
        self.mixers = nn.ModuleList(
            [
                MaskedConv1D(3 * channels, channels, 1, bias=False)
                for _ in range(self.fpn_levels)
            ]
        )
        self.temporal_convs = nn.ModuleList(
            [
                MaskedConv1D(
                    channels,
                    channels,
                    5,
                    padding=2,
                    groups=channels,
                    bias=False,
                )
                for _ in range(self.fpn_levels)
            ]
        )
        self.output_norms = nn.ModuleList(
            [LayerNorm(channels) for _ in range(self.fpn_levels)]
        )

    def forward(self, backbone_feats, fpn_masks):
        if len(backbone_feats) != self.fpn_levels:
            raise ValueError("backbone pyramid level count does not match the neck")
        outputs = []
        for level, (features, mask) in enumerate(zip(backbone_feats, fpn_masks)):
            base = self.input_norms[level](features)
            left = F.pad(base[..., 1:] - base[..., :-1], (1, 0))
            right = F.pad(base[..., :-1] - base[..., 1:], (0, 1))
            changes = torch.cat((base, left, right), dim=1)
            mixed, _ = self.mixers[level](changes, mask)
            mixed = F.relu(mixed)
            mixed, _ = self.temporal_convs[level](mixed, mask)
            outputs.append(self.output_norms[level](base + mixed))
        return outputs


class DecoupledLocalizationSystem(nn.Module):
    """Optional boundary-sensitive neck followed by the direct localization head."""

    def __init__(self, head, neck=None):
        super().__init__()
        self.neck = neck
        self.head = head

    def forward(self, features, masks):
        if self.neck is not None:
            features = self.neck(features, masks)
        return self.head(features, masks)


def initialize_from_trident_regression_trunk(head, trident_reg_head):
    """Reuse the parent's localization-specific temporal trunk and level scales."""

    if len(head.convs) != len(trident_reg_head.head):
        raise ValueError("parent and direct localization trunks have different depth")
    if len(head.scales) != len(trident_reg_head.scale):
        raise ValueError("parent and direct localization heads have different levels")
    for destination, source in zip(head.convs, trident_reg_head.head):
        destination.load_state_dict(source.state_dict())
    for destination, source in zip(head.norms, trident_reg_head.norm):
        destination.load_state_dict(source.state_dict())
    for destination, source in zip(head.scales, trident_reg_head.scale):
        destination.load_state_dict(source.state_dict())
    return head


def balanced_soft_quality_loss(logits, targets, positive_mask, valid_mask):
    """Balance assigned action points and background for soft IoU supervision."""

    logits = logits[valid_mask]
    targets = targets[valid_mask]
    positives = positive_mask[valid_mask]
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive_loss = losses[positives]
    negative_loss = losses[~positives]
    terms = []
    if positive_loss.numel():
        terms.append(positive_loss.mean())
    if negative_loss.numel():
        terms.append(negative_loss.mean())
    if not terms:
        return logits.sum() * 0.0
    return torch.stack(terms).mean()


def fuse_class_and_quality(class_probabilities, quality_logits, gamma):
    """Calibrate localization quality independently before class expansion."""

    quality = quality_logits.sigmoid().squeeze(-1)
    if gamma <= 0:
        return class_probabilities
    return class_probabilities * quality.clamp_min(1e-8).pow(float(gamma)).unsqueeze(-1)


def blend_localization_offsets(parent_offsets, direct_offsets, alpha):
    """Use a learned class-agnostic stream as a bounded Trident correction."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("boundary blend alpha must lie in [0, 1]")
    return (1.0 - alpha) * parent_offsets + alpha * direct_offsets
