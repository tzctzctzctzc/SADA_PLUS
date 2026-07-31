import math

import torch
from torch import nn

from .blocks import LayerNorm, MaskedConv1D


class AuxBoundaryHead(nn.Module):
    """Class-agnostic start/end evidence head over all FPN levels."""

    def __init__(
            self, input_dim, feat_dim, kernel_size=3, num_layers=2,
            with_ln=False, prior_prob=0.01):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for index in range(num_layers):
            in_dim = input_dim if index == 0 else feat_dim
            self.convs.append(MaskedConv1D(
                in_dim,
                feat_dim,
                kernel_size,
                stride=1,
                padding=kernel_size // 2,
                bias=not with_ln,
            ))
            self.norms.append(LayerNorm(feat_dim) if with_ln else nn.Identity())
        self.output = MaskedConv1D(
            feat_dim, 2, kernel_size, stride=1, padding=kernel_size // 2
        )
        bias = -math.log((1.0 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.output.conv.bias, bias)
        self.activation = nn.ReLU()

    def forward(self, fpn_feats, fpn_masks):
        if len(fpn_feats) != len(fpn_masks):
            raise ValueError("FPN features and masks must have equal lengths")
        starts, ends = [], []
        for feature, mask in zip(fpn_feats, fpn_masks):
            output = feature
            for conv, norm in zip(self.convs, self.norms):
                output, _ = conv(output, mask)
                output = self.activation(norm(output))
            logits, _ = self.output(output, mask)
            starts.append(logits[:, 0])
            ends.append(logits[:, 1])
        return starts, ends
