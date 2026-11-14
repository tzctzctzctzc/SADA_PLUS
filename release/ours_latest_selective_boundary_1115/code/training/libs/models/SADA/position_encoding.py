# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.#
#  ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Positional encodings for the transformer.
"""
import math
import torch

'''
class SinusoidalPositionEncoding1D(torch.nn.Module):
    def __init__(self, n, feat_dim):
        super().__init__()
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if feat_dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(feat_dim))
        pe = torch.zeros(n, feat_dim)
        position = torch.arange(0, n).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, feat_dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / feat_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        self.pe = pe
'''

class SinusoidalPositionEncoding1D(torch.nn.Module):
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x: shape B x T x F
        b, t, f = x.shape

        # Assume the mask is all negative (since in this case there is not masking)
        mask = torch.zeros((b, t), dtype=torch.bool).to(x.device)

        not_mask = ~mask
        x_embed = not_mask.cumsum(1, dtype=torch.float32)  # B x T
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # B x T x C
        # n,t,c
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

        return pos_x + x


def build_position_encoding(feat_dim):
    return SinusoidalPositionEncoding1D(num_pos_feats = feat_dim, normalize=True)