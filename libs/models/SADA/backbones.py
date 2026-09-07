import torch
from torch import nn
from torch.nn import functional as F

from .blocks import (
    ActionFormerBlock, ConvBlock, LayerNorm, MaskedConv1D, SGPBlock,
    get_sinusoid_encoding,
)
from .models import register_backbone


@register_backbone("SGP")
class SGPBackbone(nn.Module):
    """
        A backbone that combines SGP layer with transformers
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            sgp_mlp_dim,  # the numnber of dim in SGP
            n_embd_ks,  # conv kernel size of the embedding network
            max_len,  # max sequence length
            arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
            scale_factor=2,  # dowsampling rate for the branch,
            with_ln=False,  # if to attach layernorm after conv
            path_pdrop=0.0,  # droput rate for drop path
            downsample_type='max',  # how to downsample feature in FPN
            sgp_win_size=[-1] * 6,  # size of local window for mha
            k=1.5,  # the K in SGP
            init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
            use_abs_pe=False,  # use absolute position embedding
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(sgp_win_size) == (1 + arch[2])
        self.arch = arch
        self.sgp_win_size = sgp_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) #/ (n_embd ** 0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                SGPBlock(n_embd, 1, 1, n_hidden=sgp_mlp_dim, k=k, init_conv_vars=init_conv_vars))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(SGPBlock(n_embd, self.sgp_win_size[1 + idx], self.scale_factor, path_pdrop=path_pdrop,
                                        n_hidden=sgp_mlp_dim, downsample_type=downsample_type, k=k,
                                        init_conv_vars=init_conv_vars))
        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))
        
        emb_net_output = x.clone()
        
        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert T <= self.max_len, "Reached max length."
            pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if T >= self.max_len:
                pe = F.interpolate(
                    self.pos_embd, T, mode='linear', align_corners=False)
            else:
                pe = self.pos_embd
            # add pe to x
            x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem network
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)


        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks, emb_net_output


@register_backbone("convTransformer")
class ActionFormerBackbone(nn.Module):
    """ActionFormer convolutional embedding plus local-attention pyramid."""

    def __init__(
            self,
            n_in,
            n_embd,
            n_head,
            n_embd_ks,
            max_len,
            arch=(2, 2, 5),
            mha_win_size=(9, 9, 9, 9, 9, 9),
            scale_factor=2,
            with_ln=False,
            attn_pdrop=0.0,
            proj_pdrop=0.0,
            path_pdrop=0.0,
            use_abs_pe=False,
    ):
        super().__init__()
        if len(arch) != 3 or len(mha_win_size) != 1 + arch[-1]:
            raise ValueError('invalid ActionFormer architecture or window sizes')
        self.max_len = max_len
        self.use_abs_pe = use_abs_pe
        self.relu = nn.ReLU(inplace=True)
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for index in range(arch[0]):
            self.embd.append(MaskedConv1D(
                n_in if index == 0 else n_embd,
                n_embd,
                n_embd_ks,
                stride=1,
                padding=n_embd_ks // 2,
                bias=not with_ln,
            ))
            self.embd_norm.append(LayerNorm(n_embd) if with_ln else nn.Identity())
        if use_abs_pe:
            encoding = get_sinusoid_encoding(max_len, n_embd) / (n_embd ** 0.5)
            self.register_buffer('pos_embd', encoding, persistent=False)
        self.stem = nn.ModuleList([
            ActionFormerBlock(
                n_embd, n_head,
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size[0],
            )
            for _ in range(arch[1])
        ])
        self.branch = nn.ModuleList([
            ActionFormerBlock(
                n_embd, n_head,
                n_ds_strides=(scale_factor, scale_factor),
                attn_pdrop=attn_pdrop,
                proj_pdrop=proj_pdrop,
                path_pdrop=path_pdrop,
                mha_win_size=mha_win_size[index + 1],
            )
            for index in range(arch[2])
        ])
        self.apply(self.__init_weights__)

    @staticmethod
    def __init_weights__(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        for conv, norm in zip(self.embd, self.embd_norm):
            x, mask = conv(x, mask)
            x = self.relu(norm(x))
        emb_net_output = x.clone()
        if self.use_abs_pe:
            if self.training and x.shape[-1] > self.max_len:
                raise ValueError('ActionFormer input exceeds maximum training length')
            position = self.pos_embd
            if not self.training and x.shape[-1] >= self.max_len:
                position = F.interpolate(
                    position, x.shape[-1], mode='linear', align_corners=False
                )
            x = x + position[:, :, :x.shape[-1]] * mask.to(x.dtype)
        for block in self.stem:
            x, mask = block(x, mask)
        out_feats = (x,)
        out_masks = (mask,)
        for block in self.branch:
            x, mask = block(x, mask)
            out_feats += (x,)
            out_masks += (mask,)
        return out_feats, out_masks, emb_net_output


@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """

    def __init__(
            self,
            n_in,  # input feature dimension
            n_embd,  # embedding dimension (after convolution)
            n_embd_ks,  # conv kernel size of the embedding network
            arch=(2, 2, 5),  # (#convs, #stem convs, #branch convs)
            scale_factor=2,  # dowsampling rate for the branch
            with_ln=False,  # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            if idx == 0:
                in_channels = n_in
            else:
                in_channels = n_embd
            self.embd.append(MaskedConv1D(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )
            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = tuple()
        out_masks = tuple()
        # 1x resolution
        out_feats += (x,)
        out_masks += (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
