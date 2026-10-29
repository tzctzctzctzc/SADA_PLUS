"""保持原FPN/七位置，显式保留时间角色；不强迫注意力分布变尖。"""
import torch
from torch import nn
from torch.nn import functional as F

from .prototype_capacity import VLMProposalCoAdaptationBranch, _sample_sequence


def role_specification():
    return dict(variant='temporal_roles', embedding_dim=128, hidden_dim=256,
        layers=1, heads=4, dropout=0., pooling='masked_temporal_roles_mean_residual',
        sample_ratios=[-.25, 0., .25, .5, .75, 1., 1.25],
        role_groups=[[0], [1, 2], [3], [4, 5], [6]],
        role_features=['early-minus-middle', 'late-minus-middle', 'late-minus-early',
                       'left-context-minus-inside', 'right-context-minus-inside'],
        fpn_routing='unchanged_source_range_target_proposal_first_level',
        prototypes_per_class=1, prototype_momentum=.9, source_weight=1.,
        target_alignment_weight=.1, target_negative_supervision=False)


def sample_masked_tokens(features, proposals, strides, masks, ratios):
    """坐标均为crop内原特征单位；仅在原指定金字塔层采样。"""
    if masks is None:
        raise ValueError('必须提供实际FPN有效mask')
    rows, valid_rows = [], []
    for b, proposal in enumerate(proposals):
        for segment, level in zip(proposal['segments'], proposal['levels']):
            level = int(level)
            duration = (segment[1] - segment[0]).clamp(min=1.)
            positions = torch.stack([segment[0] + x * duration for x in ratios]) / float(strides[level])
            feat = features[level][b]
            mask = masks[level][b].reshape(1, -1).to(feat.dtype)
            valid = ((_sample_sequence(mask, positions)[0] >= 1.-1e-6)
                     & (positions >= 0) & (positions <= feat.shape[-1]-1))
            rows.append(_sample_sequence(feat, positions).T)
            valid_rows.append(valid)
    if not rows:
        return features[0].new_empty((0, len(ratios), features[0].shape[1])), \
               torch.empty((0, len(ratios)), dtype=torch.bool, device=features[0].device)
    return torch.stack(rows), torch.stack(valid_rows)


class TemporalRoleBranch(VLMProposalCoAdaptationBranch):
    groups = ((0,), (1, 2), (3,), (4, 5), (6,))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.classifier.in_features
        self.role_residual = nn.Linear(5*d, d, bias=False)
        # 初始为同mask的原均值路径；额外路径通过真实源分类/目标对齐学习。
        nn.init.zeros_(self.role_residual.weight)

    @staticmethod
    def masked_mean(tokens, valid):
        return (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)

    def encoded_tokens(self, rows, valid):
        tokens = self.input_projection(rows) + self.position_embedding[None]
        tokens = tokens.masked_fill(~valid[..., None], 0.)
        encoded = torch.zeros_like(tokens)
        active = valid.any(1)
        if active.any():
            encoded[active] = self.encoder(tokens[active], src_key_padding_mask=~valid[active])
        return encoded

    def fuse_roles(self, tokens, valid, disable_roles=False):
        mean = self.masked_mean(tokens, valid)
        values = [self.masked_mean(tokens[:, g], valid[:, g]) for g in self.groups]
        ready = [valid[:, g].any(1) for g in self.groups]
        left, early, middle, late, right = values
        inside = self.masked_mean(tokens[:, 1:6], valid[:, 1:6])
        inside_ready = valid[:, 1:6].any(1)
        # 缺一个角色时不凭空制造“差值”：只对同时有效的两端构造差异。
        differences = [(early-middle, ready[1] & ready[2]),
                       (late-middle, ready[3] & ready[2]),
                       (late-early, ready[3] & ready[1]),
                       (left-inside, ready[0] & inside_ready),
                       (right-inside, ready[4] & inside_ready)]
        role_input = torch.cat([x * m[:, None] for x, m in differences], dim=1)
        residual = self.role_residual(role_input)
        value = mean if disable_roles else mean + residual
        active = valid.any(1)
        embedding = self.output_norm(value) * active[:, None]
        return dict(embedding=F.normalize(embedding, dim=-1), logits=self.classifier(embedding),
                    valid=active, statistics=dict(
                        role_residual_norm=residual.norm(dim=1).mean().detach(),
                        role_base_norm=mean.norm(dim=1).mean().detach(),
                        role_early_late_valid=(ready[1] & ready[3]).float().mean().detach()))

    def encode_tokens(self, rows, valid, disable_roles=False):
        if not rows.shape[0]:
            z = rows.new_zeros((0, self.classifier.in_features))
            return dict(embedding=z, logits=z.new_zeros((0, self.num_classes)),
                        valid=valid.any(1), statistics={})
        return self.fuse_roles(self.encoded_tokens(rows, valid), valid, disable_roles)

    def encode_masked(self, fpn_feats, proposals, fpn_strides, fpn_masks):
        rows, valid = sample_masked_tokens(fpn_feats, proposals, fpn_strides, fpn_masks, self.sample_ratios)
        return self.encode_tokens(rows, valid)

    def source_loss(self, encoded, proposals):
        targets = self.flatten_field(proposals, 'targets')
        if targets is None or not targets.numel():
            return encoded['logits'].sum()*0.
        return super().source_loss(encoded, [dict(targets=targets*encoded['valid'][:, None])])
