"""OADA启发的相对时间位置条件化；这是待验证的迁移，不是已证实创新。"""
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'prototype-relative-phase-alignment-v1'
POLICY = dict(coordinate='left/(left+right)', source='assigned_source_gt_offsets',
              target='detached_current_predicted_offsets',
              roles='linear_interpolation_centres_0_0.5_1', background='uniform',
              conditioning='role_specific_discriminator_input_sectors',
              role_prior='same_uniform_common_roles_both_domains_normalized_within_role',
              minimum_role_mass_per_domain=1.,
              unsupported_class='original_domain_loss_with_uniform_condition',
              initialization='repeat_original_discriminator_first_weight',
              loss_weight='unchanged', target_gt_used=False)


def role_probabilities(offsets):
    """连续角色归属，不把几何归属概率称为校准uncertainty。"""
    offsets = offsets.detach()
    if offsets.shape[-1] != 2 or not torch.isfinite(offsets).all():
        raise ValueError('角色需要有限的左右距离')
    if (offsets < 0).any():
        raise ValueError('前景角色的左右距离不能为负')
    duration = offsets.sum(-1)
    # 零长度预测没有方向信息，按中点处理；不伪造左右端点。
    u = torch.where(duration > 0, offsets[..., 0] / duration.clamp_min(1e-12),
                    torch.full_like(duration, .5))
    q = (1 - (u[..., None] - u.new_tensor([0., .5, 1.])).abs() * 2).clamp_min(0)
    return q, u


def conditioned(features, roles):
    if roles.shape != (features.shape[0], 3):
        raise ValueError('角色与实际有效特征点未配对')
    return (roles.detach()[..., None] * features[:, None, :]).flatten(1)


def balanced_domain_losses(predict, source, target, qs, qt, background=False):
    """固定相同role先验，防止D仅靠两域role频率差异区分域。

    D确实看到不同输入sector；每个sector内按该域soft质量归一化。
    缺少跨域共同role或只有一个域时回退原class域损失，保留原路由。
    """
    common = (qs.sum(0) >= 1.) & (qt.sum(0) >= 1.)
    roles = common.nonzero().flatten().tolist() if not background else []
    if not roles:
        ps = predict(conditioned(source, source.new_full((len(source),3),1/3)))
        pt = predict(conditioned(target, target.new_full((len(target),3),1/3)))
        ls = F.binary_cross_entropy_with_logits(ps.flatten(),ps.new_zeros(ps.numel())) if ps.numel() else source.sum()*0
        lt = F.binary_cross_entropy_with_logits(pt.flatten(),pt.new_ones(pt.numel())) if pt.numel() else target.sum()*0
        return ls,lt,0
    ls,lt = source.sum()*0,target.sum()*0
    for role in roles:
        rs=source.new_zeros((len(source),3));rs[:,role]=1
        rt=target.new_zeros((len(target),3));rt[:,role]=1
        ps,pt=predict(conditioned(source,rs)).flatten(),predict(conditioned(target,rt)).flatten()
        ws,wt=qs[:,role].detach(),qt[:,role].detach()
        ls=ls+(F.binary_cross_entropy_with_logits(ps,torch.zeros_like(ps),reduction='none')*ws).sum()/ws.sum()
        lt=lt+(F.binary_cross_entropy_with_logits(pt,torch.ones_like(pt),reduction='none')*wt).sum()/wt.sum()
    return ls/len(roles),lt/len(roles),len(roles)


def expand_discriminators(model):
    """不消耗CPU/CUDA RNG；复制权重后任意q求和为1时初始函数相同。"""
    cfg = model.cfg['phase_alignment']
    if cfg != dict(schema=SCHEMA, policy=POLICY):
        raise ValueError('相对位置协议不是冻结版本')
    if model.use_trident_head or not model.cfg['model'].get('weight_sharing'):
        raise ValueError('首版仅适配原共享SADA普通左右回归头')
    for disc in model.disc_dom_conditional_adv:
        old = disc.initial_layers[0]
        wide = deepcopy(old)
        wide.in_features = old.in_features * 3
        wide.weight = nn.Parameter(old.weight.detach().repeat(1, 3))
        disc.initial_layers[0] = wide


def canonical_initial_state(state):
    """仅用于证明共同检测器冷初始化，不把训练后展开权重折叠。"""
    result = {}
    for key, value in state.items():
        if key.startswith('disc_dom_conditional_adv.') and key.endswith('initial_layers.0.weight'):
            parts = value.chunk(3, dim=1)
            if not (torch.equal(parts[0], parts[1]) and torch.equal(parts[0], parts[2])):
                raise ValueError('初始角色判别器不等价于原判别器')
            value = parts[0]
        result[key] = value
    return result


def expand_reference_state(state):
    """只读诊断加载原成熟教师；正式训练不走此入口。"""
    return {k: v.repeat(1, 3) if k.startswith('disc_dom_conditional_adv.') and
            k.endswith('initial_layers.0.weight') else v for k, v in state.items()}
