"""现有左右距离周围的离散边界分布，以及原视频坐标下的时间等变。

仅数学核心，尚未接入SADA训练或推理。均值初始等于原回归输出；
分布宽度不是经校准的不确定性，不能直接作为伪标签可信度。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class RelativeBoundaryDistribution(nn.Module):
    def __init__(self, channels, bins=17):
        super().__init__()
        if bins < 3 or bins % 2 != 1: raise ValueError('使用至少3个奇数bins')
        # 正距离atoms，uniform期望严格归一到1，不截断原回归的长距离。
        factors = torch.exp(torch.linspace(-math.log(4), math.log(4), bins))
        self.register_buffer('factors', factors/factors.mean())
        # 新模块构造不挪动原检测器的CPU初始化随机流。
        with torch.random.fork_rng(devices=[]):
            self.logits = nn.Conv1d(channels, 2*bins, 1)
            nn.init.zeros_(self.logits.weight); nn.init.zeros_(self.logits.bias)
        self.bins = bins

    def forward(self, features, original_offsets):
        # original_offsets: [B,2,T]，单位为该层stride；probability: [B,2,T,K]。
        batch, _, length = features.shape
        logits = self.logits(features).view(batch, 2, self.bins, length).permute(0,1,3,2)
        probability = logits.softmax(-1)
        multiplier = (probability*self.factors).sum(-1)
        offsets = original_offsets*multiplier
        return dict(offsets=offsets, logits=logits, probability=probability)


def source_distribution_loss(logits, original_offsets, assigned_gt_offsets, positive, factors):
    """仅源GT监督；超出有限atoms的真值采用删失尾部目标，并单列比例。

    参数距离均为[B,2,T]；positive为[B,T]。原offset分母detach，避免
    网络通过挪动label的bin来降低CE；真实距离均值仍需原DIoU监督。
    """
    ratio = assigned_gt_offsets/original_offsets.detach().clamp_min(1e-4)
    upper = torch.searchsorted(factors, ratio.contiguous()).clamp(1, len(factors)-1)
    lower = upper-1
    mix = ((ratio-factors[lower])/(factors[upper]-factors[lower])).clamp(0,1)
    logs = logits.log_softmax(-1)
    low_loss = -logs.gather(-1,lower.unsqueeze(-1)).squeeze(-1)
    high_loss = -logs.gather(-1,upper.unsqueeze(-1)).squeeze(-1)
    weight = positive[:,None,:].expand_as(ratio)
    count = weight.sum().clamp_min(1)
    loss = (((1-mix)*low_loss+mix*high_loss)*weight).sum()/count
    return dict(loss=loss, positive_points=positive.sum(),
        lower_tail_fraction=((ratio<factors[0])*weight).sum()/count,
        upper_tail_fraction=((ratio>factors[-1])*weight).sum()/count)


def absolute_boundary_support(original_offsets, points, strides, factors, crop_origin):
    """[N,2]距离 -> [N,2,K]端点位置，统一为原视频feature坐标。

    points/strides为原point_generator的坐标和stride，不以token序号替代。
    """
    distance = original_offsets[...,None]*strides[:,None,None]*factors
    signs = distance.new_tensor([-1.,1.])[None,:,None]
    return points[:,None,None]+crop_origin+signs*distance


def wasserstein_1d(first_position, first_probability, second_position, second_probability):
    """任意有序/无序离散支持上的精确W1，支持点和概率均可求导。

    最后一维是atoms；需要调用方决定哪一视图stop-gradient。
    """
    position = torch.cat((first_position,second_position),-1)
    mass = torch.cat((first_probability,-second_probability),-1)
    ordered, index = position.sort(-1)
    cumulative = mass.gather(-1,index).cumsum(-1)[...,:-1]
    return (cumulative.abs()*(ordered[...,1:]-ordered[...,:-1])).sum(-1)


def common_grid_indices(first_points, second_points, first_valid, second_valid, first_origin, second_origin):
    """只对应同一原视频的同层、相同绝对grid，不插值凑对应点。"""
    first = first_points+first_origin; second = second_points+second_origin
    if not len(first) or not len(second):
        empty = first_points.new_empty((0,),dtype=torch.long); return empty,empty
    index = torch.searchsorted(second,first).clamp(max=len(second)-1)
    same = torch.isclose(second[index],first,atol=1e-6,rtol=0.)
    good = same & first_valid.bool() & second_valid[index].bool()
    return torch.nonzero(good).flatten(),index[good]


def completely_visible(support, first_origin, first_length, second_origin, second_length):
    """整套端点atoms都需在两视图真实支持内，不能以均值可见冒充。"""
    left = max(first_origin,second_origin)
    right = min(first_origin+first_length,second_origin+second_length)
    return (support.amin(dim=(-1,-2))>=left) & (support.amax(dim=(-1,-2))<=right)


def temporal_distribution_loss(weak_support, weak_probability, strong_support, strong_probability, eligible, duration):
    # teacher只是同模型弱视图的停止梯度目标，没有额外教师模型。
    distance = wasserstein_1d(weak_support.detach(),weak_probability.detach(),strong_support,strong_probability)
    per_point = distance.mean(-1)/duration.detach().clamp_min(1.)
    return (per_point*eligible).sum()/eligible.sum().clamp_min(1)
