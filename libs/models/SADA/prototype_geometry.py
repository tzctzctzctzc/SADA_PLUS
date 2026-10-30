"""同层真实检测点的边界离散度；只调节原型目标项，不学习目标端点。"""
import math
import torch

from .prototype_capacity import RQ3ProposalPrototype, VLMProposalCoAdaptationBranch

VARIANT = 'geometry_weighted'
SCHEMA = 'rq3-prototype-geometry-v1'
POLICY = dict(max_points=7, minimum_points=2, support='inside_proposal_valid_original_fpn',
              uncertainty='sqrt_mean_population_endpoint_variance_div_proposal_duration',
              weight='1/(1+u/source_median)', missing_evidence_weight=1.,
              local_weight_normalization=False, detach_geometry=True,
              target_gt_used=False, target_boundary_regression=False)


@torch.no_grad()
def boundary_dispersion(segment, level, batch_index, observation):
    """所有坐标均为crop局部特征时间；offset只乘一次真实point stride。"""
    grid = observation['points'][int(level)]
    offsets = observation['offsets'][int(level)][batch_index]  # T x 2
    mask = observation['masks'][int(level)][batch_index].reshape(-1).bool()
    if grid.ndim != 2 or grid.shape[1] != 4 or offsets.shape != (len(grid), 2):
        raise ValueError('几何观测形状不是T×4 grid和T×2 offset')
    if len(mask) != len(grid):
        raise ValueError('有效mask与真实grid不匹配')
    duration = segment[1] - segment[0]
    if not torch.isfinite(segment).all() or float(duration) <= 0:
        raise ValueError('非法proposal时间区间')
    eligible = mask & (grid[:, 0] >= segment[0]) & (grid[:, 0] <= segment[1])
    indices = eligible.nonzero(as_tuple=True)[0]
    available = len(indices)
    if available > POLICY['max_points']:
        # 选择真实grid索引，不对特征/预测插值，也不借用段外的7个原型token。
        choice = torch.linspace(0, available - 1, POLICY['max_points'], device=indices.device).round().long()
        indices = indices[choice]
    if len(indices) < POLICY['minimum_points']:
        return dict(uncertainty=None, count=len(indices), available=available,
                    indices=indices, endpoints=offsets.new_empty((0, 2)))
    chosen = offsets[indices].detach()
    if not torch.isfinite(chosen).all() or (chosen < 0).any():
        raise ValueError('真实回归输出非有限或距离为负')
    locations, strides = grid[indices, 0], grid[indices, 3]
    endpoints = torch.stack((locations - chosen[:, 0] * strides,
                             locations + chosen[:, 1] * strides), dim=-1)
    uncertainty = endpoints.var(dim=0, unbiased=False).mean().sqrt() / duration
    return dict(uncertainty=uncertainty, count=len(indices), available=available,
                indices=indices, endpoints=endpoints)


@torch.no_grad()
def proposal_weights(proposals, observation, tau):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError('源域尺度tau必须为正有限值')
    weights, uncertainties, counts = [], [], []
    for batch_index, proposal in enumerate(proposals):
        for segment, level in zip(proposal['segments'], proposal['levels']):
            row = boundary_dispersion(segment, int(level), batch_index, observation)
            u = row['uncertainty']
            weights.append(segment.new_tensor(1.) if u is None else 1. / (1. + u / tau))
            uncertainties.append(segment.new_tensor(-1.) if u is None else u)
            counts.append(row['count'])
    device = observation['offsets'][0].device
    if not weights:
        return dict(weights=torch.empty(0, device=device), uncertainty=torch.empty(0, device=device),
                    point_counts=torch.empty(0, device=device, dtype=torch.long))
    return dict(weights=torch.stack(weights).detach(), uncertainty=torch.stack(uncertainties).detach(),
                point_counts=torch.tensor(counts, device=device, dtype=torch.long))


class GeometryBranch(VLMProposalCoAdaptationBranch):
    def prototype_alignment(self, source, source_targets, target, selection):
        self._update_source_prototypes(source['embedding'], source_targets)
        selected = selection['selected'].clone()
        labels = selection['labels']
        selected &= self.source_prototype_ready[labels]
        zero = source['embedding'].sum() * 0. + target['embedding'].sum() * 0.
        if not selected.any():
            return zero, 0, zero.detach()
        weights = selection['geometry_weights'][selected].detach()
        similarity = (target['embedding'][selected] * self.source_prototypes[labels[selected]].detach()).sum(-1)
        # 分母仍为原实际对齐候选数；不除sum(q)，单候选时也能真实降权。
        return (weights * (1. - similarity)).mean(), int(labels[selected].unique().numel()), similarity.mean().detach()


class GeometryPrototype(RQ3ProposalPrototype):
    def __init__(self, *args, tau, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.branch = GeometryBranch(args[0], args[1], hidden_dim=256, embedding_dim=128,
                                         confidence_threshold=0., prototype_momentum=.9)
        self.geometry_tau = float(tau)

    def geometry_selection(self, target, selection, observation):
        if observation is None:
            raise ValueError('几何分支缺少本次普通前向的真实回归观测')
        result = proposal_weights(target, observation, self.geometry_tau)
        selection['geometry_weights'] = result['weights']
        used = selection['selected'] & self.branch.source_prototype_ready[selection['labels']]
        valid = used & (result['point_counts'] >= POLICY['minimum_points'])
        q = result['weights'][used]
        return dict(geometry_evidence=int(valid.sum()), geometry_missing=int((used & ~valid).sum()),
                    geometry_weight_mean=float(q.mean()) if q.numel() else 1.,
                    geometry_weighted_candidates=int((q < 1.).sum()),
                    geometry_uncertainty_mean=float(result['uncertainty'][valid].mean()) if valid.any() else 0.,
                    geometry_single_candidate_crop=sum(int(len(p['segments']) == 1) for p in target))
