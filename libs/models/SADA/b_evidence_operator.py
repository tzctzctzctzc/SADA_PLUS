"""统一描述的观测量，保留各时间bin之间的相对证据强弱。

源GT只负责类别成员权重；描述器观测使用与目标、读出同式的检测概率。
不增加检测头，不把A当边界教师。所有条件代码均停止梯度。
"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .b_enhancement import EnhancedAlignment
from .class_path_alignment import conditional_discrepancy
from .uda_refinement import rotating_starts


MODES = ('legacy', 'shared_conditional', 'gt_mass', 'shared_mass', 'shared_residual', 'shared_joint_path')


def joint_path(original, mass):
    """保留两条路径的独立坐标；不让相反方向的差异在相加时抵消。"""
    return F.normalize(torch.cat([original, mass], dim=-1), dim=-1)


def supplement_path(original, mass, ratio=.1):
    """训练与冻结缓存共用的路径补充；端点不作多余归一化。"""
    if ratio == 0:
        return original
    if ratio == 1:
        return mass
    return F.normalize((1-ratio)*original+ratio*mass, dim=-1)


def relation_kernel(a, b, bandwidths=(.5, 1., 2.)):
    distance = torch.cdist(a.detach(), b.detach()).square()
    return torch.stack([torch.exp(-distance / h) for h in bandwidths]).mean(0)


def conditioned_discrepancy(source, target, sw, tw, sa, ta, bandwidths, alpha):
    if alpha == 0:
        return conditional_discrepancy(source, target, sw, tw, bandwidths)
    modifiers = [(1-alpha)+alpha*relation_kernel(a, b, bandwidths)
                 for a, b in ((sa, sa), (ta, ta), (sa, ta))]
    sw, tw = sw.detach(), tw.detach()
    terms, strengths = [], []
    for c in range(sw.shape[1]):
        a, b = sw[:, c], tw[:, c]
        if float(a.sum()) <= 1e-6 or float(b.sum()) <= 1e-6:
            continue
        strength = torch.minimum(a.sum().clamp(max=1), b.sum().clamp(max=1))
        a, b = a/a.sum(), b/b.sum()
        parts = []
        for xs, xt in zip(source, target):
            x, y = xs[:, c], xt[:, c]
            def kernel(u, v):
                distance = torch.cdist(u, v).square()
                return torch.stack([torch.exp(-distance/h) for h in bandwidths]).mean(0)
            ss, tt, st = [kernel(u, v)*m for (u, v), m in
                          zip(((x, x), (y, y), (x, y)), modifiers)]
            parts.append((ss*a[:, None]*a[None]).sum() +
                         (tt*b[:, None]*b[None]).sum() -
                         2*(st*a[:, None]*b[None]).sum())
        terms.append(strength*torch.stack(parts).mean())
        strengths.append(strength)
    zero = target[0].sum()*0
    return (torch.stack(terms).mean() if terms else zero,
            torch.stack(strengths).mean() if strengths else sw.new_zeros(()), len(terms))


def prototype_codes(model, features, boxes_by_video):
    """每个域一次A编码，顺序严格按视频、层、窗口；无额外随机抽样。"""
    branch = model.rq3_capacity.branch
    if not branch.source_prototype_ready.all():
        return None
    proposals = []
    for boxes in boxes_by_video:
        radius = (boxes[:, 1]-boxes[:, 0]).clamp_min(1)/2
        ranges = model.rq3_capacity.regression_ranges
        levels = torch.full((len(boxes),), len(ranges)-1, device=boxes.device, dtype=torch.long)
        for level, (low, high) in reversed(list(enumerate(ranges))):
            levels[(radius >= low) & (radius <= high)] = level
        proposals.append(dict(segments=boxes, levels=levels))
    with torch.no_grad():
        embedding = branch.encode([x.detach() for x in features], proposals,
                                  model.rq3_capacity.strides)['embedding']
        score = embedding @ F.normalize(branch.source_prototypes.detach(), dim=-1).T
        return F.normalize(score-score.mean(-1, keepdim=True), dim=-1)


class EvidenceAlignment(EnhancedAlignment):
    def __init__(self, channels, input_channels, classes, settings):
        super().__init__(channels, input_channels, classes, settings)
        if self.variant != 'path8':
            raise ValueError('本次仅核验已work的path8，不同时更换滤波器或维数')
        self.evidence_mode = settings.get('b_evidence_mode', 'shared_mass')
        if self.evidence_mode not in MODES:
            raise ValueError(self.evidence_mode)
        self.semantic_alpha = float(settings.get('b_semantic_alpha', 0.))
        if not 0 <= self.semantic_alpha <= 1:
            raise ValueError('语义条件系数范围不合法')
        self.temporal_residual = float(settings.get('b_temporal_residual', .1))
        if not 0 <= self.temporal_residual <= 1:
            raise ValueError('时间证据补充比例必须在[0,1]')
        self.vlm_gate = bool(settings.get('b_vlm_gate', False))
        self.vlm_gate_threshold = float(settings.get('b_vlm_gate_threshold', .8))
        self.spec.update(b_evidence_mode=self.evidence_mode,
                         b_semantic_alpha=self.semantic_alpha,
                         b_temporal_residual=self.temporal_residual,
                         b_same_view=bool(settings.get('b_same_view', True)),
                         b_vlm_gate=self.vlm_gate,
                         b_vlm_gate_threshold=self.vlm_gate_threshold)

    @property
    def descriptor_dims(self):
        pool, path, frequency = super().descriptor_dims
        return (pool, 2*path, frequency) if self.evidence_mode == 'shared_joint_path' else (pool, path, frequency)

    def describe_windows(self, z, semantics, bands=None):
        if self.evidence_mode in ('legacy', 'shared_conditional'):
            return super().describe_windows(z, semantics, bands)
        if self.evidence_mode in ('shared_residual', 'shared_joint_path'):
            parts, member = super().describe_windows(z, semantics, bands)
            if self.evidence_mode == 'shared_residual' and self.temporal_residual == 0:
                return parts, member
            if bands is not None:
                raise ValueError('本次不更换频带')
            q = semantics.detach().clamp(0, 1)
            weighted = z[:, None]*q[:, :, None]
            mass = F.adaptive_avg_pool1d(weighted.reshape(-1, z.shape[1], z.shape[-1]), self.bins)
            mass = F.normalize(mass.reshape(z.shape[0], q.shape[1], -1), dim=-1)
            # 只补充路径，不改池化/FFT。0与1分别精确还原已有两个对照。
            parts[1] = (joint_path(parts[1], mass) if self.evidence_mode == 'shared_joint_path'
                        else supplement_path(parts[1], mass, self.temporal_residual))
            return parts, member
        if bands is not None:
            raise ValueError('本次不改变频带方案')
        q = semantics.detach().clamp(0, 1)
        n, d, t = z.shape
        weighted = z[:, None]*q[:, :, None]
        pooled = weighted.sum(-1)/q.sum(-1).clamp_min(1e-6)[:, :, None]
        # 不除每个bin内的q均值；归一化仍去掉整窗公共幅度，只保留相对时间质量。
        path = F.adaptive_avg_pool1d(weighted.reshape(n*q.shape[1], d, t), self.bins)
        path = path.reshape(n, q.shape[1], -1)  # channel-major D x bins
        window = torch.hann_window(t, periodic=False, device=z.device, dtype=z.dtype)
        frequency = torch.log1p(torch.fft.rfft(weighted*window, norm='ortho').abs().square()).flatten(2)
        return [F.normalize(v, dim=-1) for v in (pooled, path, frequency)], q.mean(-1)

    def evidence_rows(self, model, videos, features, observation, membership, semantic=False):
        strides = model.rq3_capacity.strides
        valid = [[min(x.shape[-1], math.ceil(v['feats'].shape[-1]/s)) for v in videos]
                 for x, s in zip(features, strides)]
        prep = self.prepare_maps(features, valid)
        # 一次采样保证观测与类别成员权重坐标完全相同。
        joined = [torch.cat([o.detach(), w.detach()], dim=-1) for o, w in zip(observation, membership)]
        raw = self.read_windows(videos, prep, joined, strides)
        if raw is None:
            return None
        z, both, bands, levels = raw
        classes = observation[0].shape[-1]
        q, weight = both[:, :classes], both[:, classes:]
        parts, _ = self.describe_windows(z, q, bands)
        codes = None
        if semantic:
            boxes_by_video = []
            ratios = torch.linspace(0, 1, 32, device=z.device)
            for batch, video in enumerate(videos):
                boxes = []
                for level, stride in enumerate(strides):
                    n = valid[level][batch]
                    if n < 4:
                        continue
                    width = min(n, 32)
                    for start in rotating_starts(n, width, 2, int(self.calls)):
                        p = start+(width-1)*ratios
                        boxes.append(torch.stack([p[0]*stride, p[-1]*stride]))
                boxes_by_video.append(torch.stack(boxes) if boxes else z.new_empty((0, 2)))
            assert sum(len(b) for b in boxes_by_video) == len(z)
            codes = prototype_codes(model, features, boxes_by_video)
        return dict(parts=parts, weight=weight.mean(-1), levels=levels, code=codes,
                    observation=q, projected=z)

    def forward(self, model, source_videos, target_videos, source_features, target_features,
                epoch, source_semantics=None, target_semantics=None,
                source_prediction_semantics=None):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos):
            raise ValueError('目标TRAIN GT必须为空')
        zero = target_features[0].sum()*0
        if epoch < self.spec['start_epoch']:
            return zero, dict(active=zero.detach())
        shared = self.evidence_mode.startswith('shared')
        if shared and source_prediction_semantics is None:
            raise ValueError('共享观测必须传入同次源前向的分类概率，禁止退回GT')
        observation = source_prediction_semantics if shared else source_semantics
        # A早期未ready时显式退回alpha0，源GT类别权重始终保留。
        use_semantic = self.semantic_alpha > 0 and bool(model.rq3_capacity.branch.source_prototype_ready.all())
        with torch.no_grad():
            source = self.evidence_rows(model, source_videos, source_features, observation,
                                        source_semantics, use_semantic)
        target = self.evidence_rows(model, target_videos, target_features, target_semantics,
                                    target_semantics, use_semantic)
        terms, strengths = [], []
        if source is not None and target is not None:
            for level in source['levels'].unique().tolist():
                a, b = source['levels'] == level, target['levels'] == level
                if not b.any():
                    continue
                loss, strength, count = conditioned_discrepancy(
                    [v[a] for v in source['parts']], [v[b] for v in target['parts']],
                    source['weight'][a], target['weight'][b],
                    source['code'][a] if use_semantic else None,
                    target['code'][b] if use_semantic else None,
                    self.spec['bandwidths'], self.semantic_alpha if use_semantic else 0.)
                if count:
                    terms.append(loss); strengths.append(strength)
        value = torch.stack(terms).mean() if terms else zero
        vlm_gate = zero.new_ones(())
        if self.vlm_gate and target_videos:
            from .vlm_gated_boundary import class_reliability
            cache=getattr(getattr(model,'rq3_capacity',None),'_moe_target_cache',None)
            values=[]
            for video in target_videos:
                rows=cache.get(str(video['metadata']['video_id']),()) if cache is not None else ()
                selected=[float(r.get('routing_score',0.)) for r in rows if r.get('participation_selected',False)]
                values.append(sum(selected)/len(selected) if selected else 0.)
            if values: vlm_gate=torch.as_tensor(values,device=zero.device,dtype=zero.dtype).mean().clamp(.25,1.)
            value=value*vlm_gate
        ramp = min(1., (epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        if self.training:
            self.calls.add_(1)
        return ramp*self.spec['alignment_weight']*value, dict(active=zero.detach()+1,
            alignment=value.detach(), aligned_groups=zero.detach()+len(terms), ramp=zero.detach()+ramp,
            evidence_strength=torch.stack(strengths).mean() if strengths else zero.detach(),
            vlm_gate=vlm_gate.detach(),
            semantic_ready=zero.detach()+int(use_semantic))

    def fingerprint(self):
        h = hashlib.sha256(super().fingerprint().encode())
        h.update(Path(__file__).read_bytes())
        h.update(json.dumps(self.spec, sort_keys=True).encode())
        return h.hexdigest()
