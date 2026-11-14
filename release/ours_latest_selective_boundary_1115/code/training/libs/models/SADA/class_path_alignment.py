"""类条件多尺度路径对齐；借鉴CPDA/RAINCOAT，不是两篇论文的完整复现。

任务适配：源多标签占用率、目标停止梯度软分数、局部时间路径与功率谱。
保留原SADA域对抗/Prototype；无辅助可学习编码器、无原输入重构。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .uda_routes import sample
from .uda_refinement import rotating_starts

CLASS_PATH_SPEC = dict(enabled=True, kind='time_frequency', revision=4,
    retain_discriminator=True, start_epoch=5, ramp_epochs=5, alignment_weight=.1,
    length=32, windows_per_video_scale=2, projection_dim=16, bins=4,
    bandwidths=[.5, 1., 2.], initialization_seed=11072026, source_anchor=False)


def path_features(windows, semantics, projection, bins=4):
    """每类分别保留局部时序；不将重叠标签归一为互斥类别。"""
    q = semantics.detach().clamp(0, 1)  # N,C,T
    z = torch.einsum('dc,nct->ndt', projection, windows)
    weighted = z[:, None] * q[:, :, None]  # N,K,D,T
    n, k, d, t = weighted.shape
    pooled = weighted.sum(-1) / q.sum(-1).clamp_min(1e-6)[:, :, None]
    path = F.adaptive_avg_pool1d(weighted.reshape(n*k, d, t), bins)
    occupancy = F.adaptive_avg_pool1d(q.reshape(n*k, 1, t), bins)
    # 空bin保持0，非空bin独立求均值；保留动作出现/消失的时间位置。
    path = (path / occupancy.clamp_min(1e-6)).reshape(n, k, -1)
    window = torch.hann_window(t, periodic=False, device=z.device, dtype=z.dtype)
    power = torch.fft.rfft(weighted * window, norm='ortho').abs().square()
    power = torch.log1p(power).flatten(2)
    return [F.normalize(v, dim=-1) for v in (pooled, path, power)], q.mean(-1)


def conditional_discrepancy(source, target, sw, tw, bandwidths):
    """按类别归一化权重，同时用有界证据量防止微弱伪标签被放大。"""
    terms=[]; strengths=[]
    sw=sw.detach();tw=tw.detach()
    for c in range(sw.shape[1]):
        a=sw[:, c];b=tw[:, c]
        if float(a.sum()) <= 1e-6 or float(b.sum()) <= 1e-6: continue
        # 有效窗口数和置信度仅控制一次强度，不乘1/C类别归一化亲和度。
        strength=torch.minimum(a.sum().clamp(max=1), b.sum().clamp(max=1))
        a=a/a.sum();b=b/b.sum()
        components=[]
        for xs, xt in zip(source, target):
            x=xs[:,c];y=xt[:,c]
            def kernel(u,v):
                distance=torch.cdist(u,v).square()
                return torch.stack([torch.exp(-distance/h) for h in bandwidths]).mean(0)
            value=(kernel(x,x)*a[:,None]*a[None,:]).sum()
            value=value+(kernel(y,y)*b[:,None]*b[None,:]).sum()
            value=value-2*(kernel(x,y)*a[:,None]*b[None,:]).sum()
            components.append(value)
        terms.append(strength*torch.stack(components).mean());strengths.append(strength)
    if not terms: return source[0].sum()*0, sw.new_zeros(()), 0
    return torch.stack(terms).mean(),torch.stack(strengths).mean(),len(terms)


class ClassConditionalPathAlignment(nn.Module):
    def __init__(self, channels, input_channels, classes, settings):
        super().__init__();self.spec=dict(CLASS_PATH_SPEC, **settings)
        g=torch.Generator().manual_seed(self.spec['initialization_seed'])
        projection=torch.randn(self.spec['projection_dim'],channels,generator=g)/math.sqrt(channels)
        self.register_buffer('projection',projection)
        self.register_buffer('calls',torch.zeros((),dtype=torch.long))

    def windows(self, model, videos, features, semantics):
        rows=[];weights=[];levels=[]
        ratios=torch.linspace(0,1,self.spec['length'],device=features[0].device)
        for b,video in enumerate(videos):
            for l,(f,stride) in enumerate(zip(features,model.rq3_capacity.strides)):
                valid=min(f.shape[-1],math.ceil(video['feats'].shape[-1]/stride))
                if valid<4:continue
                width=min(valid,self.spec['length'])
                for start in rotating_starts(valid,width,self.spec['windows_per_video_scale'],int(self.calls)):
                    positions=start+(width-1)*ratios
                    rows.append(sample(f[b],positions))
                    weights.append(sample(semantics[l][b].detach().T,positions))
                    levels.append(l)
        if not rows:return None,None,None
        return torch.stack(rows),torch.stack(weights),torch.tensor(levels,device=features[0].device)

    def forward(self, model, source_videos, target_videos, source_features, target_features,
                epoch, source_semantics=None, target_semantics=None):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos):
            raise ValueError('目标TRAIN GT必须为空')
        zero=source_features[0].sum()*0
        if epoch<self.spec['start_epoch']:return zero,dict(active=zero.detach())
        sw,sq,sl=self.windows(model,source_videos,source_features,source_semantics)
        tw,tq,tl=self.windows(model,target_videos,target_features,target_semantics)
        terms=[];strengths=[];counts=[]
        if sw is not None and tw is not None:
            # 可选单变量实验：源域路径只作当前batch的参考，匹配梯度只从目标特征进入。
            # 共享骨干仍会因目标更新改变，不能称为独立冻结源模型。
            sx,sa=path_features(sw.detach() if self.spec['source_anchor'] else sw,sq,self.projection,self.spec['bins'])
            tx,ta=path_features(tw,tq,self.projection,self.spec['bins'])
            for level in sl.unique().tolist():
                a=sl==level;b=tl==level
                if not b.any():continue
                loss,strength,count=conditional_discrepancy([x[a] for x in sx],[x[b] for x in tx],sa[a],ta[b],self.spec['bandwidths'])
                if count:terms.append(loss);strengths.append(strength);counts.append(count)
        align=torch.stack(terms).mean() if terms else zero
        ramp=min(1.,(epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        if self.training:self.calls.add_(1)
        scalar=lambda x:zero.detach()+x
        return ramp*self.spec['alignment_weight']*align,dict(active=scalar(1),alignment=align.detach(),
            reconstruction=scalar(0),aligned_groups=scalar(len(terms)),active_classes=scalar(sum(counts)),
            evidence_strength=torch.stack(strengths).mean() if strengths else zero.detach(),
            target_confidence=tq.max(1).values.mean().detach() if tq is not None else zero.detach(),ramp=scalar(ramp))
