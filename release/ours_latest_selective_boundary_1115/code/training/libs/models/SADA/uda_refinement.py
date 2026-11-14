"""时频语义门控运输与跨尺度选择性域蒸馏。Prototype/原D损失保持。

语义门控是运输计划的软筛选，不宣称求解最优部分OT。
域标签仅表示数据来自源/目标，不是目标动作GT。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .uda_routes import MultiscaleTimeFrequencyAlignment, MultiscaleDiscriminatorAggregation, TF_V2_SPEC, MDA_V2_SPEC, sample

TF_V3_SPEC=dict(TF_V2_SPEC,revision=3,sampling='rotating_strata',semantic_cost=1.)
MDA_V3_SPEC=dict(MDA_V2_SPEC,revision=3,aggregation_schedule='selective_distillation',distillation_weight=.1,
                 minimum_teacher_gain=.01,distillation_interval=20)


def rotating_starts(valid,width,count,step):
    """各时间分层取一个窗口；计数随checkpoint恢复，不消耗主RNG。"""
    last=valid-width
    if last<=0:return [0.]
    phase=(step*.6180339887498949+.5)%1.
    return [(i+phase)/count*last for i in range(count)]


def semantic_transport(x,y,sq,tq,spec):
    """特征成本反传；教师、语义门控、运输计划均detach。未知不作背景。"""
    cost=torch.cdist(x,y).square()
    with torch.no_grad():
        sq=sq.detach().clamp(0,1);tq=tq.detach().clamp(0,1)
        confidence=sq.max(-1).values[:,None]*tq.max(-1).values[None,:]
        affinity=F.normalize(sq,p=1,dim=-1)@F.normalize(tq,p=1,dim=-1).T
        semantic_gate=affinity*confidence
        kernel=-(cost.detach()+spec['semantic_cost']*(1-affinity))/spec['sinkhorn_epsilon']
        u=cost.new_zeros(len(x));v=cost.new_zeros(len(y))
        for _ in range(spec['sinkhorn_iterations']):
            u=-math.log(len(x))-torch.logsumexp(kernel+v[None],1)
            v=-math.log(len(y))-torch.logsumexp(kernel+u[:,None],0)
        plan=(kernel+u[:,None]+v[None]).exp()*semantic_gate
    # 不按剩余质量重新归一化，防止极少量低置信候选被放大成完整对齐。
    return (plan*cost).sum(),plan.sum().detach()


class SemanticTimeFrequencyAlignment(MultiscaleTimeFrequencyAlignment):
    def __init__(self,channels,input_channels,classes,settings):
        super().__init__(channels,input_channels,classes,dict(TF_V3_SPEC,**settings))

    def semantic_windows(self,model,videos,features,semantics):
        windows=[];refs=[];levels=[];semantic=[]
        ratios=torch.linspace(0,1,self.spec['length'],device=features[0].device)
        for batch,video in enumerate(videos):
            for level,(feature,stride) in enumerate(zip(features,model.rq3_capacity.strides)):
                valid=min(feature.shape[-1],math.ceil(video['feats'].shape[-1]/stride))
                if valid<4:continue
                width=min(valid,self.spec['length'])
                for start in rotating_starts(valid,width,self.spec['windows_per_video_scale'],int(self.calls)):
                    positions=start+(width-1)*ratios
                    windows.append(sample(feature[batch],positions))
                    refs.append(self.raw_projection@sample(video['feats'].detach(),positions*stride))
                    # 输入是B,T,C。源GT保留重叠类别；目标来自既有检测器伪标签分支，分数停止梯度；权重仍随检测器学习。
                    semantic.append(sample(semantics[level][batch].detach().T,positions).amax(-1))
                    levels.append(level)
        if not windows:return None,None,None,None
        return torch.stack(windows),torch.stack(refs),torch.tensor(levels,device=features[0].device),torch.stack(semantic)

    def forward(self,model,source_videos,target_videos,source_features,target_features,epoch,source_semantics=None,target_semantics=None):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos):raise ValueError('目标训练GT必须为空')
        zero=source_features[0].sum()*0
        if epoch<self.spec['start_epoch']:return zero,dict(active=zero.detach(),aligned_groups=zero.detach())
        if source_semantics is None or target_semantics is None:raise ValueError('缺少源GT或停止梯度的目标软语义')
        sw,sr,sl,sq=self.semantic_windows(model,source_videos,source_features,source_semantics)
        tw,tr,tl,tq=self.semantic_windows(model,target_videos,target_features,target_semantics)
        recons=[];sz=tz=None
        if sw is not None:
            sz,pred=self.encode(sw);recons.append(F.mse_loss(pred,sr)/sr.square().mean().clamp_min(1e-4))
        if tw is not None:
            tz,pred=self.encode(tw);recons.append(F.mse_loss(pred,tr)/tr.square().mean().clamp_min(1e-4))
        terms=[];masses=[]
        if sz is not None and tz is not None:
            for level in sl.unique().tolist():
                a=sl==level;b=tl==level
                if b.any():
                    loss,mass=semantic_transport(sz[a],tz[b],sq[a],tq[b],self.spec);terms.append(loss);masses.append(mass)
        align=torch.stack(terms).mean() if terms else zero
        rec=torch.stack(recons).mean() if recons else zero
        ramp=min(1.,(epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        if self.training:self.calls.add_(1)
        scalar=lambda x:zero.detach()+x
        return ramp*(self.spec['alignment_weight']*align+self.spec['reconstruction_weight']*rec),dict(
            active=scalar(1),alignment=align.detach(),reconstruction=rec.detach(),aligned_groups=scalar(len(terms)),
            transport_mass=torch.stack(masses).mean() if masses else zero.detach(),
            source_windows=scalar(0 if sw is None else len(sw)),target_windows=scalar(0 if tw is None else len(tw)),
            target_batch_nonempty=scalar(tw is not None),ramp=scalar(ramp))


def choose_teacher(fit,validation,receiver,margin):
    """按第一份样本选教师，第二份只验证；失败保留本尺度自身。"""
    candidate=int(fit.argmin())
    if candidate==receiver:return None
    if fit[candidate]+margin<fit[receiver] and validation[candidate]+margin<validation[receiver]:return candidate
    return None


class SelectiveDomainDistillation(MultiscaleDiscriminatorAggregation):
    def __init__(self,levels,settings):
        super().__init__(levels,dict(MDA_V3_SPEC,**settings))
        self.register_buffer('distillation_events',torch.zeros((),dtype=torch.long))
        self.register_buffer('last_selected',torch.zeros(()))

    def distillation_loss(self,discriminators,epoch):
        zero=next(discriminators[0].parameters()).sum()*0
        self.last_selected.zero_()
        if epoch<self.spec['start_epoch'] or (int(self.calls)+1)%self.spec['distillation_interval']:return zero
        modes=[[m.training for m in d.modules()] for d in discriminators]
        for d in discriminators:d.eval()
        terms=[]
        try:
            for level,student in enumerate(discriminators):
                groups=[(x.detach(),label,w) for l,x,label,w in self.cache if l==level and len(x)>=6]
                if {label for x,label,w in groups}!={0,1}:continue
                # 每个类别/域代表集合分成不重叠的选择、验证、蒸馏三份。
                def risks(offset):
                    result=[]
                    with torch.no_grad():
                        for d in discriminators:
                            domains=[]
                            for domain in (0,1):
                                losses=[]
                                for x,label,w in groups:
                                    if label!=domain:continue
                                    logits=d(x[offset::3]).flatten()
                                    losses.append(w*F.binary_cross_entropy_with_logits(logits,torch.full_like(logits,float(label))))
                                domains.append(torch.stack(losses).mean())
                            result.append(torch.stack(domains).mean())
                    return torch.stack(result)
                teacher=choose_teacher(risks(0),risks(1),level,self.spec['minimum_teacher_gain'])
                if teacher is None:continue
                domains=[]
                for domain in (0,1):
                    losses=[]
                    for x,label,w in groups:
                        if label!=domain:continue
                        x=x[2::3]
                        with torch.no_grad():q=discriminators[teacher](x).flatten().sigmoid()
                        z=student(x).flatten()
                        # KL去除教师熵，仅学生D反传，不向教师或检测特征反传。
                        ce=F.binary_cross_entropy_with_logits(z,q)
                        entropy=F.binary_cross_entropy(q.clamp(1e-7,1-1e-7),q)
                        losses.append(w*(ce-entropy))
                    domains.append(torch.stack(losses).mean())
                terms.append(torch.stack(domains).mean())
            self.last_selected.fill_(len(terms))
            if terms:self.distillation_events.add_(1)
            return self.spec['distillation_weight']*torch.stack(terms).mean() if terms else zero
        finally:
            for d,flags in zip(discriminators,modes):
                for module,flag in zip(d.modules(),flags):module.training=flag

    @torch.no_grad()
    def after_step(self,discriminators,epoch):
        if epoch>=self.spec['start_epoch']:self.calls.add_(1)
        self.cache=[];self.last_parameter_change.zero_()
        # 仅清理缓存/记录时钟，绝不平均D参数或修改Adam矩。
