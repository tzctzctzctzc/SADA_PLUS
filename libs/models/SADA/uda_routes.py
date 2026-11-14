"""两条独立UDA适配；检测头、Prototype与数据顺序保持原版。

TF借鉴RAINCOAT(ICML23)的时频表征、Sinkhorn和原输入重构，非整篇复现；
MDA借鉴SOA(IJCV26)的跨尺度判别器参数聚合，不添加GCO。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

TF_SPEC=dict(enabled=True,kind='time_frequency',start_epoch=5,ramp_epochs=5,
             width=32,length=32,modes=8,windows_per_domain=24,bank_size=16,
             sinkhorn_epsilon=.05,sinkhorn_iterations=40,alignment_weight=.1,
             reconstruction_weight=.1,initialization_seed=11062026)
MDA_SPEC=dict(enabled=True,kind='mda',start_epoch=5,representatives=32,
              accuracy_trigger=.5,optimizer_moments='retain_original',measurement='eval_dropout_off')
TF_V2_SPEC=dict(TF_SPEC,revision=2,sampling='dense_per_scale',windows_per_video_scale=2)
MDA_V2_SPEC=dict(MDA_SPEC,revision=2,aggregation_schedule='periodic',aggregation_interval=20,mixing=.1)


def sample(sequence,positions):
    positions=positions.clamp(0,sequence.shape[-1]-1)
    left=positions.floor().long();right=(left+1).clamp(max=sequence.shape[-1]-1)
    weight=(positions-left)[None]
    return sequence[:,left]*(1-weight)+sequence[:,right]*weight


def sinkhorn_cost(x,y,epsilon=.05,iterations=40):
    cost=torch.cdist(x,y).square()
    # 计划用detach成本求解；反传真实配对代价，不能把整个OT损失detach。
    with torch.no_grad():
        kernel=-cost.detach()/epsilon
        u=cost.new_zeros(len(x));v=cost.new_zeros(len(y))
        for _ in range(iterations):
            u=-math.log(len(x))-torch.logsumexp(kernel+v[None],dim=1)
            v=-math.log(len(y))-torch.logsumexp(kernel+u[:,None],dim=0)
        plan=(kernel+u[:,None]+v[None]).exp()
    # 包含熵项，交叉减两项自身运输构成去偏Sinkhorn散度。
    return (plan*cost).sum()+epsilon*(plan*(plan.clamp_min(1e-30).log()+math.log(len(x)*len(y)))).sum()


def sinkhorn_divergence(x,y,spec):
    args=(spec['sinkhorn_epsilon'],spec['sinkhorn_iterations'])
    return sinkhorn_cost(x,y,*args)-.5*sinkhorn_cost(x,x,*args)-.5*sinkhorn_cost(y,y,*args)


class TimeFrequencyAlignment(nn.Module):
    def __init__(self,channels,input_channels,classes,settings):
        super().__init__();self.spec=dict(TF_SPEC,**settings);s=self.spec;w=s['width'];m=s['modes'];length=s['length']
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(s['initialization_seed'])
            self.temporal=nn.Sequential(nn.Conv1d(channels,w,3,padding=1),nn.GELU(),nn.Conv1d(w,w,3,padding=1),nn.GELU())
            self.spectral_real=nn.Parameter(torch.randn(channels,w,m)/math.sqrt(channels))
            self.spectral_imag=nn.Parameter(torch.randn(channels,w,m)/math.sqrt(channels))
            self.decoder=nn.Linear(w*(4+2*m),w*length)
            projection=torch.randn(w,input_channels)/math.sqrt(input_channels)
        self.register_buffer('raw_projection',projection)
        dimension=w*(4+2*m)
        self.register_buffer('source_bank',torch.zeros(classes+1,s['bank_size'],dimension))
        self.register_buffer('bank_counts',torch.zeros(classes+1,dtype=torch.long))
        self.register_buffer('bank_cursor',torch.zeros(classes+1,dtype=torch.long))
        self.register_buffer('calls',torch.zeros((),dtype=torch.long))

    def encode(self,windows):
        temporal=F.adaptive_avg_pool1d(self.temporal(windows),4).flatten(1)
        fft=torch.fft.rfft(windows,norm='ortho')[...,:self.spec['modes']]
        weights=torch.complex(self.spectral_real,self.spectral_imag)
        spectral=torch.einsum('ncm,cwm->nwm',fft,weights)
        # 连续实虚部避免零幅度处phase角不稳定；仍保留频率幅值与相位信息。
        feature=torch.cat((temporal,spectral.real.flatten(1),spectral.imag.flatten(1)),dim=1)
        normalized=F.normalize(feature,dim=-1)
        return normalized,self.decoder(feature).reshape(len(windows),self.spec['width'],self.spec['length'])

    def windows(self,model,videos,features,source):
        from .prototype_capacity import source_gt_proposals,fully_visible_cache,cached_proposals_for_videos
        proto=model.rq3_capacity;classes=model.num_class_orig
        if source:
            proposals=source_gt_proposals(videos,classes,len(features),proto.regression_ranges)
        else:
            if any(v['segments'].numel() or v['labels'].numel() for v in videos):raise ValueError('TF目标TRAIN出现GT')
            kept,_,_=fully_visible_cache(proto.cache,videos,proto.confidence_threshold)
            proposals=[]
            for i,video in enumerate(videos):
                proposals.extend(cached_proposals_for_videos({str(video['metadata']['video_id']):kept[i]},[video],
                    classes,len(features),features[0].device,features[0].dtype))
        rows=[]
        for batch,(video,proposal) in enumerate(zip(videos,proposals)):
            segments=proposal['segments'];labels=proposal.get('targets',proposal['evidence'])
            for i,segment in enumerate(segments):
                evidence=labels[i]
                if source:category=int(evidence[:classes].argmax()) if evidence[:classes].sum()>0 else None
                elif evidence[:classes].sum()>0:category=int(evidence[:classes].argmax())
                elif evidence[classes+1]>0:category=classes
                else:category=None # 未知和弃权不当背景。
                if category is not None:rows.append((batch,segment,int(proposal['levels'][i]),category))
            if source:
                # 只有源GT能确定额外背景窗口；不把未知目标区域补成背景。
                length=video['feats'].shape[-1]
                for start in torch.linspace(0,max(0,length-32),5).tolist():
                    end=min(length,start+32)
                    if not len(segments) or not ((segments[:,0]<end)&(segments[:,1]>start)).any():
                        rows.append((batch,segments.new_tensor([start,end]),0,classes))
        if len(rows)>self.spec['windows_per_domain']:
            indices=torch.linspace(0,len(rows)-1,self.spec['windows_per_domain']).long().tolist();rows=[rows[i] for i in indices]
        windows=[];references=[];categories=[]
        ratios=torch.linspace(0,1,self.spec['length'],device=features[0].device)
        for batch,segment,level,category in rows:
            start,end=segment.detach();span=float(end-start)
            if span<4:continue
            while level>0 and span/proto.strides[level]<4:level-=1
            positions=start+(end-start)*ratios
            windows.append(sample(features[level][batch],positions/proto.strides[level]))
            # 重构目标来自固定的原始输入投影，而非随网络更新的FPN自身。
            raw=sample(videos[batch]['feats'].detach(),positions)
            references.append(self.raw_projection@raw);categories.append(category)
        if not windows:return None,None,None
        return torch.stack(windows),torch.stack(references),torch.tensor(categories,device=features[0].device)

    def forward(self,model,source_videos,target_videos,source_features,target_features,epoch):
        zero=source_features[0].sum()*0
        if epoch<self.spec['start_epoch']:return zero,dict(active=zero.detach(),aligned_groups=zero.detach())
        sw,sr,sc=self.windows(model,source_videos,source_features,True)
        tw,tr,tc=self.windows(model,target_videos,target_features,False)
        sz=tz=None;recons=[]
        if sw is not None:
            sz,reconstruction=self.encode(sw)
            recons.append(F.mse_loss(reconstruction,sr.detach())/sr.detach().square().mean().clamp_min(1e-4))
        if tw is not None:
            tz,reconstruction=self.encode(tw)
            recons.append(F.mse_loss(reconstruction,tr.detach())/tr.detach().square().mean().clamp_min(1e-4))
        terms=[]
        if tz is not None:
            for category in tc.unique().tolist():
                current=sz[sc==category] if sz is not None else tz[:0]
                count=int(self.bank_counts[category]);bank=self.source_bank[category,:count].detach().clone()
                reference=torch.cat((current,bank),0)
                if len(reference):terms.append(sinkhorn_divergence(reference,tz[tc==category],self.spec))
        with torch.no_grad():
            if sz is not None:
                for value,category in zip(sz.detach(),sc.tolist()):
                    cursor=int(self.bank_cursor[category]);self.source_bank[category,cursor].copy_(value)
                    self.bank_cursor[category]=(cursor+1)%self.spec['bank_size']
                    self.bank_counts[category]=min(self.spec['bank_size'],int(self.bank_counts[category])+1)
            self.calls.add_(1)
        alignment=torch.stack(terms).mean() if terms else zero
        reconstruction=torch.stack(recons).mean() if recons else zero
        ramp=min(1.,(epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        total=ramp*(self.spec['alignment_weight']*alignment+self.spec['reconstruction_weight']*reconstruction)
        scalar=lambda x:zero.detach()+x
        return total,dict(active=scalar(1),alignment=alignment.detach(),reconstruction=reconstruction.detach(),
            aligned_groups=scalar(len(terms)),source_windows=scalar(0 if sw is None else len(sw)),
            target_windows=scalar(0 if tw is None else len(tw)),ramp=scalar(ramp))


class MultiscaleDiscriminatorAggregation(nn.Module):
    def __init__(self,levels,settings):
        super().__init__();self.spec=dict(MDA_SPEC,**settings);self.cache=[]
        self.register_buffer('calls',torch.zeros((),dtype=torch.long))
        self.register_buffer('aggregations',torch.zeros((),dtype=torch.long))
        self.register_buffer('last_accuracy',torch.ones(levels))
        self.register_buffer('last_weights',torch.ones(levels)/levels)
        self.register_buffer('last_parameter_change',torch.zeros(()))

    def observe(self,level,features,domain,weight):
        if not len(features):return
        indices=torch.linspace(0,len(features)-1,min(len(features),self.spec['representatives']),device=features.device).long()
        self.cache.append((level,features.detach()[indices].clone(),domain,weight))

    @torch.no_grad()
    def after_step(self,discriminators,epoch):
        from torch.nn.utils.stateless import functional_call
        rows=self.cache;self.cache=[]
        self.last_parameter_change.zero_()
        if epoch<self.spec['start_epoch'] or not rows:return
        self.calls.add_(1)
        periodic=self.spec.get('aggregation_schedule')=='periodic'
        if periodic and int(self.calls)%self.spec['aggregation_interval']:return
        modes=[[m.training for m in d.modules()] for d in discriminators]
        for d in discriminators:d.eval()
        try:
            states=[{k:v.detach().clone() for k,v in d.state_dict().items()} for d in discriminators]
            def measure(weights=None):
                losses=[];accuracies=[]
                for level,d in enumerate(discriminators):
                    domain_loss=[];domain_accuracy=[]
                    for domain in (0,1):
                        ll=[];aa=[]
                        for l,features,label,weight in rows:
                            if l!=level or label!=domain:continue
                            logits=(d(features) if weights is None else functional_call(d,weights,(features,))).reshape(-1)
                            ll.append(F.binary_cross_entropy_with_logits(logits,torch.full_like(logits,float(domain)))*weight)
                            aa.append(((logits>=0)==bool(domain)).float().mean())
                        if not ll:break
                        domain_loss.append(torch.stack(ll).mean());domain_accuracy.append(torch.stack(aa).mean())
                    if len(domain_loss)!=2:return None,None
                    losses.append(torch.stack(domain_loss).mean());accuracies.append(torch.stack(domain_accuracy).mean())
                return torch.stack(losses),torch.stack(accuracies)
            local,accuracy=measure()
            if local is None:return
            self.last_accuracy.copy_(accuracy)
            periodic=self.spec.get('aggregation_schedule')=='periodic'
            if periodic:
                if int(self.calls)%self.spec['aggregation_interval']:return
            elif not (accuracy<self.spec['accuracy_trigger']).all():return
            average={k:torch.stack([s[k] for s in states]).mean(0) if v.is_floating_point() else v
                     for k,v in states[0].items()}
            global_loss,_=measure(average)
            weights=(local-global_loss).softmax(0);self.last_weights.copy_(weights)
            shared={k:sum(weights[i]*s[k] for i,s in enumerate(states)) if v.is_floating_point() else v
                    for k,v in states[0].items()}
            mixing=self.spec['mixing'] if periodic else 1.
            updated=[{k:(1-mixing)*s[k]+mixing*v if v.is_floating_point() else v for k,v in shared.items()} for s in states]
            change=sum((states[i][k]-value).square().sum() for i,u in enumerate(updated)
                       for k,value in u.items() if value.is_floating_point()).sqrt()
            for d,u in zip(discriminators,updated):d.load_state_dict(u,strict=True)
            self.last_parameter_change.copy_(change);self.aggregations.add_(1)
            # 按原实现保留Adam矩；不悄悄重置优化器。
        finally:
            for d,flags in zip(discriminators,modes):
                for module,flag in zip(d.modules(),flags):module.training=flag


class MultiscaleTimeFrequencyAlignment(TimeFrequencyAlignment):
    """按FPN尺度匹配无标签有效窗口，短动作不再被候选最小时长过滤。

    固定32个真实特征点的窗口（不足时用有效长度）；源/目标对称取样。
    无候选伪标签、无源类别bank；按尺度OT避免不同物理时长直接匹配。
    """
    def __init__(self,channels,input_channels,classes,settings):
        super().__init__(channels,input_channels,classes,dict(TF_V2_SPEC,**settings))
        del self.source_bank,self.bank_counts,self.bank_cursor

    def windows(self,model,videos,features,source):
        windows=[];references=[];levels=[]
        ratios=torch.linspace(0,1,self.spec['length'],device=features[0].device)
        for batch,video in enumerate(videos):
            raw_length=video['feats'].shape[-1]
            for level,(feature,stride) in enumerate(zip(features,model.rq3_capacity.strides)):
                valid=min(feature.shape[-1],math.ceil(raw_length/stride))
                if valid<4:continue # 整个尺度不足4点时跳过；不筛掉短动作。
                width=min(valid,self.spec['length']);last=valid-width
                starts=torch.linspace(0,last,self.spec['windows_per_video_scale']).unique().tolist()
                for start in starts:
                    positions=start+(width-1)*ratios
                    windows.append(sample(feature[batch],positions))
                    raw=sample(video['feats'].detach(),positions*stride)
                    references.append(self.raw_projection@raw);levels.append(level)
        if not windows:return None,None,None
        return torch.stack(windows),torch.stack(references),torch.tensor(levels,device=features[0].device)

    def forward(self,model,source_videos,target_videos,source_features,target_features,epoch):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos):raise ValueError('TF目标TRAIN出现GT')
        zero=source_features[0].sum()*0
        if epoch<self.spec['start_epoch']:return zero,dict(active=zero.detach(),aligned_groups=zero.detach())
        sw,sr,sl=self.windows(model,source_videos,source_features,True)
        tw,tr,tl=self.windows(model,target_videos,target_features,False)
        recons=[];sz=tz=None
        if sw is not None:
            sz,reconstruction=self.encode(sw)
            recons.append(F.mse_loss(reconstruction,sr)/sr.square().mean().clamp_min(1e-4))
        if tw is not None:
            tz,reconstruction=self.encode(tw)
            recons.append(F.mse_loss(reconstruction,tr)/tr.square().mean().clamp_min(1e-4))
        terms=[]
        if sz is not None and tz is not None:
            for level in sl.unique().tolist():
                if (tl==level).any():terms.append(sinkhorn_divergence(sz[sl==level],tz[tl==level],self.spec))
        alignment=torch.stack(terms).mean() if terms else zero
        reconstruction=torch.stack(recons).mean() if recons else zero
        ramp=min(1.,(epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        total=ramp*(self.spec['alignment_weight']*alignment+self.spec['reconstruction_weight']*reconstruction)
        self.calls.add_(1)
        scalar=lambda x:zero.detach()+x
        return total,dict(active=scalar(1),alignment=alignment.detach(),reconstruction=reconstruction.detach(),
            aligned_groups=scalar(len(terms)),source_windows=scalar(0 if sw is None else len(sw)),
            target_windows=scalar(0 if tw is None else len(tw)),target_batch_nonempty=scalar(tw is not None),ramp=scalar(ramp))
