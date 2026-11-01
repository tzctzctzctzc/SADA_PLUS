"""实例条件化时序能量。正能量峰是同一实例的完整区间，不是类别相似度。

借鉴 https://arxiv.org/abs/1909.12297 的可微候选优化；这里采用源监督
区间质量/相对实例排序，并不冒充原论文的连续归一化密度估计。
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = 'prototype-temporal-instance-energy-v1'
SPEC = dict(schema=SCHEMA, seed=123, hidden=128, layers=2, heads=4,
    tokens_per_region=8, regions=5, dropout=0., source_epochs=50, patience=15,
    source_lr=.0003, source_group_batch=4, candidate_chunk=64,
    source_instances='all-GT-plus-top20-per-predicted-class-real-proposals',
    source_candidates='GT-near-middle-long-cross-instance-same-class-neighbour-background',
    joint_epochs=50, joint_energy_weight=.1, joint_start='mature-original-EMA-extra-adaptation',
    refinement_steps=10, starts_per_axis=5, retained_modes=4, step_size=.1,
    gain_margin=.02, conflict_margin=.02, conflict_iou=.5,
    domain_supervision='source-only; target original Prototype and adversarial protocol unchanged',
    teacher_cache='original-Qwen8B-40-of-200-no-new-query',
    score='negative-energy-not-normalized-continuous-density',
    original_class_scores_unchanged=True, target_gt=False)


def joint_decision(history):
    """额外适配须允许成熟epoch0胜出，不能强迫选一个更差的训练后模型。"""
    epochs=[x[0] for x in history]
    if epochs!=[0]+list(range(5,epochs[-1]+1,5)):raise ValueError('联合源验证历史缺失')
    best=float('inf');which=0
    for i,(epoch,value) in enumerate(history):
        if not math.isfinite(value):raise ValueError('源验证非有限')
        if value<best:best=value;which=epoch
        if i<len(history)-1 and epoch-which>=15:raise ValueError('错过首次合法停止')
    epoch=epochs[-1]
    return dict(epoch=epoch,best_epoch=which,best_loss=best,reason='source_patience_exhausted' if epoch-which>=15 else 'max_epochs' if epoch==50 else 'running')


def interval_iou(a, b):
    inter=(torch.minimum(a[...,1],b[...,1])-torch.maximum(a[...,0],b[...,0])).clamp_min(0)
    return inter/((a[...,1]-a[...,0])+(b[...,1]-b[...,0])-inter).clamp_min(1e-6)


def sample_tokens(features, masks, boxes, levels, strides, samples=8):
    """线性采样对FPN值和连续端点均可导；越界点mask，不伪造上下文。

    输入与原points一致的特征时间单位；同一框保持原层号。返回顺序是
    开始窗口/内部/结束窗口/左外部/右外部，每区域保留真实时间顺序。
    """
    n=len(boxes);channels=features[0].shape[1]
    output=features[0].new_zeros(n,5*samples,channels)
    valid=torch.zeros(n,5*samples,device=boxes.device,dtype=torch.bool)
    s,e=boxes.unbind(-1);d=(e-s).clamp_min(1e-6)
    endpoints=torch.stack([s-.25*d,s+.25*d,s,e,e-.25*d,e+.25*d,s-.5*d,s,e,e+.5*d],-1).reshape(n,5,2)
    u=torch.linspace(0,1,samples,device=boxes.device,dtype=boxes.dtype)
    coordinates=(endpoints[:,:,0,None]*(1-u)+endpoints[:,:,1,None]*u).reshape(n,-1)
    for level,f in enumerate(features):
        ids=(levels==level).nonzero(as_tuple=True)[0]
        if not len(ids):continue
        pos=coordinates[ids]/float(strides[level]);length=f.shape[-1]
        inside=(pos>=0)&(pos<=length-1)
        pos=pos.clamp(0,length-1);lo=pos.floor().long();hi=(lo+1).clamp_max(length-1)
        w=(pos-lo).unsqueeze(-1);seq=f[0].T
        take=masks[level][0].reshape(-1).bool()
        ok=inside&take[lo]&take[hi]
        output[ids]=(seq[lo]*(1-w)+seq[hi]*w)*ok[:,:,None]
        valid[ids]=ok
    return output,valid


class TemporalIntervalEnergy(nn.Module):
    def __init__(self,channels=512,classes=10,hidden=128):
        super().__init__();self.classes=classes;self.hidden=hidden
        self.projection=nn.Linear(channels,hidden)
        self.role=nn.Embedding(5,hidden);self.position=nn.Linear(2,hidden)
        self.category=nn.Embedding(classes,hidden)
        block=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(block,2)
        self.norm=nn.LayerNorm(hidden)
        self.score=nn.Sequential(nn.Linear(hidden*2+2,hidden),nn.GELU(),nn.Linear(hidden,1))
        self.accept=nn.Linear(hidden,1)
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)

    def encode(self,tokens,valid,labels):
        count=tokens.shape[1];per=count//5
        region=torch.arange(5,device=tokens.device).repeat_interleave(per)
        t=torch.linspace(-1,1,per,device=tokens.device).repeat(5)
        location=torch.stack([t,region.to(t.dtype)/4],-1)
        x=self.projection(tokens)+self.role(region)[None]+self.position(location)[None]
        cls=self.category(labels)[:,None]
        x=torch.cat([cls,x],1)
        padding=torch.cat([torch.zeros(len(x),1,device=x.device,dtype=torch.bool),~valid],1)
        return self.norm(self.encoder(x,src_key_padding_mask=padding)[:,0])

    def forward(self,anchor,anchor_valid,tokens,valid,delta,labels):
        # 同一实例的anchor与candidate有独立时序编码，不能省略anchor身份。
        a=self.encode(anchor,anchor_valid,labels)
        n,k=tokens.shape[:2]
        b=self.encode(tokens.reshape(-1,*tokens.shape[2:]),valid.reshape(-1,valid.shape[-1]),
                      labels[:,None].expand(-1,k).reshape(-1)).reshape(n,k,-1)
        score=self.score(torch.cat([a[:,None].expand(-1,k,-1),b,delta],-1)).squeeze(-1)
        return score,self.accept(a).squeeze(-1)


def candidate_group(anchor, truth, segments, labels, instance, length):
    """显式加入邻居/跨实例/背景候选；质量始终相对已匹配实例计算。

    别的真实实例不是全局背景。kind=5表示同类邻居，记录其真实身份；
    无匹配anchor通过accept监督拒绝，不强拉向随意选择的GT。
    """
    d=(anchor[1]-anchor[0]).clamp_min(1.);s,e=anchor
    axis=anchor.new_tensor([-.5,-.25,0,.25,.5]);a,b=torch.meshgrid(axis,axis,indexing='ij')
    boxes=anchor[None]+d*torch.stack([a.flatten(),b.flatten()],-1)
    kinds=torch.zeros(len(boxes),device=anchor.device,dtype=torch.long)
    extras=[anchor,truth,truth+truth.new_tensor([-.05,.05])*(truth[1]-truth[0]),
            torch.stack([truth[0]+.25*(truth[1]-truth[0]),truth[1]-.25*(truth[1]-truth[0])]),
            torch.stack([s-d,e+d]),torch.stack([s-2*d,s-d]),torch.stack([e+d,e+2*d])]
    kind=[1,2,2,3,4,6,6]
    if instance>=0 and len(segments)>1:
        same=(labels==labels[instance]).nonzero(as_tuple=True)[0]
        same=same[same!=instance]
        if len(same):
            near=same[((segments[same].mean(1)-truth.mean()).abs()).argmin()]
            extras += [segments[near],torch.stack([torch.minimum(truth[0],segments[near,0]),torch.maximum(truth[1],segments[near,1])])]
            kind += [5,4]
        other=torch.arange(len(segments),device=segments.device);other=other[other!=instance]
        near=other[(segments[other].mean(1)-truth.mean()).abs().argmin()]
        extras.append(torch.stack([torch.minimum(truth[0],segments[near,0]),torch.maximum(truth[1],segments[near,1])]))
        kind.append(4)
    boxes=torch.cat([boxes,torch.stack(extras)]);kinds=torch.cat([kinds,torch.tensor(kind,device=anchor.device)])
    boxes=boxes.clamp(0,float(length));legal=boxes[:,1]-boxes[:,0]>=1
    q=interval_iou(boxes,truth[None]) if instance>=0 else boxes.new_zeros(len(boxes))
    return boxes[legal],q[legal],kinds[legal]


def energy_loss(score,accept,quality,valid,matched):
    """同时学习实例可用性、绝对定位质量及有效候选的质量次序。

    质量差是显式源GT监督，不用下降的分布CE冒充MAP选框改善。
    """
    qloss=F.binary_cross_entropy_with_logits(score,quality,reduction='none')
    gap=quality[:,:,None]-quality[:,None,:]
    pairs=(valid[:,:,None]&valid[:,None,:])&(gap>.05)
    ranking=F.softplus(-(score[:,:,None]-score[:,None,:]))*gap.clamp_min(0)*pairs
    rank=ranking.sum((1,2))/((gap.clamp_min(0)*pairs).sum((1,2))).clamp_min(1e-6)
    fit=(qloss*valid).sum(1)/valid.sum(1).clamp_min(1)
    rejection=F.binary_cross_entropy_with_logits(accept,matched.to(score.dtype),reduction='none')
    return fit+rank+rejection


def score_intervals(head,features,masks,anchor,boxes,levels,labels,strides):
    at,av=sample_tokens(features,masks,anchor,levels,strides)
    n,k=boxes.shape[:2];lv=levels[:,None].expand(-1,k).reshape(-1)
    bt,bv=sample_tokens(features,masks,boxes.reshape(-1,2),lv,strides)
    bt=bt.reshape(n,k,*bt.shape[1:]);bv=bv.reshape(n,k,-1)
    delta=(boxes-anchor[:,None])/(anchor[:,1]-anchor[:,0]).clamp_min(1)[:,None,None]
    scores,accept=head(at,av,bt,bv,delta,labels)
    # 缺少区内或端点采样的区间不允许成为高能量峰。外侧越界是正常背景。
    inner=bv[:,:,:24].sum(-1)>=18
    valid=inner&((boxes[:,:,1]-boxes[:,:,0])>=1)
    return scores,accept,valid


def search(head,features,masks,anchors,levels,labels,strides,length):
    """5x5多起点→四个不同区间模式→10步可微优化；冲突峰拒绝移动。

    固定原分数和类别，仅改变NMS前边界；逐步回溯保证接受的score不降。
    对有限模式解释相对竞争，不声称全空间概率校准。
    """
    if not len(anchors):return anchors.clone(),dict(searched=0,moved=0,rejected=0)
    head.eval();n=len(anchors);d=(anchors[:,1]-anchors[:,0]).clamp_min(1)
    axis=anchors.new_tensor([-.5,-.25,0,.25,.5]);a,b=torch.meshgrid(axis,axis,indexing='ij')
    boxes=(anchors[:,None]+d[:,None,None]*torch.stack([a.flatten(),b.flatten()],-1)).clamp(0,float(length))
    with torch.no_grad():
        scores,accept,valid=score_intervals(head,features,masks,anchors,boxes,levels,labels,strides)
        oldscore=score_intervals(head,features,masks,anchors,anchors[:,None],levels,labels,strides)[0][:,0].sigmoid()
        order=scores.masked_fill(~valid,-torch.inf).argsort(1,descending=True)
        starts=anchors[:,None].expand(-1,SPEC['retained_modes'],-1).clone()
        for i in range(n):
            chosen=[]
            for j in order[i].tolist():
                if not valid[i,j]:continue
                if not chosen or (interval_iou(boxes[i,j],boxes[i,chosen])<.9).all():chosen.append(j)
                if len(chosen)==SPEC['retained_modes']:break
            if chosen:starts[i,:len(chosen)]=boxes[i,chosen]
    u=((starts-anchors[:,None])/d[:,None,None]).detach()
    lr=anchors.new_full((n,SPEC['retained_modes'],1),SPEC['step_size'])
    for _ in range(SPEC['refinement_steps']):
        with torch.enable_grad():
            u.requires_grad_(True);b0=(anchors[:,None]+d[:,None,None]*u).clamp(0,float(length))
            sc,_,ok=score_intervals(head,features,masks,anchors,b0,levels,labels,strides)
            grad=torch.autograd.grad((sc*ok).sum(),u)[0]
        with torch.no_grad():
            trial=(u+lr*grad.clamp(-1,1)).clamp(-1,1)
            proposal=(anchors[:,None]+d[:,None,None]*trial).clamp(0,float(length))
            ns,_,nv=score_intervals(head,features,masks,anchors,proposal,levels,labels,strides)
            take=nv&(ns>=sc);u=torch.where(take[:,:,None],trial,u.detach());lr=torch.where(take[:,:,None],lr,lr*.5)
    with torch.no_grad():
        final=(anchors[:,None]+d[:,None,None]*u).clamp(0,float(length))
        sc,_,ok=score_intervals(head,features,masks,anchors,final,levels,labels,strides)
        sc=sc.sigmoid().masked_fill(~ok,-1);j=sc.argmax(1);index=torch.arange(n,device=anchors.device);best=final[index,j]
        conflict=((interval_iou(best[:,None],final)<SPEC['conflict_iou'])&((sc[index,j,None]-sc)<SPEC['conflict_margin'])&ok).any(1)
        reject=conflict|(sc[index,j]<oldscore+SPEC['gain_margin'])|(accept.sigmoid()<.5)
        result=torch.where(reject[:,None],anchors,best)
    return result,dict(searched=n,moved=int((result!=anchors).any(1).sum()),rejected=int(reject.sum()),conflicting_modes=int(conflict.sum()))


def video_groups(features,masks,anchors,truths,levels,classes,instances,segments,gt_classes,strides,length):
    """一个视频的一组anchor，padding只屏蔽候选，不改变实例归属。"""
    boxes=[];quality=[];kinds=[]
    for a,t,instance in zip(anchors,truths,instances):
        b,q,k=candidate_group(a,t,segments,gt_classes,int(instance),length)
        boxes.append(b);quality.append(q);kinds.append(k)
    maximum=max(map(len,boxes));n=len(boxes)
    stack=anchors[:,None].expand(-1,maximum,-1).clone();q=anchors.new_zeros(n,maximum)
    padding=torch.zeros(n,maximum,device=anchors.device,dtype=torch.bool)
    kind=torch.full((n,maximum),-1,device=anchors.device,dtype=torch.long)
    for i,b in enumerate(boxes):
        stack[i,:len(b)]=b;q[i,:len(b)]=quality[i];padding[i,:len(b)]=True;kind[i,:len(b)]=kinds[i]
    return stack,q,padding,kind


def install_joint(ctx, head, ledger=None):
    """真实可微源FPN路径接入原训练；不重算backbone、不改原单epoch循环。

    新模块加入原两阶段生成器的现有decay/no_decay组，scheduler组数不变；
    原模型和EMA都注册同构模块，原clip/capture/strict restore继续覆盖它。
    """
    import json
    from pathlib import Path
    from tools.baseline_protocol import assert_no_target_gt
    from tools.prototype_target_cache import duration_levels
    ctx.model.add_module('temporal_energy',head)
    ctx.model_ema.module.add_module('temporal_energy',deepcopy(head))
    ctx.model.temporal_energy_state=dict(step=0,source_groups=0,gradient_checks=[])
    ctx.model_ema.module.temporal_energy_state=deepcopy(ctx.model.temporal_energy_state)
    for optimizer in [ctx.optimizer_gen_source,ctx.optimizer_gen_source_alignment]:
        for name,p in head.named_parameters():
            no_decay=name.endswith('bias') or ('norm' in name and name.endswith('weight'))
            optimizer.param_groups[int(no_decay)]['params'].append(p)
    captured=[];active=[False];original=ctx.model.forward
    def neck_hook(module,inputs,outputs):
        if active[0]:captured.append(outputs)
    ctx.model.neck_source.register_forward_hook(neck_hook)
    ctx.temporal_energy_state=ctx.model.temporal_energy_state
    def forward(*args,**kwargs):
        ctx.temporal_energy_state=ctx.model.temporal_energy_state
        training=kwargs.get('mode')=='train';captured.clear();active[0]=training
        try:result=original(*args,**kwargs)
        finally:active[0]=False
        if not training:return result
        assert_no_target_gt(kwargs['video_list_tgt']);assert captured
        features,masks=captured[0];losses=[];groups=0
        for video_index,v in enumerate(kwargs['video_list_src']):
            truth=v['segments'];labels=v['labels']
            if not len(truth):continue
            f=[x[video_index:video_index+1] for x in features];m=[x[video_index:video_index+1] for x in masks]
            # 轮换四种已声明的错误方向，覆盖每个crop内全部源实例，不用目标GT。
            d=(truth[:,1]-truth[:,0]).clamp_min(1);axis=truth.new_tensor([[-.2,0],[0,.2],[.2,-.2],[-.2,.2]])
            offsets=axis[(torch.arange(len(truth),device=truth.device)+ctx.temporal_energy_state['step'])%4]
            anchors=(truth+d[:,None]*offsets).clamp(0,v['feats'].shape[-1]);levels=duration_levels(anchors,ctx.model.reg_range)
            parts=[]
            for ids in torch.arange(len(truth),device=truth.device).split(SPEC['source_group_batch']):
                boxes,q,pad,_=video_groups(f,m,anchors[ids],truth[ids],levels[ids],labels[ids],ids,truth,labels,ctx.model.fpn_strides,v['feats'].shape[-1])
                score,accept,valid=score_intervals(head,f,m,anchors[ids],boxes,levels[ids],labels[ids],ctx.model.fpn_strides)
                parts.append(energy_loss(score,accept,q,pad&valid,torch.ones(len(ids),device=truth.device,dtype=torch.bool)).mean());groups+=len(ids)
            losses.append(torch.stack(parts).mean())
        if losses:
            loss=torch.stack(losses).mean();assert torch.isfinite(loss)
            if not ctx.temporal_energy_state['gradient_checks']:
                fg=torch.autograd.grad(loss,features[0],retain_graph=True,allow_unused=True)[0]
                norm=0. if fg is None else float(fg.detach().norm());assert math.isfinite(norm)
                # 零初始化末层第一步对FPN为0是数学事实；只记录首个真实非零事件。
                if norm>0:ctx.temporal_energy_state['gradient_checks'].append(dict(source_fpn_gradient_norm=norm,target_gt_used=False))
            result[0]['downstream_loss']=result[0]['downstream_loss']+SPEC['joint_energy_weight']*loss
            result[0]['temporal_energy_loss']=loss;result[0]['temporal_energy_groups']=loss.new_tensor(groups)
        ctx.temporal_energy_state['step']+=1;ctx.temporal_energy_state['source_groups']+=groups
        if ledger:
            with Path(ledger).open('a') as log:log.write(json.dumps(dict(step=ctx.temporal_energy_state['step'],source_videos=[x['metadata']['video_id'] for x in kwargs['video_list_src']],source_groups=groups,target_gt_used=False))+'\n')
        captured.clear();return result
    ctx.model.forward=forward
    return ctx
