"""原Prototype时序表示的多尺度候选BG分支；仅在实验副本中联合更新。"""
import torch
from torch import nn
from tools.background_four_routes_math import ProposalEncoder,head_objective,source_symbols
from tools.background_linear_temporal import region_samples


class NativeBackground(nn.Module):
    def __init__(self,detector):
        super().__init__();self.backbone=detector.backbone_source;self.neck=detector.neck_source;self.dropout=detector.dropout
        self.preprocess=detector.preprocessing;self.strides=list(detector.fpn_strides)
        self.head=ProposalEncoder();self.body_mode(False)

    def body_mode(self,enabled):
        self.backbone.requires_grad_(enabled);self.neck.requires_grad_(enabled)
        self.backbone.eval();self.neck.eval();self.dropout.eval()

    def features(self,raw,metadata):
        video=[dict(feats=raw.T,segments=raw.new_empty(0,2),labels=torch.empty(0,dtype=torch.long,device=raw.device),metadata=metadata)]
        x,mask=self.preprocess(video,mode='test');f,m,_=self.backbone(self.dropout(x),mask);f,m=self.neck(f,m)
        return [a[0] for a in f],[a[0,0].bool() for a in m]

    def sample(self,raw,metadata,boxes):
        features,masks=self.features(raw,metadata);seq=[];valid=[]
        for f,m,stride in zip(features,masks,self.strides):
            a,b=region_samples(f,m,boxes/stride);seq.append(torch.cat(a,1));valid.append(torch.cat(b,1))
        return torch.stack(seq),torch.stack(valid),features[0].shape[-1]

    def logits(self,seq,valid,metadata):
        levels,n=seq.shape[:2]
        z=self.head(seq.flatten(0,1),valid.flatten(0,1),metadata.repeat(levels,1))
        return z.reshape(levels,n,2)


def proposal_loss(logits,bg,quality):
    levels=logits.shape[0];return head_objective(logits.flatten(0,1),bg.repeat(levels),quality.repeat(levels))


def cost_proposal_loss(logits,bg,quality,delta):
    # 原已审APcost函数AST不改，仅避免导入其CPU AP/Numba依赖。
    from pathlib import Path
    from torch.nn import functional as F
    from tools.hard_background_modules import balanced_bg,quality_loss
    if not hasattr(cost_proposal_loss,'function'):
        ns=dict(torch=torch,F=F);source_symbols(Path(__file__).with_name('background_ap_cost.py'),['cost_balanced_bg'],ns);cost_proposal_loss.function=ns['cost_balanced_bg']
    levels=logits.shape[0];z=logits.flatten(0,1);b=bg.repeat(levels);q=quality.repeat(levels);d=delta.repeat(levels)
    return .5*balanced_bg(z[:,0],b)+.5*cost_proposal_loss.function(z[:,0],b,d)+.25*quality_loss(z[:,1],q)


def critical_anchor(seq,teacher,valid,bg,delta):
    # 只锁定真实删后会损失AP的源FIT nonBG，而不是保护教师在背景上的幻觉。
    weight=torch.where(bg<.5,(-delta).clamp_min(0),torch.zeros_like(delta))
    inside=valid[:,:,4:20];error=(seq[:,:,4:20]-teacher[:,:,4:20]).square().mean(-1)
    per=(error*inside).sum(-1)/inside.sum(-1).clamp_min(1)
    return ((per*weight[None]).sum(-1)/weight.sum().clamp_min(1e-12)).mean()
