"""Ti-FAD/PDA机制迁移到DyFADet；不宣称原论文数值复现。

保留六层时序金字塔、原检测头及UDA实例分配。预训练文本缓存冻结；
文本投影、双向交互/阶段对齐、前景头和原视觉网络联合更新。
完整强度的跨模态路径；不添加零门控，也不把文本缩成一个辅助原型损失。
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from vendor.tifad.tv_attn import TVCA


def masked_softmax(logits, valid, dim):
    # 对没有有效点的输入返回0，不把padding当成背景观察。
    weights=logits.masked_fill(~valid, -1e4).softmax(dim)*valid.to(logits.dtype)
    return weights/weights.sum(dim,keepdim=True).clamp_min(1e-12)


class CrossAttention(nn.Module):
    def __init__(self,dim,heads=4):
        super().__init__();self.heads=heads;self.dim=dim
        self.q=nn.Linear(dim,dim);self.k=nn.Linear(dim,dim)
        self.v=nn.Linear(dim,dim);self.out=nn.Linear(dim,dim)

    def forward(self,q,kv,key_valid=None,importance=None):
        b,t,d=q.shape;h=self.heads
        query=self.q(q).reshape(b,t,h,d//h).transpose(1,2)
        key=self.k(kv).reshape(b,-1,h,d//h).transpose(1,2)
        value=self.v(kv).reshape(b,-1,h,d//h).transpose(1,2)
        score=query@key.transpose(-2,-1)/math.sqrt(d//h)
        if importance is not None:score=score*importance[:,None]
        if key_valid is None:weights=score.softmax(-1)
        else:weights=masked_softmax(score,key_valid[:,None,None,:],-1)
        return self.out((weights@value).transpose(1,2).reshape(b,t,d))


def salient_mask(features,text,valid,sigma,ratio=.01,initial_length=1152):
    """作者公开THUMOS配置top1%/threshold100；局部Gaussian融合为SAM。

    与作者实现一样是显著位置周围的软支持，不是学习的定位不确定性。
    padding不参加topk；保留作者相邻中心合并、sigmoid网格宽度及核求和归一化。
    """
    logits=features@text.transpose(-2,-1)
    salience=logits.max(-1).values
    result=[]
    for i in range(len(features)):
        ids=valid[i].nonzero().flatten();mask=salience[i]*0
        if len(ids):
            k=max(1,int(len(ids)*ratio))
            centers=ids[salience[i,ids].topk(k).indices].sort().values.tolist()
            merged=[];threshold=100*len(ids)/initial_length
            for center in centers:
                if not merged or center-merged[-1]>threshold:merged.append(float(center))
                else:merged[-1]=(merged[-1]+center)/2
            centers=torch.tensor(merged,device=features.device,dtype=features.dtype)
            coordinates=torch.arange(features.shape[1],device=features.device,dtype=features.dtype)
            distance=coordinates[:,None]-centers[None,:]
            widths=torch.sigmoid(sigma[i,centers.long()]).clamp_min(1e-6)
            mask=torch.exp(-.5*(distance/widths[None,:]).square()).sum(-1)*valid[i]
            mask=mask/(mask.max()+1e-5)
        result.append(mask)
    return torch.stack(result)


class TiLevel(nn.Module):
    def __init__(self,dim):
        super().__init__();self.vn=nn.LayerNorm(dim);self.tn=nn.LayerNorm(dim)
        self.text_self_attention=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0.)
        # 作者TVCA实现：六个Q/K/V投影、SAM、双向残差以及视觉/文本各自FFN。
        self.cross=TVCA(dim,dim,dim,4,dropout=0.,mlp_ratio=4.)
        self.sigma=nn.Conv1d(dim,1,3,padding=1)

    def forward(self,x,text,valid):
        t=self.tn(text);text=text+self.text_self_attention(t,t,t,need_weights=False)[0]
        v=self.vn(x);t=self.tn(text)
        sam=salient_mask(v,t,valid,self.sigma(v.transpose(1,2)).squeeze(1))
        visual,text=self.cross(x,text,sam[:,:,None],attention_mask_v=valid)
        visual=visual*valid[...,None]
        return visual,text,sam


class SemanticPyramid(nn.Module):
    def __init__(self,embeddings,variant,dim=512,levels=6):
        super().__init__();assert variant in ('tifad','pda')
        assert embeddings.ndim==3 and embeddings.shape[:2]==(10,4)
        self.variant=variant;self.register_buffer('text_embeddings',embeddings.clone())
        self.text_projection=nn.ModuleList([nn.Linear(embeddings.shape[-1],dim) for _ in range(4)])
        self.norm=nn.LayerNorm(dim)
        self.hard_fg=nn.Conv1d(dim,1,3,padding=1)
        self.soft_fg=nn.Conv1d(dim,1,3,padding=1)
        if variant=='tifad':
            self.levels=nn.ModuleList([TiLevel(dim) for _ in range(levels)])
            self.text_readout=nn.Linear(dim,dim)
            self.bias_lang=nn.Parameter(torch.zeros(dim));self.bias0=nn.Parameter(torch.zeros(()))
            self.logit_scale=nn.Parameter(torch.zeros(()))
        else:
            self.phase_attention=nn.ModuleList([CrossAttention(dim) for _ in range(4)])
            self.phase_positions=nn.Parameter(torch.zeros(4,dim))
            self.phase_self_attention=nn.MultiheadAttention(dim,4,batch_first=True,dropout=0.)
            self.phase_weight=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
            self.phase_regression=nn.Sequential(nn.Linear(dim*4,dim*2),nn.GELU(),nn.Linear(dim*2,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
            self.phase_classification=nn.Linear(dim,dim)

    def forward(self,features,masks):
        projected=torch.stack([p(self.text_embeddings[:,i]) for i,p in enumerate(self.text_projection)],1)
        cf=[];rf=[];logits=[];fg=[];soft=[];observations=[]
        # Ti-FAD的类文本使用整体动作描述；PDA四阶段保持独立，不先平均。
        text=projected[:,3].unsqueeze(0).expand(features[0].shape[0],-1,-1)
        for level,(feature,mask) in enumerate(zip(features,masks)):
            x=feature.transpose(1,2);valid=mask.squeeze(1).bool();xn=self.norm(x)
            if self.variant=='tifad':
                visual,text,sam=self.levels[level](x,text,valid)
                reg=visual
                score=x.new_zeros((*x.shape[:2],10))
                observations.append(dict(level=level,sam_mean=float(sam.detach().sum()/valid.sum().clamp_min(1))))
            else:
                filtered=[];aligned=[];scores=[];coverage=[]
                for phase in range(4):
                    t=projected[:,phase].unsqueeze(0).expand(len(x),-1,-1)
                    sim=F.normalize(xn,dim=-1)@F.normalize(t,dim=-1).transpose(-2,-1)
                    salience=masked_softmax(sim.max(-1).values,valid,-1)
                    threshold=valid.sum(-1,keepdim=True).clamp_min(1).reciprocal()
                    selected=(salience>=threshold)&valid
                    phase_x=x*selected[...,None]
                    a=(phase_x+self.phase_attention[phase](self.norm(phase_x),t))*selected[...,None]
                    filtered.append(phase_x);aligned.append(a)
                    scores.append(10*(F.normalize(a,dim=-1)@F.normalize(t,dim=-1).transpose(-2,-1)))
                    coverage.append(float(selected.detach().sum()/valid.sum().clamp_min(1)))
                # 采用论文附录的softmax阶段权重；不是正文sigmoid和附录混用。
                phase_tokens=torch.stack(filtered,-2)+self.phase_positions
                b,t,p,d=phase_tokens.shape
                phase_tokens=phase_tokens.reshape(b*t,p,d)
                phase_tokens=phase_tokens+self.phase_self_attention(phase_tokens,phase_tokens,phase_tokens,need_weights=False)[0]
                weights=self.phase_weight(phase_tokens).reshape(b,t,4).softmax(-1)
                score=(torch.stack(scores,-2)*weights[...,None]).sum(-2)
                visual=x+self.phase_classification((torch.stack(aligned,-2)*weights[...,None]).sum(-2))
                reg=x+self.phase_regression(torch.cat(filtered,-1))
                observations.append(dict(level=level,phase_coverage=coverage))
            visual=visual*valid[...,None];reg=reg*valid[...,None]
            hard=self.hard_fg(reg.transpose(1,2)).transpose(1,2)
            smooth=self.soft_fg(reg.transpose(1,2)).transpose(1,2)
            cf.append(visual.transpose(1,2));rf.append(reg.transpose(1,2))
            logits.append(score*valid[...,None]);fg.append(hard);soft.append(smooth)
        if self.variant=='tifad':
            # 与作者相同：最后更新的类文本供全部尺度读出，归一化/2后投影，学习温度和类偏置。
            normalized=F.normalize(text,dim=-1);bias=normalized@self.bias_lang+self.bias0
            readout=self.text_readout(normalized/2)
            logits=[((v.transpose(1,2)@readout.transpose(1,2))/self.logit_scale.exp()+bias[:,None,:]).clamp(-50000,50000)*m.squeeze(1)[...,None] for v,m in zip(cf,masks)]
        return cf,rf,logits,fg,soft,observations


class TextDetector(nn.Module):
    def __init__(self,detector,embeddings,variant,dim=512,levels=6):
        super().__init__();self.detector=detector
        self.semantic=SemanticPyramid(embeddings,variant,dim,levels)
        self.enabled=True;self.last_semantic=None

    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('detector'),name)

    @property
    def loss_normalizer(self):return self.detector.loss_normalizer

    @loss_normalizer.setter
    def loss_normalizer(self,value):self.detector.loss_normalizer=value

    def dense(self,videos):
        d=self.detector;x,m=d.preprocessing(videos)
        f,m=d.backbone(x,m);f,m=d.neck(f,m);points=d.point_generator(f)
        if self.enabled:
            cf,rf,extra,fg,soft,obs=self.semantic(f,m)
            cls=[a.transpose(1,2)+b for a,b in zip(d.cls_head(cf,m),extra)]
            reg=[a.transpose(1,2) for a in d.reg_head(rf,m)]
            self.last_semantic=(fg,soft,obs)
        else:
            cls=[a.transpose(1,2) for a in d.cls_head(f,m)]
            reg=[a.transpose(1,2) for a in d.reg_head(f,m)];self.last_semantic=None
        return points,[a.squeeze(1) for a in m],cls,reg

    def forward(self,videos):
        outputs=self.dense(videos)
        if not self.training:return self.inference(videos,*outputs)
        points,masks,cls,reg=outputs
        targets,offsets=self.detector.label_points(points,[v['segments'].to(self.device) for v in videos],[v['labels'].to(self.device) for v in videos])
        losses=self.detector.losses(masks,cls,reg,targets,offsets)
        if self.enabled:
            from libs.modeling.losses import sigmoid_focal_loss
            valid=torch.cat(masks,1);target=torch.stack(targets)
            positive=(target.sum(-1)>0)&valid
            hard=positive.to(cls[0].dtype)
            off=torch.stack(offsets);left,right=off[...,0],off[...,1]
            # 沿用Ti-FAD代码Gaussian软标签与sigma=1，在原stride归一化offset上计算。
            center=torch.exp(-.5*off.square()).amin(-1)*positive
            h,s,_=self.last_semantic
            denominator=float(self.loss_normalizer)
            hard_loss=sigmoid_focal_loss(torch.cat(h,1).squeeze(-1)[valid],hard[valid],reduction='sum')/denominator
            soft_loss=sigmoid_focal_loss(torch.cat(s,1).squeeze(-1)[valid],center[valid],reduction='sum')/denominator
            if self.semantic.variant=='pda':soft_loss=soft_loss*0
            losses=dict(losses,text_foreground_hard=hard_loss,text_foreground_soft=soft_loss)
            losses['final_loss']=losses['final_loss']+hard_loss+soft_loss
        return losses

    def inference(self,videos,points,masks,cls,reg):
        # Ti-FAD/PDA前景分支实际用于最终检测分数，采用概率乘积而非logit近似。
        # UDA伪标签仍使用分类分数筛选，并独立验证边界稳定性；不把前景概率冒充分类置信度。
        if self.enabled:
            hard,soft,_=self.last_semantic;fused=[]
            for c,h,s in zip(cls,hard,soft):
                logp=F.logsigmoid(c)+F.logsigmoid(h)
                if self.semantic.variant=='tifad':logp=.5*(F.logsigmoid(c)+F.logsigmoid(s))+F.logsigmoid(h)
                logp=logp.clamp_max(-1e-7)
                fused.append(logp-torch.log1p(-logp.exp()))
            cls=fused
        return self.detector.inference(videos,points,masks,cls,reg)
