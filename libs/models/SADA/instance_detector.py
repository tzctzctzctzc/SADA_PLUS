"""实例级时序检测器：复用SGP编码，独立query解码与区间质量读出。

设计参考TadTR（TIP2022）的动作查询、区间细化和RoI质量估计。
这里独立实现六层金字塔上的一维可变形采样，不使用其CUDA扩展。
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import make_backbone,make_neck
from .blocks import LayerNorm


def specification(num_queries):
    return dict(schema='multiscale-instance-query-v1',num_queries=int(num_queries),dim=256,
        heads=8,decoder_layers=4,ff_dim=1024,points_per_level=4,roi_bins=16,
        roi_expansion=1.5,dropout=0.,classes=10,quality='source-supervised-max-tIoU',
        adversarial=False,variance_head=False,old_dense_detection_heads=False)


def inverse_sigmoid(x):
    x=x.clamp(1e-5,1-1e-5);return torch.log(x)-torch.log1p(-x)


def segments_from_cw(cw):
    return torch.stack((cw[...,0]-cw[...,1]/2,cw[...,0]+cw[...,1]/2),-1).clamp(0,1)


def sample_1d(feature,positions,mask):
    """feature: B,H,T,D；positions: B,H,N,K，坐标以该FPN层的格点为单位。"""
    b,h,t,d=feature.shape;n,k=positions.shape[-2:]
    left=positions.floor().long();fraction=positions-left.to(positions.dtype)
    values=feature.new_zeros(b,h,n,k,d);support=positions.new_zeros(b,h,n,k)
    for indices,weight in ((left,1-fraction),(left+1,fraction)):
        bounded=indices.clamp(0,t-1);flat=bounded.reshape(b,h,-1)
        sampled=feature.gather(2,flat[...,None].expand(-1,-1,-1,d)).reshape(b,h,n,k,d)
        valid=mask[:,None,:].expand(b,h,t).gather(2,flat).reshape(b,h,n,k)
        valid=valid&(indices>=0)&(indices<t)
        mass=weight*valid.to(weight.dtype);values=values+sampled*mass[...,None];support=support+mass
    return values,support


class TemporalCrossAttention(nn.Module):
    def __init__(self,dim,heads,levels,points):
        super().__init__();self.heads=heads;self.levels=levels;self.points=points
        self.offsets=nn.Linear(dim,heads*levels*points)
        self.weights=nn.Linear(dim,heads*levels*points)
        self.value=nn.Linear(dim,dim);self.output=nn.Linear(dim,dim)
        nn.init.zeros_(self.offsets.weight);nn.init.zeros_(self.weights.weight);nn.init.zeros_(self.weights.bias)
        offsets=torch.linspace(-2.,2.,points).repeat(heads,levels,1)
        with torch.no_grad():self.offsets.bias.copy_(offsets.flatten())

    def forward(self,query,reference,features,masks,lengths,strides):
        b,n,c=query.shape;h=self.heads;k=self.points
        offsets=self.offsets(query).reshape(b,n,h,self.levels,k).permute(0,2,1,3,4)
        weights=self.weights(query).reshape(b,n,h,self.levels*k).softmax(-1).reshape(b,n,h,self.levels,k).permute(0,2,1,3,4)
        result=query.new_zeros(b,h,n,c//h);mass=query.new_zeros(b,h,n)
        for level,(feature,mask,stride) in enumerate(zip(features,masks,strides)):
            value=self.value(feature.transpose(1,2)).reshape(b,-1,h,c//h).permute(0,2,1,3)
            center=reference[...,0][:,None,:,None]*lengths[:,None,None,None]/stride
            if reference.shape[-1]==2:
                radius=reference[...,1][:,None,:,None]*lengths[:,None,None,None]/(2*stride)
            else:radius=1.
            sampled,support=sample_1d(value,center+offsets[:,:,:,level]*radius,mask)
            w=weights[:,:,:,level];result=result+(sampled*w[...,None]).sum(3);mass=mass+(w*support).sum(3)
        result=result/mass.clamp_min(1e-6)[...,None]
        return self.output(result.permute(0,2,1,3).reshape(b,n,c))


class DecoderLayer(nn.Module):
    def __init__(self,spec,levels):
        super().__init__();d=spec['dim']
        self.self_attention=nn.MultiheadAttention(d,spec['heads'],dropout=spec['dropout'],batch_first=True)
        self.cross_attention=TemporalCrossAttention(d,spec['heads'],levels,spec['points_per_level'])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.ff=nn.Sequential(nn.Linear(d,spec['ff_dim']),nn.ReLU(),nn.Linear(spec['ff_dim'],d))

    def forward(self,x,position,reference,features,masks,lengths,strides):
        q=x+position;x=self.norms[0](x+self.self_attention(q,q,x,need_weights=False)[0])
        x=self.norms[1](x+self.cross_attention(x+position,reference,features,masks,lengths,strides))
        return self.norms[2](x+self.ff(x))


class InstanceTemporalDetector(nn.Module):
    def __init__(self,cfg,spec):
        super().__init__();self.cfg=deepcopy(cfg);self.spec=deepcopy(spec)
        m=cfg['model'];self.max_length=cfg['dataset']['max_seq_len'];self.strides=[m['scale_factor']**i for i in range(m['backbone_arch'][-1]+1)]
        self.pad_multiple=self.strides[-1];levels=len(self.strides);d=spec['dim']
        if self.max_length%self.pad_multiple:raise ValueError('窗口长度必须兼容金字塔下采样')
        windows=m['n_sgp_win_size'];windows=[windows]*levels if isinstance(windows,int) else windows
        self.encoder=make_backbone('SGP',n_in=m['input_dim'],n_embd=m['embd_dim'],sgp_mlp_dim=m['sgp_mlp_dim'],
            n_embd_ks=m['embd_kernel_size'],max_len=self.max_length,arch=m['backbone_arch'],scale_factor=m['scale_factor'],
            with_ln=m['embd_with_ln'],path_pdrop=m['droppath'],downsample_type=m['downsample_type'],
            sgp_win_size=windows,use_abs_pe=m['use_abs_pe'],k=m['k'],init_conv_vars=m['init_conv_vars'])
        self.neck=make_neck(m['fpn_type'],in_channels=[m['embd_dim']]*levels,out_channel=m['fpn_dim'],
            scale_factor=m['scale_factor'],with_ln=m['fpn_with_ln'])
        self.projections=nn.ModuleList([nn.Sequential(nn.Conv1d(m['fpn_dim'],d,1),LayerNorm(d)) for _ in range(levels)])
        self.level_position=nn.Parameter(torch.empty(levels,d));nn.init.normal_(self.level_position,std=.02)
        self.query_position=nn.Embedding(spec['num_queries'],d);self.query_content=nn.Embedding(spec['num_queries'],d)
        self.initial_reference=nn.Linear(d,1)
        self.decoder=nn.ModuleList([DecoderLayer(spec,levels) for _ in range(spec['decoder_layers'])])
        self.class_heads=nn.ModuleList([nn.Linear(d,spec['classes']) for _ in self.decoder])
        self.segment_heads=nn.ModuleList([nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU(),nn.Linear(d,2)) for _ in self.decoder])
        self.quality_head=nn.Sequential(nn.Linear(spec['roi_bins']*d,d),nn.ReLU(),nn.Linear(d,d),nn.ReLU(),nn.Linear(d,1))
        for layer in self.class_heads:nn.init.constant_(layer.bias,math.log(.01/.99))
        for i,layer in enumerate(self.segment_heads):
            nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)
            if i==0:layer[-1].bias.data[1]=-2.

    @property
    def device(self):return next(self.parameters()).device

    def forward(self,videos):
        # 前向只读特征；源GT在独立criterion中使用，目标GT无需进入检测器。
        lengths=torch.tensor([v['feats'].shape[-1] for v in videos],device=self.device)
        if bool((lengths<=0).any()) or int(lengths.max())>self.max_length:raise ValueError('单个窗口长度越界')
        x=torch.stack([F.pad(v['feats'].to(self.device),(0,self.max_length-v['feats'].shape[-1])) for v in videos])
        mask=torch.arange(self.max_length,device=self.device)[None,:]<lengths[:,None]
        features,masks,_=self.encoder(x,mask[:,None]);features,masks=self.neck(features,masks)
        projected=[];valid=[];d=self.spec['dim']
        for i,(feat,msk,proj,stride) in enumerate(zip(features,masks,self.projections,self.strides)):
            coords=torch.arange(feat.shape[-1],device=self.device)[None,:]*stride/lengths[:,None]
            frequencies=10000**(torch.arange(0,d,2,device=self.device,dtype=feat.dtype)/d)
            angles=coords[...,None]*2*math.pi/frequencies
            position=torch.stack((angles.sin(),angles.cos()),-1).flatten(-2).transpose(1,2)
            projected.append((proj(feat)+position+self.level_position[i][None,:,None])*msk.to(feat.dtype));valid.append(msk[:,0].bool())
        b=len(videos);position=self.query_position.weight[None].expand(b,-1,-1)
        q=self.query_content.weight[None].expand(b,-1,-1);reference=self.initial_reference(position).sigmoid();outputs=[]
        for layer,cls,segment in zip(self.decoder,self.class_heads,self.segment_heads):
            q=layer(q,position,reference,projected,valid,lengths,self.strides)
            delta=segment(q)
            cw=torch.cat(((delta[...,:1]+inverse_sigmoid(reference[...,:1])).sigmoid(),
                (delta[...,1:]+inverse_sigmoid(reference[...,1:])).sigmoid() if reference.shape[-1]==2 else delta[...,1:].sigmoid()),-1)
            outputs.append(dict(pred_logits=cls(q),pred_segments=segments_from_cw(cw),pred_cw=cw,instance_embeddings=q))
            reference=cw.detach()
        result=outputs[-1];segments=result['pred_segments'].detach();center=segments.mean(-1);width=segments[...,1]-segments[...,0]
        fractions=(torch.arange(self.spec['roi_bins'],device=self.device,dtype=x.dtype)+.5)/self.spec['roi_bins']-.5
        locations=(center[...,None]+width[...,None]*self.spec['roi_expansion']*fractions)*lengths[:,None,None]
        pooled,support=sample_1d(projected[0].transpose(1,2)[:,None],locations[:,None],valid[0])
        pooled=pooled[:,0]/support[:,0].clamp_min(1e-6)[...,None]
        result['pred_quality_logits']=self.quality_head(pooled.flatten(-2)).squeeze(-1)
        result['aux_outputs']=outputs[:-1];result['feature_lengths']=lengths
        return result

    @torch.no_grad()
    def predict_video(self,video):
        if self.training:raise ValueError('预测须先eval')
        length=video['feats'].shape[-1];step=max(self.pad_multiple,self.max_length//2)
        starts=list(range(0,max(length-self.max_length,0)+1,step))
        if starts[-1]!=max(length-self.max_length,0):starts.append(max(length-self.max_length,0))
        segments=[];scores=[];labels=[]
        for start in starts:
            clip=dict(feats=video['feats'][:,start:start+self.max_length]);output=self([clip])
            probability,label=output['pred_logits'][0].sigmoid().max(-1)
            scores.append((probability*output['pred_quality_logits'][0].sigmoid()).sqrt().cpu());labels.append(label.cpu())
            segments.append((output['pred_segments'][0]*clip['feats'].shape[-1]+start).cpu())
        seg=torch.cat(segments);score=torch.cat(scores);label=torch.cat(labels)
        meta=video['metadata'];seg=seg+float(meta.get('crop_start',0))
        seg=(seg*meta['feat_stride']+.5*meta['feat_num_frames'])/meta['fps']
        seg=seg.clamp(min=0,max=float(meta['duration']))
        test=self.cfg['test_cfg'];keep=(score>=test['min_score'])&((seg[:,1]-seg[:,0])>=test['duration_thresh'])
        seg,score,label=seg[keep],score[keep],label[keep]
        if len(starts)>1 and test['nms_method']!='none':
            from libs.utils.nms import batched_nms
            seg,score,label=batched_nms(seg,score,label,test['iou_threshold'],test['min_score'],test['max_seg_num'],
                use_soft_nms=test['nms_method']=='soft',multiclass=True,sigma=test['nms_sigma'],voting_thresh=test['voting_thresh'])
        else:
            order=score.argsort(descending=True)[:test['max_seg_num']];seg,score,label=seg[order],score[order],label[order]
        return dict(video_id=meta['video_id'],segments=seg,scores=score,labels=label)
