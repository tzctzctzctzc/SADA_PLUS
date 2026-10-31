"""路线一：冻结Prototype后的源监督边界扰动精修，不改变语义分支。"""
import torch
from torch import nn

SCHEMA='prototype-uncertainty-local-refiner-v1'
SPEC=dict(seed=123, source_only=True, frozen_detector=True, max_epochs=50, patience=15,
          batch=256, lr=.001, hidden=256, jitter=.2, context_fraction=.25,
          residual_center_limit=.5, log_duration_limit=.69314718056,
          source_jitters=4, real_candidates_per_video=200, minimum_match_iou=.1,
          uncertainty_use='inverse-scale-residual-damping', target_gt=False)

def iou(a,b):
    inter=(torch.minimum(a[:,None,1],b[None,:,1])-torch.maximum(a[:,None,0],b[None,:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])[:,None]+(b[:,1]-b[:,0])[None,:]-inter).clamp_min(1e-6)

def aligned_iou(a,b):
    inter=(torch.minimum(a[:,1],b[:,1])-torch.maximum(a[:,0],b[:,0])).clamp_min(0)
    return inter/((a[:,1]-a[:,0])+(b[:,1]-b[:,0])-inter).clamp_min(1e-6)

@torch.no_grad()
def context(features,masks,boxes,levels,strides):
    """五个区域的真实有效网格mean/max；不插值、不读取GT、不沿用原7位置。"""
    width=features[0].shape[1]*10
    out=features[0].new_zeros((len(boxes),width));valid=torch.zeros(len(boxes),device=out.device,dtype=torch.bool)
    for level,f in enumerate(features):
        seq=f[0].T;pos=torch.arange(len(seq),device=seq.device)*strides[level]
        mask=masks[level][0].reshape(-1).bool()
        for ids in (levels==level).nonzero(as_tuple=True)[0].split(8):
            if not len(ids):continue
            s,e=boxes[ids].unbind(1);d=(e-s).clamp_min(1)
            ranges=[(s-.25*d,s+.25*d),(s,e),(e-.25*d,e+.25*d),(s-.5*d,s),(e,e+.5*d)]
            pools=[];usable=torch.ones(len(ids),device=out.device,dtype=torch.bool)
            for a,b in ranges:
                take=(pos[None]>=a[:,None])&(pos[None]<=b[:,None])&mask[None]
                n=take.sum(1);usable &= n>0
                mean=take.to(seq.dtype)@seq/n.clamp_min(1)[:,None]
                maximum=seq[None].expand(len(ids),-1,-1).masked_fill(~take[:,:,None],-torch.inf).max(1).values
                maximum=torch.where((n>0)[:,None],maximum,torch.zeros_like(maximum))
                pools.extend([mean,maximum])
            out[ids]=torch.cat(pools,1);valid[ids]=usable
    return out,valid

class Refiner(nn.Module):
    def __init__(self,channels=512):
        super().__init__()
        self.encoder=nn.Sequential(nn.LayerNorm(channels*10),nn.Linear(channels*10,256),nn.GELU())
        self.out=nn.Linear(256,4)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,x,boxes):
        y=self.out(self.encoder(x));scale=.01+.49*y[:,2:].sigmoid()
        d=(boxes[:,1]-boxes[:,0]).clamp_min(1e-4);c=boxes.mean(1)
        trust=1/(1+scale.sum(1))
        c=c+d*.5*y[:,0].tanh()*trust
        d=d*torch.exp(.69314718056*y[:,1].tanh()*trust)
        return torch.stack([c-d/2,c+d/2],1),scale

def objective(model,x,boxes,gt):
    pred,scale=model(x,boxes);duration=(boxes[:,1]-boxes[:,0]).clamp_min(1e-4)
    error=(pred-gt).abs()/duration[:,None]
    nll=(error/scale+scale.log()).mean()
    return 1-aligned_iou(pred,gt).mean()+.1*nll,pred,scale
