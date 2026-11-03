"""固定时间尺度重采样与原物理时间网格上的分类/回归融合。"""
import torch
from torch.nn import functional as F


def rescale_raw(raw,scale):
    if scale==1.:return raw,1.
    assert len(raw)>1
    length=max(2,round((len(raw)-1)*scale)+1);ratio=(length-1)/(len(raw)-1)
    return F.interpolate(raw.T[None],size=length,mode='linear',align_corners=True)[0].T,ratio


def align_head(value,valid,base_length,ratio):
    # 每层原点i*stride映射到缩放层坐标i*ratio；两端位置保持同一物理时刻。
    query=torch.arange(base_length,device=value.device,dtype=value.dtype)*ratio
    left=query.floor().long();right=left+1;inside=(left>=0)&(right<len(valid));l=left.clamp(0,len(valid)-1);r=right.clamp(0,len(valid)-1)
    mask=inside&valid[l]&valid[r];fraction=(query-left).clamp(0,1)
    result=value[:,l]*(1-fraction)[None]+value[:,r]*fraction[None]
    return result,mask


def fuse_heads(base_cls,base_reg,views):
    if not views:return base_cls,base_reg
    probabilities=base_cls.sigmoid();offsets=base_reg;count=torch.ones(base_cls.shape[-1],device=base_cls.device)
    for cls,reg,valid,ratio in views:
        p,mask=align_head(cls.sigmoid(),valid,base_cls.shape[-1],ratio);r,_=align_head(reg,valid,base_cls.shape[-1],ratio)
        probabilities=probabilities+p*mask[None];offsets=offsets+r/ratio*mask[None];count=count+mask
    probability=probabilities/count[None]
    return torch.logit(probability),offsets/count[None]
