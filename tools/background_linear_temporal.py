"""冻结FPN上的低容量读出，以及同参数量的内部时序方向消融。"""
import torch
from torch import nn
from torch.nn import functional as F
from tools.proposal_foreground_math import sample_regions
from tools.hard_background_modules import balanced_bg, quality_loss

MODES=('max_linear','temporal_abs','temporal_signed')


def region_samples(feature,valid,boxes):
    """与原4/16/4相同坐标和双邻点有效规则；不修改框，不复制越界特征。"""
    boxes=boxes.detach();length=boxes[:,1]-boxes[:,0];values=[];masks=[]
    for a,b,n in ((boxes[:,0]-.25*length,boxes[:,0],4),(boxes[:,0],boxes[:,1],16),(boxes[:,1],boxes[:,1]+.25*length,4)):
        f=(torch.arange(n,device=feature.device,dtype=feature.dtype)+.5)/n
        x=a[:,None]+(b-a)[:,None]*f;lo=x.floor().long();hi=x.ceil().long();w=x-lo
        l=lo.clamp(0,feature.shape[-1]-1);h=hi.clamp(0,feature.shape[-1]-1)
        ok=(x>=0)&(x<=feature.shape[-1]-1)&valid[l]&valid[h]
        v=(feature[:,l]*(1-w)[None]+feature[:,h]*w[None]).permute(1,2,0)
        values.append(v);masks.append(ok)
    return values,masks


def temporal_descriptor(values,masks,signed):
    def mean(v,m):
        return (v*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
    left,inside,right=[mean(v,m) for v,m in zip(values,masks)]
    early=mean(values[1][:,:8],masks[1][:,:8]);late=mean(values[1][:,8:],masks[1][:,8:])
    # 缺一侧时不把缺失零值解释为动作变化；两个分支共用同一个有效标志。
    pair=masks[1][:,:8].any(1)&masks[1][:,8:].any(1)
    difference=(late-early)*pair[:,None]
    if not signed:difference=difference.abs()
    coverage=[m.to(inside.dtype).mean(1,keepdim=True) for m in masks]
    return torch.cat((inside-left,inside,inside-right,difference,*coverage,pair[:,None].to(inside.dtype)),1)


def descriptors(feature,valid,boxes,labels,scores):
    """无可训练参数和随机操作；可逐视频缓存，与直接前向逐值核验。"""
    values,masks=region_samples(feature,valid,boxes)
    metadata=torch.cat((F.one_hot(labels.long(),10).to(feature.dtype),
        torch.logit(scores.detach().clamp(1e-5,1-1e-5))[:,None]/5,
        torch.log1p((boxes[:,1]-boxes[:,0]).detach()/max(1,feature.shape[-1]))[:,None]),1)
    reps={'max_linear':sample_regions(feature,valid,boxes),
          'temporal_abs':temporal_descriptor(values,masks,False),
          'temporal_signed':temporal_descriptor(values,masks,True)}
    return {m:torch.cat((r,metadata),1).detach() for m,r in reps.items()}


class LinearBackground(nn.Module):
    def __init__(self,mode,channels=512):
        super().__init__();assert mode in MODES;self.mode=mode
        size=3*channels+15 if mode=='max_linear' else 4*channels+16
        self.classifier=nn.Linear(size,2)
        # 相同零logits起点；不读取旧拟合头参数，不把随机初始化差异当表示收益。
        nn.init.zeros_(self.classifier.weight);nn.init.zeros_(self.classifier.bias)

    def forward(self,x):
        return self.classifier(x)


def objective(logits,bg,quality):
    return balanced_bg(logits[:,0],bg)+.25*quality_loss(logits[:,1],quality)
