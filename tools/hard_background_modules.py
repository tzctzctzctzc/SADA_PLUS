"""只判别高分候选的严格背景；质量辅助、相关负样本排序及可拆分正则。"""
import torch
from torch import nn
from torch.nn import functional as F
from tools.proposal_foreground_math import temporal_iou,sample_regions

MODES=('conditional','rank','context','perturb')


def targets(boxes,labels,truth,truth_labels):
    matrix=temporal_iou(boxes,truth)
    if len(truth):
        overlap=matrix.max(1).values
        quality=matrix.masked_fill(labels[:,None]!=truth_labels[None],0).max(1).values
    else:overlap=quality=boxes.new_zeros(len(boxes))
    # 错类但覆盖别的任务动作依然不是严格背景；质量和背景两个标签不混淆。
    return (overlap<.01).float().detach(),quality.detach()


def balanced_bg(logits,bg):
    losses=[F.binary_cross_entropy_with_logits(logits[bg==k],bg[bg==k]) for k in (0,1) if (bg==k).any()]
    return torch.stack(losses).mean() if losses else logits.sum()*0


def quality_loss(logits,quality):
    # VFL形式的连续同类tIoU辅助；不是把质量0的错类动作当背景。
    p=logits.sigmoid();weight=torch.where(quality>0,quality,.75*p.square())
    return (F.binary_cross_entropy_with_logits(logits,quality,reduction='none')*weight).mean() if len(logits) else logits.sum()*0


def rank_loss(logits,bg,labels,scores,limit=32):
    terms=[]
    for c in labels.unique():
        b=torch.where((labels==c)&(bg==1))[0];f=torch.where((labels==c)&(bg==0))[0]
        if not len(b) or not len(f):continue
        # 只对真正会与同类动作竞争的高分背景建配对，避免海量易背景主导。
        b=b[scores[b].argsort(descending=True)[:limit]];f=f[scores[f].argsort(descending=True)[:limit]]
        weight=(scores[b][:,None]/scores[f][None].clamp_min(1e-4)).clamp(max=2).detach()
        terms.append((F.softplus(.5-logits[b][:,None]+logits[f][None])*weight).mean())
    return torch.stack(terms).mean() if terms else logits.sum()*0


class HardBackgroundHead(nn.Module):
    def __init__(self,channels=512,width=64):
        super().__init__()
        self.projection=nn.Conv1d(channels,width,1)
        self.classifier=nn.Sequential(nn.Linear(width*3+3+10+2,width),nn.ReLU(),nn.Linear(width,2))

    def encode(self,feature,valid,boxes):
        projected=F.relu(self.projection(feature[None]))[0]
        return sample_regions(projected,valid,boxes)

    def classify(self,representation,boxes,labels,scores,video_length):
        cls=F.one_hot(labels.long(),10).to(representation.dtype)
        logit=torch.logit(scores.detach().clamp(1e-5,1-1e-5))[:,None]/5
        length=torch.log1p((boxes[:,1]-boxes[:,0]).detach()/max(1,video_length))[:,None]
        return self.classifier(torch.cat([representation,cls,logit,length],1))

    def forward(self,feature,valid,boxes,labels,scores):
        if not len(boxes):return feature.new_empty((0,2))+self.projection.weight.sum()*0
        reps=self.encode(feature,valid,boxes)
        return self.classify(reps,boxes,labels,scores,feature.shape[-1])


def context_swap(reps,bg,labels):
    """仅在表示空间交换同类候选外部上下文，保留内部内容；是待验证的反事实假设。"""
    swapped=reps.clone();used=torch.zeros(len(reps),dtype=torch.bool,device=reps.device)
    # reps=[inner-left,inner,inner-right,coverage3]，交换外部而不动inner。
    width=(reps.shape[-1]-3)//3
    for c in labels.unique():
        a=torch.where((labels==c)&(bg==1))[0];b=torch.where((labels==c)&(bg==0))[0]
        if not len(a) or not len(b):continue
        for dst,src in [(a,b),(b,a)]:
            src=src[torch.arange(len(dst),device=reps.device)%len(src)]
            inner=reps[dst,width:2*width]
            left=reps[src,width:2*width]-reps[src,:width]
            right=reps[src,width:2*width]-reps[src,2*width:3*width]
            swapped[dst,:width]=inner-left;swapped[dst,2*width:3*width]=inner-right
            swapped[dst,3*width]=reps[src,3*width];swapped[dst,3*width+2]=reps[src,3*width+2]
            used[dst]=True
    return swapped,used


def step_loss(head,item,mode,jitter=None):
    x=item;reps=head.encode(x['feature'],x['valid'],x['boxes'])
    out=head.classify(reps,x['boxes'],x['labels'],x['scores'],x['feature'].shape[-1])
    loss=balanced_bg(out[:,0],x['bg'])+.25*quality_loss(out[:,1],x['quality'])
    stats=dict(bg=float(loss.detach()),rank=0.,counterfactual=0.,consistency=0.,changed_labels=0)
    if mode!='conditional':
        r=rank_loss(out[:,0],x['bg'],x['labels'],x['scores']);loss=loss+.25*r;stats['rank']=float(r.detach())
    if mode=='context':
        cf,used=context_swap(reps,x['bg'],x['labels'])
        if used.any():
            other=head.classify(cf,x['boxes'],x['labels'],x['scores'],x['feature'].shape[-1])
            c=balanced_bg(other[used,0],x['bg'][used]);loss=loss+.1*c;stats['counterfactual']=float(c.detach())
    if mode=='perturb' and jitter is not None:
        boxes=x['boxes']+jitter*(x['boxes'][:,1]-x['boxes'][:,0])[:,None]
        bg,q=targets(boxes,x['labels'],x['truth'],x['truth_labels'])
        aug=head(x['feature'],x['valid'],boxes,x['labels'],x['scores'])
        loss=loss+.25*(balanced_bg(aug[:,0],bg)+.25*quality_loss(aug[:,1],q))
        stable=bg==x['bg']
        if stable.any():
            consistency=F.mse_loss(aug[stable,0].sigmoid(),out[stable,0].sigmoid().detach())
            loss=loss+.1*consistency;stats['consistency']=float(consistency.detach())
        stats['changed_labels']=int((~stable).sum())
    return loss,stats
