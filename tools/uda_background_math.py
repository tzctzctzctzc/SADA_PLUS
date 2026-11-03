"""细分源动作监督及三种无标签目标适应机制；不接收任何目标GT。"""
import torch
from torch import nn
from torch.nn import functional as F
from tools.background_four_routes_math import ProposalEncoder,masked_mean,head_objective
from tools.highscore_pair_math import build_batch
from tools.hard_background_modules import quality_loss
from tools.proposal_foreground_math import temporal_iou

MODES=('ontology','density','coral','teacher')


class UDAHead(ProposalEncoder):
    def __init__(self):
        super().__init__('asl',width=64);self.semantic=nn.Linear(64,97)

    def all_outputs(self,seq,valid,metadata):
        x=self.encode(seq,valid);left=masked_mean(x[:,:4],valid[:,:4]);inside=masked_mean(x[:,4:20],valid[:,4:20]);right=masked_mean(x[:,20:],valid[:,20:])
        coverage=torch.stack([valid[:,a:b].float().mean(1) for a,b in ((0,4),(4,20),(20,24))],1)
        logits=self.head(torch.cat((inside-left,inside,inside-right,coverage,metadata),1))
        return logits,self.semantic(inside),inside


def semantic_targets(boxes,gt,labels):
    """仅源GT：任一任务有重叠时，不允许较强的非任务类覆盖任务前景。"""
    out=torch.full((len(boxes),),-1,dtype=torch.long,device=boxes.device)
    if not len(gt):return out
    overlap=temporal_iou(boxes,gt);task=(labels>=0)&(labels<10);other=(labels>=10)&(labels<97)
    tm=overlap[:,task].max(1).values if task.any() else boxes.new_zeros(len(boxes))
    for mask,eligible in ((task,tm>=.01),(other,tm<.01)):
        if mask.any():
            score,index=overlap[:,mask].max(1);valid=eligible&(score>=.1);out[valid]=labels[mask][index[valid]]
    return out


def covariance(x):
    z=x-x.mean(0,keepdim=True);return z.T@z/(len(x)-1)


def conditional_coral(source,target,source_class,target_class):
    terms=[]
    for c in range(10):
        a=source[source_class==c];b=target[target_class==c]
        if len(a)>1 and len(b)>1:terms.append((covariance(a)-covariance(b)).square().sum()/(4*a.shape[1]**2))
    return torch.stack(terms).mean() if terms else source.sum()*0+target.sum()*0


def weighted_supervision(logits,bg,quality,weights=None):
    if weights is None:return head_objective(logits,bg,quality)
    terms=[]
    for k in (0,1):
        mask=bg==k
        if mask.any():
            loss=F.binary_cross_entropy_with_logits(logits[mask,0],bg[mask],reduction='none');w=weights[mask];terms.append((loss*w).sum()/w.sum().clamp_min(1e-12))
    bg_loss=torch.stack(terms).mean() if terms else logits.sum()*0
    return bg_loss+.25*quality_loss(logits[:,1],quality)


def source_objective(model,data,primary,pairs,pair_weights,importance=None,semantic_weight=.1):
    batch=build_batch(data,primary,pairs);logits,semantic,inside=model.all_outputs(batch['seq'],batch['valid'],batch['metadata']);n=len(primary);p=len(pairs)
    weights=importance[primary] if importance is not None else None
    ordinary=weighted_supervision(logits[:n],data['bg'][primary],data['quality'][primary],weights)
    pw=pair_weights
    if importance is not None:pw=pw*(importance[pairs[:,0]]*importance[pairs[:,1]]).sqrt()
    normalized=pw/pw.mean().clamp_min(1e-12)
    ranking=(F.softplus(.5-logits[n:n+p,0]+logits[n+p:,0])*normalized).mean()
    labels=data['semantic'][primary];ok=labels>=0
    aux=F.cross_entropy(semantic[:n][ok],labels[ok]) if ok.any() else semantic.sum()*0
    return ordinary+.25*ranking+semantic_weight*aux,dict(ordinary=ordinary,pair=ranking,semantic=aux),inside[:n]


@torch.no_grad()
def ema_update(student,teacher,step,decay=.99):
    alpha=min(1-1/(step+1),decay)
    for a,b in zip(teacher.parameters(),student.parameters()):a.mul_(alpha).add_(b,alpha=1-alpha)


def consistency_loss(student_logits,teacher_logits):
    return F.mse_loss(student_logits.sigmoid(),teacher_logits.detach().sigmoid())
