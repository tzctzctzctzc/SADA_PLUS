"""完整时序监督重访：输入原2304维特征，不再只训练冻结FPN上的候选读出。"""
import torch
from torch import nn
from torch.nn import functional as F

MODES=('dense10','dense97','targetssl','utility')


def occupancy(length,boxes,labels,classes=97):
    """标签为原特征中心单元被动作覆盖的比例；同类重叠取并集，保留短动作。"""
    y=torch.zeros(length,classes,dtype=torch.float32)
    # boxes已经按原loader变换至特征grid；这不是原视频输入感受野的推断。
    for c in range(classes):
        intervals=sorted((float(b[0]),float(b[1])) for b,k in zip(boxes,labels) if int(k)==c and b[1]>b[0])
        merged=[]
        for a,b in intervals:
            if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
            else:merged.append([a,b])
        t=torch.arange(length,dtype=torch.float32)
        for a,b in merged:y[:,c]+=(torch.minimum(t+.5,t.new_tensor(b))-torch.maximum(t-.5,t.new_tensor(a))).clamp_min(0)
    return y.clamp(0,1)


def dense_loss(logits,target):
    # 每类正负分别归一；无正例的类只学习负例，不伪造正例。
    p=target;n=1-p;ps=p.sum(0);ns=n.sum(0)
    a=(F.softplus(-logits)*p).sum(0)/ps.clamp_min(1e-12)
    b=(F.softplus(logits)*n).sum(0)/ns.clamp_min(1e-12)
    return ((a*(ps>0)+b*(ns>0))/((ps>0).float()+(ns>0).float()).clamp_min(1)).mean()


def sample_sequence(values,boxes,n=16):
    """线性插值，返回真正双邻点均有效的mask，禁止越界clamp伪造有效证据。"""
    t=(torch.arange(n,device=values.device,dtype=values.dtype)+.5)/n
    grid=boxes[:,0,None]+(boxes[:,1]-boxes[:,0])[:,None]*t
    lo=grid.floor().long();hi=lo+1;valid=(lo>=0)&(hi<len(values))
    alpha=grid-lo
    z=values[lo.clamp(0,len(values)-1)]*(1-alpha[...,None])+values[hi.clamp(0,len(values)-1)]*alpha[...,None]
    return z*valid[...,None],valid


def masked_mean(x,valid):return (x*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)


class DenseEvidence(nn.Module):
    def __init__(self):
        super().__init__();width=64
        self.project=nn.Linear(2304,width)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv1d(width,width,3,padding=d,dilation=d,groups=width),nn.GELU(),nn.Conv1d(width,width,1)) for d in (1,2,4)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in self.blocks])
        self.activity=nn.Linear(width,97)
        self.utility=nn.Sequential(nn.Linear(3*width+13,width),nn.GELU(),nn.Linear(width,2))

    def forward(self,raw):
        z=F.gelu(self.project(raw))
        for block,norm in zip(self.blocks,self.norms):z=norm(z+block(z.T[None])[0].T)
        return self.activity(z),z

    def candidate(self,logits,z,boxes,labels,scores):
        inside,valid=sample_sequence(z,boxes)
        # 左右只使用同长半区间；候选任务存在证据仍仅来自区间内部。
        span=(boxes[:,1]-boxes[:,0])/2
        left,lm=sample_sequence(z,torch.stack((boxes[:,0]-span,boxes[:,0]),1),4)
        right,rm=sample_sequence(z,torch.stack((boxes[:,1],boxes[:,1]+span),1),4)
        a=masked_mean(inside,valid);b=masked_mean(left,lm);c=masked_mean(right,rm)
        metadata=torch.cat((F.one_hot(labels.long(),10).float(),torch.logit(scores.clamp(1e-5,1-1e-5))[:,None]/5,
                            torch.log1p((boxes[:,1]-boxes[:,0])/len(z))[:,None],valid.float().mean(1,keepdim=True)),1)
        utility=self.utility(torch.cat((a-b,a,a-c,metadata),1))
        p,pm=sample_sequence(logits[:,:10].sigmoid(),boxes)
        # 任一任务动作的可见证据都保护；不足4有效采样点直接保留。
        task=p.masked_fill(~pm[...,None],0).amax((1,2))
        bg=(1-task).masked_fill(pm.sum(1)<4,0)
        score=(utility[:,0]-utility[:,1]).masked_fill(pm.sum(1)<4,-1e6)
        return bg,score,utility


def utility_loss(outputs,bg,delta):
    """分开拟合有益真BG与有害nonBG边际；不是把边际相加当作真实集合AP。"""
    benefit=torch.where(bg.bool(),delta.clamp_min(0),torch.zeros_like(delta))
    harm=torch.where(~bg.bool(),(-delta).clamp_min(0),torch.zeros_like(delta))
    target=torch.stack((benefit,harm),1)*10000
    return F.smooth_l1_loss(F.softplus(outputs),torch.log1p(target),reduction='mean')


def temporal_ssl(a,b):
    """同一时间两种掩码视图作正对，不把目标预测背景当伪GT；TS2Vec机制适配。"""
    a=F.normalize(a,dim=-1);b=F.normalize(b,dim=-1);sim=a@b.T/.2
    labels=torch.arange(len(a),device=a.device)
    return (F.cross_entropy(sim,labels)+F.cross_entropy(sim.T,labels))/2
