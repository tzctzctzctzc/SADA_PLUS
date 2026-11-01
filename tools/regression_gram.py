"""DARE-GRAM回归域适应的可微核；仅作迁移可行性核验，不宣称检测收益。

依据：https://arxiv.org/abs/2303.13325，作者MIT代码的角度/谱尺度目标。
用特征矩阵的薄SVD计算同一截断Gram逆，避免对1537维Gram重复分解。
秩按论文解释方差比例选择；与作者代码的边界索引规则不完全相同，显式记录。
"""
import torch
from torch.nn import functional as F


SPEC=dict(schema='regression-gram-transfer-readiness-v2',explained_variance=.9,
          angle_weight=.05,scale_weight=.001,feature='actual-three-tap-regression-input',
          target_confidence=.5,max_points_per_video=128,videos_per_domain=2,
          rank_rule='common maximum rank-count reaching .9 explained variance; detached',
          zero_columns='both numerically absent: angle0; one absent: constant angle1; common: unit-column cosine',
          numerical_precision='float64 loss kernel only; detector precision unchanged',
          target_gt=False,formal_training=False)


def differentiable_detector_features(model,video):
    # 旧缓存helper有@no_grad；这里明确使用相同前向运算的可微路径。
    if model.backbone_source is not model.backbone_target or model.neck_source is not model.neck_target:
        raise ValueError('核验仅支持当前共享定位基座')
    inputs,masks=model.preprocessing([video],mode='test')
    features,masks,_=model.backbone_source(model.dropout(inputs),masks)
    return model.neck_source(features,masks)


def augmented_svd(features):
    a=torch.cat([torch.ones_like(features[:,:1]),features],1).double()
    _,s,vh=torch.linalg.svd(a,full_matrices=False)
    return s.square(),vh


def inverse_from_spectrum(eigenvalues,vh,rank):
    return (vh[:rank].T/eigenvalues[:rank])@vh[:rank]


def gram_alignment(source,target):
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1]:
        raise ValueError('源目标必须是相同回归特征维数的二维矩阵')
    if min(len(source),len(target))<4:raise ValueError('回归点不足，不假造对齐信号')
    es,vs=augmented_svd(source);et,vt=augmented_svd(target)
    def count(e):
        if not torch.isfinite(e).all() or e[0]<=0:raise ValueError('非有限或零谱')
        return int(torch.searchsorted(e.detach().cumsum(0)/e.detach().sum(),SPEC['explained_variance']))+1
    rank=max(count(es),count(et));rank=min(rank,len(es),len(et))
    # 截断秩内若数值退化则显式拒绝，不用很小特征值制造巨大伪逆梯度。
    if min(float(es[rank-1]/es[0]),float(et[rank-1]/et[0]))<1e-10:
        raise ValueError('共同子空间退化，不能安全迁移')
    gs=inverse_from_spectrum(es,vs,rank);gt=inverse_from_spectrum(et,vt,rank)
    # 稀疏ReLU回归特征可有全零列：两个相同零向量没有夹角差。
    # 固定eps=1e-6会将小范数逆Gram列当不相似，甚至相同域也出现假差异。
    ns=torch.linalg.vector_norm(gs,dim=0);nt=torch.linalg.vector_norm(gt,dim=0)
    tolerance=torch.finfo(gs.dtype).eps*torch.maximum(ns.max(),nt.max()).detach()
    present_s=ns.detach()>tolerance;present_t=nt.detach()>tolerance;common=present_s&present_t
    distance=(present_s^present_t).to(gs.dtype)
    distance[common]=(1-((gs[:,common]/ns[common])*(gt[:,common]/nt[common])).sum(0)).abs()
    angle=distance.mean()
    scale=torch.linalg.vector_norm(es[:rank]-et[:rank])/rank
    loss=SPEC['angle_weight']*angle+SPEC['scale_weight']*scale
    return loss,dict(rank=rank,angle=float(angle.detach()),scale=float(scale.detach()),
                     source_points=len(source),target_points=len(target),dimension=source.shape[1],
                     common_inverse_columns=int(common.sum()),one_sided_inverse_columns=int((present_s^present_t).sum()),
                     both_zero_inverse_columns=int((~present_s&~present_t).sum()))


def regression_inputs(head,features,masks):
    """保留原回归塔与边缘padding，按原offset_head三格卷积的顺序展开。"""
    rows=[]
    for f,m in zip(features,masks):
        for conv,norm in zip(head.head,head.norm):
            f,_=conv(f,m);f=head.act(norm(f))
        rows.append(F.unfold(f.unsqueeze(2),(1,3),padding=(0,1)).transpose(1,2))
    return torch.cat(rows,1)
