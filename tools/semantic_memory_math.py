"""完整GT片段语义库重访；不把source查询视频自身留在近邻银行中。"""
import torch
from tools.dense_revisit_math import sample_sequence


def ordered_descriptor(values,boxes):
    x,m=sample_sequence(values,boxes,16);bins=[]
    for start in (0,4,8,12):
        z=x[:,start:start+4];v=m[:,start:start+4];mean=(z*v[...,None]).sum(1)/v.sum(1,keepdim=True).clamp_min(1)
        bins.append(torch.nn.functional.normalize(mean,dim=1))
    return torch.nn.functional.normalize(torch.cat(bins,1),dim=1),m.sum(1)>=4


def semantic_distances(query,bank,labels,bank_video,query_video,k=5,bank_norm=None):
    """完整float64距离；每类最近k均值，再由任一任务类提供前景证据。"""
    if bank_norm is None:bank_norm=bank.double().square().sum(1)
    d=(query.double().square().sum(1,keepdim=True)+bank_norm[None]-2*query.double()@bank.double().T).clamp_min(0)
    d[:,bank_video==query_video]=float('inf');columns=[]
    for c in range(98):
        idx=torch.nonzero((labels==c)&(bank_video!=query_video)).flatten()
        if len(idx):columns.append(d[:,idx].topk(min(k,len(idx)),largest=False).values.mean(1))
        else:columns.append(torch.full((len(query),),float('inf'),device=query.device,dtype=torch.float64))
    allclasses=torch.stack(columns,1);fg=allclasses[:,:10].min(1).values;bg=allclasses[:,10:].min(1).values
    return fg,fg-bg
