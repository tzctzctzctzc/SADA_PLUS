"""高分难背景对AP关键前景：配对共享非视觉条件，避免仅靠分数/长度排序。"""
import torch
from torch.nn import functional as F
from tools.background_four_routes_math import ProposalEncoder,head_objective


def build_batch(data,primary,pairs):
    b,f=pairs[:,0],pairs[:,1]
    indices=torch.cat((primary,b,f))
    batch={k:x[indices] for k,x in data.items() if k in ('seq','valid','metadata','bg','quality')}
    metadata=batch['metadata'].clone();n=len(primary);p=len(pairs)
    shared=(data['metadata'][b]+data['metadata'][f])/2
    metadata[n:n+p]=shared;metadata[n+p:]=shared
    batch['metadata']=metadata
    return batch


def objective(model,batch,primary_count,pair_weights,enabled):
    logits=model(batch['seq'],batch['valid'],batch['metadata']);n=primary_count;p=len(pair_weights)
    ordinary=head_objective(logits[:n],batch['bg'][:n],batch['quality'][:n])
    ranking=F.softplus(.5-logits[n:n+p,0]+logits[n+p:,0])
    weight=pair_weights/pair_weights.mean().clamp_min(1e-12)
    pair_loss=(ranking*weight).mean()
    return ordinary+(.25*pair_loss if enabled else pair_loss*0),dict(ordinary=ordinary.detach(),pair=pair_loss.detach())


def make_model():return ProposalEncoder('asl',width=64)
