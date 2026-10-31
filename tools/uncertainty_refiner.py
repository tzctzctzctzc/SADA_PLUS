"""路线一：冻结Prototype后的源监督边界扰动精修，不改变语义分支。"""
import torch
from torch import nn

SCHEMA='prototype-uncertainty-local-refiner-gate-v1'
SPEC=dict(seed=123, source_only=True, frozen_detector=True, max_epochs=50, patience=15,
          batch=256, lr=.001, hidden=256, jitter=.2, context_fraction=.25,
          residual_center_limit=.5, log_duration_limit=.69314718056,
          source_jitters=4, real_candidates_per_video=200, minimum_match_iou=.1,
          uncertainty_use='inverse-scale-residual-damping', target_gt=False)
SPEC.update(revision=1, frozen_refiner=True, gate_threshold=0.5,
            gate_supervision='source-actual-intervention-iou-gain-gt-0.001',
            gate_selection='equal-video-source-validation-balanced-BCE',
            gate_hidden=64, gate_epochs=50, gate_patience=15)

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

class GainGate(nn.Module):
    """判断一次具体修正是否有益；不是预测原区间是否正确。"""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(260),nn.Linear(260,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(-1)

@torch.no_grad()
def gate_input(refiner,x,boxes):
    refined,scale=refiner(x,boxes)
    correction=(refined-boxes)/(boxes[:,1]-boxes[:,0]).clamp_min(1e-4)[:,None]
    return torch.cat([refiner.encoder(x),correction,scale],1),refined

def gated_boxes(boxes,refined,logits):
    return torch.where((logits>0)[:,None],refined,boxes)

def fit_gain_gate(refiner,train,val,entry):
    import numpy as np
    from tools.baseline_protocol import save
    refiner.eval().requires_grad_(False)
    def encode(data):
        xs=[];ys=[];gains=[]
        with torch.no_grad():
            for start in range(0,len(data['boxes']),256):
                x,b,g=[data[k][start:start+256].cuda() for k in ['x','boxes','gt']]
                z,p=gate_input(refiner,x,b)
                gain=aligned_iou(p,g)-aligned_iou(b,g)
                xs.append(z);ys.append((gain>.001).float());gains.extend(gain.cpu().tolist())
        ids=np.asarray(data['video_ids']);counts={v:int((ids==v).sum()) for v in set(ids)}
        weights=torch.tensor([1/counts[v] for v in ids],device='cuda:0')
        return torch.cat(xs),torch.cat(ys),weights,gains
    tx,ty,tw,tg=encode(train);vx,vy,vw,vg=encode(val)
    positives=int(ty.sum());negatives=len(ty)-positives
    if not positives or not negatives:raise ValueError('没有正负修正收益样本')
    gate=GainGate().cuda();opt=torch.optim.AdamW(gate.parameters(),lr=.001)
    pos_weight=ty.new_tensor(negatives/positives);generator=torch.Generator().manual_seed(123)
    history=[];best=float('inf');best_epoch=0;checked=False
    for epoch in range(1,51):
        gate.train()
        for ids in torch.randperm(len(tx),generator=generator).split(256):
            ids=ids.cuda();logits=gate(tx[ids])
            terms=nn.functional.binary_cross_entropy_with_logits(logits,ty[ids],pos_weight=pos_weight,reduction='none')
            loss=(terms*tw[ids]).sum()/tw[ids].sum();opt.zero_grad();loss.backward()
            grad=sum(float(p.grad.square().sum()) for p in gate.parameters() if p.grad is not None)**.5
            if not torch.isfinite(loss) or not grad>0:raise ValueError('收益门控梯度无效')
            prior=[p.detach().clone() for p in gate.parameters()] if not checked else []
            opt.step()
            if not checked:
                assert any(not torch.equal(a,b) for a,b in zip(prior,gate.parameters()))
                save(entry/'gate_acceptance.json',dict(status='passed',loss=float(loss),gradient_norm=grad,
                    parameters_updated=True,train_positive=positives,train_negative=negatives,target_gt_used=False))
                checked=True
        gate.eval()
        with torch.no_grad():
            logits=gate(vx);terms=nn.functional.binary_cross_entropy_with_logits(logits,vy,pos_weight=pos_weight,reduction='none')
            value=float((terms*vw).sum()/vw.sum())
        history.append(dict(epoch=epoch,source_validation=value))
        if value<best:
            best=value;best_epoch=epoch;torch.save(dict(state_dict=gate.state_dict(),epoch=epoch),entry/'best_gate.pt')
        torch.save(dict(state_dict=gate.state_dict(),optimizer=opt.state_dict(),history=history,
            generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)),entry/'latest_gate.pt')
        save(entry/'gate_history.json',history);print('GAIN_GATE_EPOCH',epoch,value,flush=True)
        if epoch-best_epoch>=15:break
    gate.load_state_dict(torch.load(entry/'best_gate.pt',map_location='cpu')['state_dict'],strict=True);gate.eval()
    with torch.no_grad():score=gate(vx).sigmoid().cpu().numpy()
    gain=np.asarray(vg);accepted=score>.5;ids=np.asarray(val['video_ids']);per_video=[]
    for vid in sorted(set(ids)):
        take=ids==vid;per_video.append(dict(video_id=vid,accepted_fraction=float(accepted[take].mean()),
            apply_all_gain=float(gain[take].mean()),gated_gain=float((gain[take]*accepted[take]).mean())))
    save(entry/'gate_mechanism.json',dict(best_epoch=best_epoch,stop_epoch=epoch,
        source_only=True,threshold=.5,samples=len(gain),per_video=per_video,
        equal_video_apply_all_gain=float(np.mean([x['apply_all_gain'] for x in per_video])),
        equal_video_gated_gain=float(np.mean([x['gated_gain'] for x in per_video]))))
    return gate
