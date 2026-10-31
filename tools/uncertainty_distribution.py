"""两级源监督端点分布精修；第二级必须按第一级输出重新取真实FPN特征。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tools.uncertainty_refiner import aligned_iou

SCHEMA='prototype-progressive-boundary-distribution-v1'
SPEC=dict(route=3,revision=0,seed=123,frozen_detector=True,source_only=True,
          stages=2,bins=17,hidden=256,batch=256,lr=.001,max_epochs=50,patience=15,
          source_validation='equal-video-0.1-distribution-CE-plus-IoU',
          support='source-train-95th-percentile-absolute-endpoint-offset-clipped-0.25-to-2',
          stage2_support='source-train-residual-95th-percentile-at-most-half-stage1',
          context='five-real-regions-mean-max',same_original_level=True,
          stage2_resampling=True,epoch_zero_candidate=True,target_gt=False,target_training=False,
          uncertainty='finite-support-endpoint-distribution-variance-not-calibrated-probability')


def relative_target(boxes,gt):
    return (gt-boxes)/(boxes[:,1]-boxes[:,0]).clamp_min(1e-6)[:,None]


def support_from_source(boxes,gt,maximum=2.):
    q=float(relative_target(boxes,gt).abs().quantile(.95))
    return min(maximum,max(min(.25,maximum),q))


def soft_bin_loss(logits,target,radius):
    n=logits.shape[-1]
    z=((target/radius+1)*.5*(n-1)).clamp(0,n-1)
    lo=z.floor().long();hi=(lo+1).clamp_max(n-1);w=z-lo
    logp=logits.log_softmax(-1)
    return -((1-w)*logp.gather(-1,lo[...,None]).squeeze(-1)
             +w*logp.gather(-1,hi[...,None]).squeeze(-1)).mean(1)


class DistributionStage(nn.Module):
    def __init__(self,channels=512,radius=.5):
        super().__init__();self.encoder=nn.Sequential(nn.LayerNorm(channels*10),nn.Linear(channels*10,256),nn.GELU())
        self.out=nn.Linear(256,34);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
        self.register_buffer('grid',torch.linspace(-radius,radius,17))
        self.register_buffer('radius',torch.tensor(float(radius)))
    def forward(self,x,boxes):
        logits=self.out(self.encoder(x)).reshape(-1,2,17);p=logits.softmax(-1)
        # 减同一uniform浮点值，使零初始化严格不移动原框。
        shift=((p-torch.full_like(p,1/17))*self.grid).sum(-1)
        d=(boxes[:,1]-boxes[:,0]).clamp_min(1e-6)
        raw=boxes+shift*d[:,None]
        width=(raw[:,1]-raw[:,0]).clamp_min(.05*d);center=raw.mean(1)
        repaired=torch.stack([center-width/2,center+width/2],1)
        pred=torch.where(((raw[:,1]-raw[:,0])>=.05*d)[:,None],raw,repaired)
        variance=(p*(self.grid-shift[:,:,None]).square()).sum(-1)
        return pred,logits,variance


def objective(head,x,boxes,gt):
    pred,logits,var=head(x,boxes)
    loss=1-aligned_iou(pred,gt)+.1*soft_bin_loss(logits,relative_target(boxes,gt),head.radius)
    return loss,pred,var


def history_decision(history):
    if [x['epoch'] for x in history]!=list(range(len(history))):raise ValueError('缺少连续epoch0历史')
    best=float('inf');best_epoch=0
    for row in history:
        if not math.isfinite(row['source_validation']):raise ValueError('非有限源验证')
        if row['source_validation']<best:best=row['source_validation'];best_epoch=row['epoch']
        if row['epoch']!=history[-1]['epoch'] and row['epoch']-best_epoch>=15:raise ValueError('错过首次早停')
    stop=history[-1]['epoch']
    if stop<1 or (stop!=50 and stop-best_epoch<15):raise ValueError('未合法结束')
    return best_epoch,stop


def audit_entry(entry):
    from pathlib import Path
    from tools.baseline_protocol import read,sha
    e=Path(entry);r=read(e/'complete.json');assert r['status']=='passed' and r['schema']==SCHEMA
    for path,digest in r['artifacts'].items():assert sha(path)==digest,path
    bound=read(e/'inputs.json');ref=Path(bound['reference']);assert sha(ref/'best.pt')==bound['checkpoint_sha256']
    sources=read(e/'source_examples_reference.json');assert sha(sources['previous_complete'])==sources['previous_complete_sha256']
    for path,digest in sources['artifacts'].items():assert sha(path)==digest
    gpu=read(e/'gpu_complete.json');assert gpu['frozen_detector_unchanged'] and not gpu['target_gt_used']
    guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'];assert sha(guard['lease'])==guard['lease_sha256']
    lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'];assert guard['observed']['uuid']==lease['observed']['uuid'];assert guard['observed']['free_mib']>=12288
    for stage in [1,2]:
        hist=read(e/('stage%d_history.json'%stage));best,stop=history_decision(hist)
        ck=torch.load(e/('stage%d_best.pt'%stage),map_location='cpu');assert ck['epoch']==best and ck['specification']==SPEC
        latest=torch.load(e/('stage%d_latest.pt'%stage),map_location='cpu');assert latest['history']==hist and latest['cuda_rng'].numel()>0
        assert gpu['stages'][stage-1]['best_epoch']==best and gpu['stages'][stage-1]['stop_epoch']==stop
        accept=read(e/('stage%d_acceptance.json'%stage));assert accept['gradient_norm']>0 and accept['branch_parameters_updated'] and not accept['target_gt_used']
        if stage==2:assert accept['genuine_updated_box_resampling']
    train=read(e/'source_train_gt.json')['database'];val=read(e/'source_val_gt.json')['database'];assert set(train).isdisjoint(val)
    for domain in ['source','target']:
        assert all(not v.get('annotations') for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
        expected=read(ref/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(r['results'][domain]['control']-expected)<1e-7
        for mode in ['control','refined']:
            ev=read(e/'evaluation'/(domain+'_'+mode+'.json'));assert ev['prediction_sha256']==sha(e/(domain+'_'+mode+'.npz'));assert ev['map']==r['results'][domain][mode]
    return dict(status='passed',complete_sha256=sha(e/'complete.json'))
