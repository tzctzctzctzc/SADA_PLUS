"""源监督定位质量：单调tIoU通过概率，NMS前重评分，不直接改边界。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tools.uncertainty_refiner import iou

SCHEMA='prototype-localization-quality-rescoring-v1'
SPEC=dict(route=5,revision=0,seed=123,frozen_detector=True,source_only=True,
          thresholds=[.1,.2,.3,.4,.5],hidden=128,batch=256,lr=.001,max_epochs=50,patience=15,
          source_sampling='per-predicted-class-top20-plus-random20-remaining',
          input='five-real-regions-mean-max-plus-class-level-score-width-context-completeness',
          supervision='same-class-GT-max-IoU-after-evaluation-time-conversion-and-clipping',
          selection='equal-video-source-multi-threshold-BCE',
          probabilities='sigmoid-base-minus-cumulative-softplus-decrements',
          rescoring='original-score-times-mean-five-quality-probabilities',
          direct_boundary_change=False,original_NMS_and_voting_unchanged=True,
          target_gt=False,target_training=False,calibrated_target_probability_claim=False)


def evaluated_intervals(boxes,metadata):
    if metadata['use_temporal_grid_feats']:
        boxes=(boxes*float(metadata['feat_stride'])+.5*float(metadata['feat_num_frames']))/float(metadata['fps'])
        limit=float(metadata['duration'])
    else:limit=float(metadata['video_length'])
    return boxes.clamp(0,limit)


def same_class_quality(boxes,labels,truth,truth_labels):
    if not len(truth):return boxes.new_zeros(len(boxes))
    overlap=iou(boxes,truth);overlap.masked_fill_(labels[:,None]!=truth_labels[None,:],0)
    return overlap.max(1).values


class QualityHead(nn.Module):
    def __init__(self,channels=512,classes=10,levels=6):
        super().__init__();self.classes=classes;self.levels=levels
        self.encoder=nn.Sequential(nn.LayerNorm(channels*10),nn.Linear(channels*10,128),nn.GELU())
        self.out=nn.Sequential(nn.Linear(128+classes+levels+3,128),nn.GELU(),nn.Linear(128,5))
        nn.init.zeros_(self.out[-1].weight)
        with torch.no_grad():self.out[-1].bias.copy_(torch.tensor([2.,-2.,-2.,-2.,-2.]))
    def forward(self,x,labels,levels,scores,width,complete):
        p=scores.clamp(1e-6,1-1e-6);logit=(p/(1-p)).log()
        extra=torch.cat([F.one_hot(labels,self.classes).to(x.dtype),F.one_hot(levels,self.levels).to(x.dtype),
                         torch.stack([logit,width.clamp_min(0).log1p(),complete.to(x.dtype)],1)],1)
        raw=self.out(torch.cat([self.encoder(x),extra],1))
        return torch.cat([raw[:,:1],raw[:,:1]-F.softplus(raw[:,1:]).cumsum(1)],1)


def objective(head,x,labels,levels,scores,width,complete,quality):
    logits=head(x,labels,levels,scores,width,complete)
    target=(quality[:,None]>=quality.new_tensor(SPEC['thresholds'])).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean(1),logits,target


def rescore(scores,probabilities=None):
    if probabilities is None:return scores
    return scores*probabilities.mean(1)


def audit_history(history):
    if [x['epoch'] for x in history]!=list(range(1,len(history)+1)):raise ValueError('源质量头训练历史不连续')
    best=float('inf');best_epoch=0
    for row in history:
        if not math.isfinite(row['source_validation']):raise ValueError('源验证loss非有限')
        if row['source_validation']<best:best=row['source_validation'];best_epoch=row['epoch']
        if row['epoch']!=history[-1]['epoch'] and row['epoch']-best_epoch>=15:raise ValueError('错过首次早停')
    stop=history[-1]['epoch']
    if stop!=50 and stop-best_epoch<15:raise ValueError('质量头未合法结束')
    return best_epoch,stop


def audit_entry(entry):
    from pathlib import Path
    from tools.baseline_protocol import read,sha
    e=Path(entry);r=read(e/'complete.json');assert r['schema']==SCHEMA and r['status']=='passed'
    for path,digest in r['artifacts'].items():assert sha(path)==digest,path
    gpu=read(e/'gpu_complete.json');assert gpu['frozen_detector_unchanged'] and gpu['raw_boundaries_unchanged'] and gpu['raw_labels_unchanged'] and not gpu['target_gt_used']
    guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'];assert sha(guard['lease'])==guard['lease_sha256']
    lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'];assert guard['observed']['uuid']==lease['observed']['uuid'] and guard['observed']['free_mib']>=12288
    a=read(e/'acceptance.json');assert a['source_loss']>0 and a['gradient_norm']>0 and a['branch_parameters_updated'] and a['monotone_probabilities'] and not a['target_gt_used']
    best,stop=audit_history(read(e/'quality_history.json'));assert (best,stop)==(gpu['best_epoch'],gpu['stop_epoch'])
    ck=torch.load(e/'best_quality.pt',map_location='cpu');latest=torch.load(e/'latest_quality.pt',map_location='cpu')
    assert ck['epoch']==best and ck['specification']==SPEC and latest['history']==read(e/'quality_history.json') and latest['cuda_rng'].numel()>0
    assert set(read(e/'source_train_gt.json')['database']).isdisjoint(read(e/'source_val_gt.json')['database'])
    ref=Path(read(e/'inputs.json')['reference']);assert sha(ref/'best.pt')==read(e/'inputs.json')['checkpoint_sha256']
    for domain in ['source','target']:
        assert all(not v.get('annotations') for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
        expected=read(ref/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(r['results'][domain]['control']-expected)<1e-7
        for mode in ['control','refined']:
            ev=read(e/'evaluation'/(domain+'_'+mode+'.json'));assert ev['prediction_sha256']==sha(e/(domain+'_'+mode+'.npz')) and ev['map']==r['results'][domain][mode]
    return dict(status='passed',complete_sha256=sha(e/'complete.json'),best_epoch=best,stop_epoch=stop)
