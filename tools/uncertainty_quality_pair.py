"""只在源同类匹配候选间学习阈值资格排序；保留有界分数先验与零校正回退。"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tools.uncertainty_quality import SPEC as INITIAL_SPEC
from tools.uncertainty_distribution import history_decision

SCHEMA='prototype-conditional-quality-pair-ranking-v1'
SPEC=dict(INITIAL_SPEC,revision=2,logodds_bound=1.,
          objective='same-class-matched-threshold-pair-logistic',
          pairs_per_cell=512,pair_batch_size=256,pair_temperature=1.,
          pair_sampling='inverse-class-matched-video-count; equal valid class-threshold cells',
          matched_min_iou=.1,ranking_thresholds=[.2,.3,.4,.5],
          selection='video-balanced-within-class-matched-eligibility-AUC',
          probabilities='original-class-score-prior-plus-bounded-logodds-residual',
          rescoring='p*exp(delta)/(1-p+p*exp(delta))',epoch_zero_identity_candidate=True,
          source_examples_reused=True,shuffle='fresh-seed123-after-cache-reuse')


class ResidualQualityHead(nn.Module):
    def __init__(self,channels=512,classes=10,levels=6):
        super().__init__();self.classes=classes;self.levels=levels
        self.encoder=nn.Sequential(nn.LayerNorm(channels*10),nn.Linear(channels*10,128),nn.GELU())
        self.out=nn.Sequential(nn.Linear(128+classes+levels+3,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.out[-1].weight);nn.init.zeros_(self.out[-1].bias)
    def forward(self,x,labels,levels,scores,width,complete):
        p=scores.clamp(1e-6,1-1e-6)
        extra=torch.cat([F.one_hot(labels,self.classes).to(x.dtype),F.one_hot(levels,self.levels).to(x.dtype),
                         torch.stack([(p/(1-p)).log(),width.clamp_min(0).log1p(),complete.to(x.dtype)],1)],1)
        return SPEC['logodds_bound']*self.out(torch.cat([self.encoder(x),extra],1)).squeeze(1).tanh()


def rescore(scores,delta=None):
    if delta is None:return scores
    factor=delta.exp()
    return scores*factor/(1-scores+scores*factor)


def pair_pools(labels,quality,video_ids):
    """源GT仅定义同类资格对；与源验证AUC相同的每视频权重。"""
    labels=np.asarray(labels);quality=np.asarray(quality);video_ids=np.asarray(video_ids)
    cells=[]
    for label in range(10):
        take=np.flatnonzero((labels==label)&(quality>=SPEC['matched_min_iou']))
        if not len(take):continue
        unique,counts=np.unique(video_ids[take],return_counts=True);counts=dict(zip(unique,counts))
        weight=torch.tensor([1/counts[v] for v in video_ids[take]],dtype=torch.float64)
        for threshold in SPEC['ranking_thresholds']:
            positive=quality[take]>=threshold;negative=~positive
            if positive.any() and negative.any():
                cells.append(dict(label=label,threshold=threshold,
                    positive=torch.tensor(take[positive]),negative=torch.tensor(take[negative]),
                    positive_weight=weight[positive],negative_weight=weight[negative]))
    if not cells:raise ValueError('没有源同类定位资格对，拒绝空训练')
    return cells


def sample_pairs(cells,generator,pairs_per_cell=None):
    count=SPEC['pairs_per_cell'] if pairs_per_cell is None else pairs_per_cell
    pairs=[]
    for cell in cells:
        p=cell['positive'][torch.multinomial(cell['positive_weight'],count,True,generator=generator)]
        n=cell['negative'][torch.multinomial(cell['negative_weight'],count,True,generator=generator)]
        pairs.append(torch.stack([p,n],1))
    pairs=torch.cat(pairs)
    return pairs[torch.randperm(len(pairs),generator=generator)]


def corrected_logodds(scores,delta):
    p=scores.clamp(1e-6,1-1e-6)
    return (p/(1-p)).log()+delta


def pair_objective(positive_logits,negative_logits):
    # 正样本质量通过阈值，负样本为同类匹配但未通过；降低损失应提高前者次序。
    return F.softplus(-(positive_logits-negative_logits)/SPEC['pair_temperature'])


def weighted_auc(score,positive,weight):
    score=np.asarray(score);positive=np.asarray(positive,dtype=bool);weight=np.asarray(weight)
    pos=weight[positive].sum();neg=weight[~positive].sum()
    if pos<=0 or neg<=0:return None
    order=np.argsort(score,kind='stable');s=score[order];y=positive[order];w=weight[order]
    starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    pw=np.add.reduceat(w*y,starts);nw=np.add.reduceat(w*(~y),starts)
    return float((pw*(np.cumsum(nw)-nw+.5*nw)).sum()/(pos*neg))


def matched_ranking_auc(scores,labels,quality,video_ids):
    scores=np.asarray(scores);labels=np.asarray(labels);quality=np.asarray(quality);video_ids=np.asarray(video_ids)
    values=[]
    for label in range(10):
        take=(labels==label)&(quality>=.1)
        if not take.any():continue
        vids=video_ids[take];unique,counts=np.unique(vids,return_counts=True);lookup=dict(zip(unique,counts))
        weight=np.asarray([1/lookup[v] for v in vids])
        for threshold in [.2,.3,.4,.5]:
            value=weighted_auc(scores[take],quality[take]>=threshold,weight)
            if value is not None:values.append(value)
    if not values:raise ValueError('源留出集没有可比较的同类定位阈值对')
    return float(np.mean(values))


def audit_entry(entry):
    from pathlib import Path
    from tools.baseline_protocol import read,sha
    e=Path(entry);r=read(e/'complete.json');assert r['schema']==SCHEMA and r['status']=='passed'
    for path,digest in r['artifacts'].items():assert sha(path)==digest,path
    reuse=read(e/'source_reuse.json');assert reuse['old_head_weights_loaded'] is False and sha(reuse['previous_complete'])==reuse['previous_complete_sha256']
    for p,digest in reuse['artifacts'].items():assert sha(p)==digest
    gpu=read(e/'gpu_complete.json');assert gpu['frozen_detector_unchanged'] and gpu['raw_boundaries_unchanged'] and gpu['raw_labels_unchanged'] and not gpu['target_gt_used']
    guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'];assert sha(guard['lease'])==guard['lease_sha256']
    lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'];assert guard['observed']['uuid']==lease['observed']['uuid'] and guard['observed']['free_mib']>=12288
    init=read(e/'initialization_proof.json');assert init['exact_original_score'] and init['maximum_absolute_error']==0
    a=read(e/'acceptance.json');assert a['source_loss']>0 and a['gradient_norm']>0 and a['branch_parameters_updated'] and a['bounded_logodds'] and not a['target_gt_used']
    best,stop=history_decision(read(e/'quality_history.json'));assert (best,stop)==(gpu['best_epoch'],gpu['stop_epoch'])
    ck=torch.load(e/'best_quality.pt',map_location='cpu');latest=torch.load(e/'latest_quality.pt',map_location='cpu');assert ck['epoch']==best and ck['specification']==SPEC and latest['history']==read(e/'quality_history.json') and latest['cuda_rng'].numel()>0 and latest['pair_generator'].numel()>0
    sampling=read(e/'source_pair_sampling.json');assert sampling['matched_min_iou']==.1 and sampling['background_supervision'] is False and sampling['cells']
    assert sampling['pairs_per_epoch']==len(sampling['cells'])*SPEC['pairs_per_cell']
    assert a['same_class_matched_pairs'] and a['training_pairs_per_epoch']==sampling['pairs_per_epoch']
    assert set(read(e/'source_train_gt.json')['database']).isdisjoint(read(e/'source_val_gt.json')['database'])
    ref=Path(read(e/'inputs.json')['reference']);assert sha(ref/'best.pt')==read(e/'inputs.json')['checkpoint_sha256']
    for domain in ['source','target']:
        assert all(not v.get('annotations') for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
        expected=read(ref/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(r['results'][domain]['control']-expected)<1e-7
        for mode in ['control','refined']:
            ev=read(e/'evaluation'/(domain+'_'+mode+'.json'));assert ev['prediction_sha256']==sha(e/(domain+'_'+mode+'.npz')) and ev['map']==r['results'][domain][mode]
    return dict(status='passed',complete_sha256=sha(e/'complete.json'),best_epoch=best,stop_epoch=stop)
