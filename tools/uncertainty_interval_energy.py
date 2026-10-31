"""有限联合区间上的源监督能量；MAP选完整区间，不平均互不相容的端点。"""
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA='prototype-joint-interval-quality-ranking-v1'
SPEC=dict(route=4,revision=1,seed=123,frozen_detector=True,source_only=True,
          grid=[-.25,0.,.25],joint_candidates=9,identity_index=4,
          source_groups_per_video=32,inference_proposals_per_video=200,
          representation='real-five-region-mean-max-anchor-and-candidate-plus-offset-and-class',
          hidden=128,batch=64,lr=.001,max_epochs=50,patience=15,
          pair_temperature=.1,objective='quality-difference-weighted-all-pairs-logistic',
          selection='equal-video-source-realized-MAP-one-minus-IoU',
          decoding='joint-MAP-exact-tie-keeps-original',target_gt=False,target_training=False,
          uncertainty='finite-candidate-set-entropy-not-global-calibrated-density')


def candidates(boxes):
    axis=boxes.new_tensor(SPEC['grid']);s,e=torch.meshgrid(axis,axis,indexing='ij')
    delta=torch.stack([s.reshape(-1),e.reshape(-1)],1)
    d=(boxes[:,1]-boxes[:,0]).clamp_min(1e-6)
    return boxes[:,None]+d[:,None,None]*delta,delta.expand(len(boxes),-1,-1)


def select_interval(energy,valid):
    masked=energy.masked_fill(~valid,torch.inf);chosen=masked.argmin(1)
    # 未训练/完全平坦时，原候选是严格身份映射；不靠argmin任意偏到左上角。
    keep=(masked[:,4]<=masked.min(1).values)|~valid[:,4]
    return torch.where(keep,torch.full_like(chosen,4),chosen)


class IntervalEnergy(nn.Module):
    def __init__(self,channels=512,classes=10):
        super().__init__();self.encoder=nn.Sequential(nn.LayerNorm(channels*10),nn.Linear(channels*10,128),nn.GELU())
        self.score=nn.Sequential(nn.Linear(128*2+2+classes,128),nn.GELU(),nn.Linear(128,1))
        nn.init.zeros_(self.score[-1].weight);nn.init.zeros_(self.score[-1].bias)
        self.classes=classes
    def forward(self,anchor,variants,delta,labels):
        n,k,c=variants.shape
        a=self.encoder(anchor)[:,None].expand(-1,k,-1)
        v=self.encoder(variants.reshape(-1,c)).reshape(n,k,-1)
        cls=F.one_hot(labels,self.classes).to(v.dtype)[:,None].expand(-1,k,-1)
        return -self.score(torch.cat([a,v,delta,cls],-1)).squeeze(-1)


def objective(head,anchor,variants,delta,labels,quality,valid):
    energy=head(anchor,variants,delta,labels)
    score=-energy
    quality_gap=quality[:,:,None]-quality[:,None,:]
    score_gap=score[:,:,None]-score[:,None,:]
    weight=quality_gap.abs()*(valid[:,:,None]&valid[:,None,:])
    # 质量相同的候选不强行排序；与原框和其他候选都比较，较大IoU代价权重更大。
    pair=F.softplus(-quality_gap.sign()*score_gap/SPEC['pair_temperature'])
    return (pair*weight).sum((1,2))/weight.sum((1,2)).clamp_min(1e-6),energy


def audit_entry(entry):
    from pathlib import Path
    from tools.baseline_protocol import read,sha
    from tools.uncertainty_distribution import history_decision
    e=Path(entry);r=read(e/'complete.json');assert r['schema']==SCHEMA and r['status']=='passed'
    for path,digest in r['artifacts'].items():assert sha(path)==digest,path
    gpu=read(e/'gpu_complete.json');assert gpu['frozen_detector_unchanged'] and not gpu['target_gt_used']
    guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'];assert sha(guard['lease'])==guard['lease_sha256']
    lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'];assert guard['observed']['uuid']==lease['observed']['uuid'];assert guard['observed']['free_mib']>=12288
    best,stop=history_decision(read(e/'energy_history.json'));assert (best,stop)==(gpu['best_epoch'],gpu['stop_epoch'])
    ck=torch.load(e/'best_energy.pt',map_location='cpu');assert ck['epoch']==best and ck['specification']==SPEC
    latest=torch.load(e/'latest_energy.pt',map_location='cpu');assert latest['history']==read(e/'energy_history.json') and latest['cuda_rng'].numel()>0
    a=read(e/'acceptance.json');assert a['source_loss']>0 and a['gradient_norm']>0 and a['branch_parameters_updated'] and a['initialization_exact_identity'] and not a['target_gt_used']
    refs=read(e/'source_examples_reference.json');assert sha(refs['previous_complete'])==refs['previous_complete_sha256']
    for p,digest in refs['artifacts'].items():assert sha(p)==digest
    assert set(read(e/'source_train_gt.json')['database']).isdisjoint(read(e/'source_val_gt.json')['database'])
    ref=Path(read(e/'inputs.json')['reference']);assert sha(ref/'best.pt')==read(e/'inputs.json')['checkpoint_sha256']
    for domain in ['source','target']:
        assert all(not v.get('annotations') for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
        expected=read(ref/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(r['results'][domain]['control']-expected)<1e-7
        for mode in ['control','refined']:
            ev=read(e/'evaluation'/(domain+'_'+mode+'.json'));assert ev['prediction_sha256']==sha(e/(domain+'_'+mode+'.npz')) and ev['map']==r['results'][domain][mode]
    return dict(status='passed',complete_sha256=sha(e/'complete.json'),best_epoch=best,stop_epoch=stop)
