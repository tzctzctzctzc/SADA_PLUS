"""独立审计原MD能量完整机制；不把源课程IoU当真实检测AP。"""
import math
from pathlib import Path
import torch
from tools.baseline_protocol import read,sha
from tools.temporal_interval_energy import SCHEMA,SPEC,joint_decision


def audit_entry(entry):
    e=Path(entry);complete=read(e/'complete.json');assert complete['schema']==SCHEMA and complete['specification']==SPEC
    for p,d in complete['artifacts'].items():assert sha(p)==d,p
    cfg=read(e/'config.json');reference=Path(read(e/'inputs.json')['reference']);assert sha(reference/'best.pt')==read(e/'inputs.json')['checkpoint_sha256']
    for role in ['source_train','source_val']:
        ids=set(read(e/(role+'_gt.json'))['database']);index=read(e/(role+'_cache_index.json'));assert index['all_source_GT_covered'] and index['diagnostic_limit'] is None
        actual=set()
        for row in index['videos']:
            assert sha(row['path'])==row['sha256'];record=torch.load(row['path'],map_location='cpu');assert record['video_id']==row['video_id'] and record['video_id'] in ids;actual.add(record['video_id'])
            n=len(record['segments']);matched=record['instances'];assert set(range(n))<=set(matched.tolist())
            assert len(record['anchors'])==len(record['classes'])==len(matched)
        assert actual==ids
    assert set(read(e/'source_train_gt.json')['database']).isdisjoint(read(e/'source_val_gt.json')['database'])
    source=read(e/'source_termination.json');hist=read(e/'energy_history.json');assert not source['diagnostic'] and hist==source['history'];assert [z['epoch'] for z in hist]==list(range(len(hist)))
    best=float('inf');which=0
    for z in hist:
        assert math.isfinite(z['source_validation'])
        if z['source_validation']<best:best=z['source_validation'];which=z['epoch']
        if z is not hist[-1]:assert z['epoch']-which<15
    assert source['best_epoch']==which and (source['stop_epoch']==50 or source['stop_epoch']-which>=15)
    frozen=torch.load(e/'best_energy.pt',map_location='cpu');latest=torch.load(e/'latest_energy.pt',map_location='cpu');assert frozen['epoch']==which and latest['history']==hist and latest['cuda_rng'].numel()>0
    joint=e/'joint';t=joint_decision(read(joint/'source_history.json'));assert t['reason']!='running' and t==read(joint/'termination.json');saved=torch.load(joint/'best.pt',map_location='cpu');last=torch.load(joint/'latest.pt',map_location='cpu');assert saved['epoch']==t['best_epoch'] and saved['config']==cfg and last['epoch']==t['epoch'] and last['rng']['cuda'][0].numel()>0
    assert any(k.startswith('temporal_energy.') for k in saved['ema_state_dict'])
    assert last['model_controls']['temporal_energy_state']['step']>0
    a=read(e/'acceptance.json');j=read(e/'joint_acceptance.json');assert a['parameters_updated'] and a['head_gradient_norm']>0 and a['temporal_token_count']==40 and j['backbone_updated'] and j['actual_source_fpn_gradient'][0]['source_fpn_gradient_norm']>0
    gpu=read(e/'gpu_complete.json');guard=read(gpu['guard']);assert sha(gpu['guard'])==gpu['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256'];lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid'] and guard['observed']['uuid']==lease['observed']['uuid'] and guard['observed']['free_mib']>=lease['minimum_free_mib']>=12288
    assert not gpu['target_gt_used'] and not a['target_gt_used'] and not j['target_gt_used']
    for domain in ['source','target']:
        assert all('annotations' not in v for v in read(e/(domain+'_val_unlabeled.json'))['database'].values())
        expected=read(reference/'evaluation'/(domain+'_val.json'))['map_0.1_to_0.5'];assert abs(complete['results'][domain]['control']-expected)<1e-7
        for mode in ['control','frozen','joint']:
            ev=read(e/'evaluation'/(domain+'_'+mode+'.json'));assert ev['prediction_sha256']==sha(e/(domain+'_'+mode+'.npz')) and ev['map']==complete['results'][domain][mode]
    return dict(shift=e.name,complete=str(e/'complete.json'),complete_sha256=sha(e/'complete.json'),results=complete['results'],frozen_delta_pp=100*(complete['results']['target']['frozen']-complete['results']['target']['control']),joint_delta_pp=100*(complete['results']['target']['joint']-complete['results']['target']['control']))
