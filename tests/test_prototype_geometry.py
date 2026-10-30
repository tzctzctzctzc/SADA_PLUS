from copy import deepcopy
import hashlib
import json

import pytest
import torch

from libs.models.SADA.prototype_geometry import (boundary_dispersion, proposal_weights,
                                                GeometryBranch, POLICY, SCHEMA)
from libs.models.SADA.prototype_capacity import build_capacity


def observation(stride=2., length=24):
    x=torch.arange(length,dtype=torch.float32)*stride
    grid=torch.stack((x,x*0,x*0+1000,x*0+stride),-1)
    offsets=torch.stack(((x-0.)/stride,(60.-x)/stride),-1)[None].requires_grad_()
    return dict(points=[grid],offsets=[offsets],masks=[torch.ones(1,length,dtype=torch.bool)])


def test_real_points_inside_and_stride_decoded_once():
    obs=observation()
    row=boundary_dispersion(torch.tensor([10.,30.]),0,0,obs)
    assert row['count']==7 and row['available']==11
    x=obs['points'][0][row['indices'],0]
    assert bool(((x>=10)&(x<=30)).all()) and len(x.unique())==7
    torch.testing.assert_close(row['endpoints'],torch.tensor([[0.,60.]]).repeat(7,1))
    assert float(row['uncertainty'])==0.


def test_mask_and_short_proposal_are_not_fake_reliable():
    obs=observation();obs['masks'][0][0,6:]=False
    p=[dict(segments=torch.tensor([[10.,30.]]),levels=torch.tensor([0]))]
    row=proposal_weights(p,obs,.2)
    assert row['point_counts'].tolist()==[1] and row['uncertainty'].tolist()==[-1.]
    assert row['weights'].tolist()==[1.]  # 回退原版，不把-1作为低uncertainty。


def test_same_level_and_no_synthetic_interpolation():
    obs=observation();obs['points'].append(obs['points'][0]*2)
    obs['offsets'].append(obs['offsets'][0]);obs['masks'].append(obs['masks'][0])
    row=boundary_dispersion(torch.tensor([9.,13.]),1,0,obs)
    assert row['indices'].tolist()==[3] and row['uncertainty'] is None


def test_dispersion_normalized_by_duration_and_detached():
    obs=observation();obs['offsets'][0]=torch.zeros_like(obs['offsets'][0],requires_grad=True)
    seg=torch.tensor([10.,30.]);row=boundary_dispersion(seg,0,0,obs)
    expected=obs['points'][0][row['indices'],0].std(unbiased=False)/20.
    torch.testing.assert_close(row['uncertainty'],expected)
    p=[dict(segments=seg[None],levels=torch.tensor([0]))]
    q=proposal_weights(p,obs,float(expected))['weights']
    assert float(q[0])==pytest.approx(.5) and not q.requires_grad


def test_single_candidate_weight_not_renormalized():
    b=GeometryBranch(4,2)
    b.source_prototype_ready[0]=True;b.source_prototypes[0,0]=1.
    s=dict(embedding=torch.empty(0,128,requires_grad=True))
    t=dict(embedding=torch.nn.functional.normalize(torch.ones(1,128),dim=-1).requires_grad_())
    target=torch.empty(0,2)
    selection=dict(selected=torch.tensor([True]),labels=torch.tensor([0]),geometry_weights=torch.tensor([.25]))
    loss,_,_=b.prototype_alignment(s,target,t,selection)
    expected=.25*(1.-t['embedding'][0,0])
    torch.testing.assert_close(loss,expected)
    loss.backward();assert t['embedding'].grad.abs().sum()>0


@pytest.mark.parametrize('tau',[0.,-1.,float('nan')])
def test_bad_source_scale_rejected(tau):
    with pytest.raises(ValueError):proposal_weights([],observation(),tau)


def test_empty_proposals_and_invalid_regression():
    assert proposal_weights([],observation(),.3)['weights'].numel()==0
    obs=observation();obs['offsets'][0].data[0,5,0]=float('nan')
    with pytest.raises(ValueError):boundary_dispersion(torch.tensor([10.,30.]),0,0,obs)


def config(tmp_path,variant='geometry_weighted'):
    cache=tmp_path/'cache.json'
    cache.write_text(json.dumps(dict(shift='S1',by_video={'t':[dict(proposal_start_sec=16.,proposal_end_sec=36.,
        source_fpn_levels=[0],candidate_id='a',semantic_label='TAKE',class_probabilities=[.9]+[.1/9]*9,
        participation_selected=True)]})))
    proof=tmp_path/'calibration.json'
    proof.write_text(json.dumps(dict(schema=SCHEMA,policy=POLICY,target_gt_used=False,shift='S1',status='passed',tau=.25)))
    return dict(variant=variant,prepared_cache=str(cache),prepared_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        geometry_calibration=str(proof),geometry_calibration_sha256=hashlib.sha256(proof.read_bytes()).hexdigest())


def test_control_state_rng_strict_restore_and_target_gradients(tmp_path):
    cfg=config(tmp_path);plain=deepcopy(cfg);plain['variant']='control'
    torch.manual_seed(123);before=torch.get_rng_state().clone()
    g=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu')
    c=build_capacity(plain,8,10,[1],[[0,10000]],'cpu')
    assert torch.equal(before,torch.get_rng_state())
    assert set(g.state_dict())==set(c.state_dict())
    assert all(torch.equal(v,c.state_dict()[k]) for k,v in g.state_dict().items())
    from runpy import run_path
    from pathlib import Path
    videos=run_path(str(Path(__file__).with_name('test_prototype_capacity.py')))['videos']
    s,t=videos();sf=[s['feats'][None].requires_grad_()];tf=[t['feats'][None].requires_grad_()]
    obs=observation(1.,64)
    # 非负、有限且有离散度的普通回归距离。
    obs['offsets']=[torch.ones(1,64,2,requires_grad=True)]
    g.branch.source_prototypes[0,0]=1.;g.branch.source_prototype_ready[0]=True
    out=g([s],[t],sf,tf,5,target_geometry=obs)
    assert out['geometry_evidence']==1 and 0<out['geometry_weight_mean']<1
    assert out['alignment_loss']>0 and out['target_gradient_norm']>0
    out['loss'].backward()
    assert tf[0].grad.abs().sum()>0 and obs['offsets'][0].grad is None
    restored=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu')
    restored.load_state_dict(g.state_dict(),strict=True)
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in g.state_dict().items())


def test_calibration_tamper_and_missing_observation_rejected(tmp_path):
    cfg=config(tmp_path);g=build_capacity(cfg,8,10,[1],[[0,10000]],'cpu')
    with pytest.raises(ValueError):g.geometry_selection([],dict(),None)
    with open(cfg['geometry_calibration'],'a') as f:f.write(' ')
    with pytest.raises(ValueError):build_capacity(cfg,8,10,[1],[[0,10000]],'cpu')
