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
    selection=dict(selected=torch.tensor([True]),labels=torch.tensor([0]),geometry_weights=torch.tensor([.25]),
        geometry_point_counts=torch.tensor([7]),geometry_uncertainty=torch.tensor([.5]))
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


def test_true_sada_geometry_reuses_forward_and_preserves_detector(tmp_path,monkeypatch):
    from runpy import run_path
    from pathlib import Path
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    small_config=run_path(str(Path(__file__).with_name('test_corrected_baseline.py')))['small_config']
    videos=run_path(str(Path(__file__).with_name('test_prototype_capacity.py')))['videos']
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        v=native(*args,**kwargs)
        return v.clone() if v.requires_grad and v.device.type=='cpu' else v
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    cfg=small_config();cfg['prototype_capacity']=config(tmp_path)
    plain=deepcopy(cfg);plain['prototype_capacity']['variant']='control'
    torch.manual_seed(123);old=SADA(plain,'cpu')
    torch.manual_seed(123);new=SADA(cfg,'cpu')
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    calls=[]
    hook=new.reg_head.register_forward_hook(lambda *x:calls.append(1))
    s,t=videos()
    torch.manual_seed(777);before=old(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    torch.manual_seed(777);after=new(video_list_src=[s],video_list_tgt=[t],mode='train',curr_epoch=5)[0]
    hook.remove();assert len(calls)==3  # source/target/pl原有三次，无新增检测前向。
    assert torch.equal(before['cls_loss'],after['cls_loss']) and torch.equal(before['reg_loss'],after['reg_loss'])
    assert after['rq3_alignment_loss']<before['rq3_alignment_loss'] and after['rq3_target_gradient_norm']>0
    opt=make_optimizer(new,cfg['opt'],.0001,exclude_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    ids={id(p) for g in opt.param_groups for p in g['params']}
    assert all(id(p) in ids for p in new.rq3_capacity.parameters())
    w=new.rq3_capacity.branch.input_projection.weight.detach().clone()
    after['downstream_loss'].backward();opt.step()
    assert not torch.equal(w,new.rq3_capacity.branch.input_projection.weight)


def test_source_partition_tau_and_source_only_gate():
    from tools.prototype_geometry_protocol import source_partitions,source_summary,mechanism_gate
    fit,check=source_partitions(['a','b','c','d'])
    assert not fit&check
    rows=[dict(video_id=v,level=0,duration=10.,confidence=.8,
        uncertainty=float(i),iou=1.-i/10) for v in fit|check for i in range(1,5)]
    r=source_summary(rows);assert r['tau']==2.5 and r['audit_concordance']==1.
    reports={'S'+str(i):{'summary':r} for i in range(1,7)}
    assert mechanism_gate(reports)['status']=='passed'
    for row in rows:row['iou']=row['uncertainty']/10.
    bad=source_summary(rows);reports={s:{'summary':bad} for s in reports}
    assert mechanism_gate(reports)['status']=='no_source_mechanism_signal'
    with pytest.raises(ValueError):source_summary([dict(x,uncertainty=0.) for x in rows])


def test_pairing_rejects_method_or_training_changes(tmp_path):
    from tools.prototype_geometry_protocol import config_for,assert_pair
    proof=tmp_path/'cal.json';proof.write_text(json.dumps(dict(code={'commit':'new'},shift='S1')))
    old=dict(seed=123,loader=dict(batch_size=2),opt={'learning_rate':.1},
        baseline=dict(code={'commit':'old'},shift='S1',tmpdir='/old',target_gt_used=False),
        prototype_capacity=dict(variant='control',specification={},prepared_cache='/f',prepared_cache_sha256='sha'))
    cfg=config_for(old,proof,{'commit':'new'},tmp_path);assert_pair(cfg,old)
    cfg['opt']['learning_rate']=.2
    with pytest.raises(ValueError):assert_pair(cfg,old)


def test_source_dense_calibration_matches_actual_test_forward():
    from runpy import run_path
    from pathlib import Path
    from libs.models.SADA.meta_archs import SADA
    from tools.calibrate_prototype_geometry import source_dense_observation
    cfg=run_path(str(Path(__file__).with_name('test_corrected_baseline.py')))['small_config']()
    video=run_path(str(Path(__file__).with_name('test_prototype_capacity.py')))['videos']()[0]
    video['metadata']['duration']=64.;video['metadata']['video_length']=64
    video['metadata']['use_temporal_grid_feats']=False
    # inference读取与正式dataset相同的顶层元数据。
    for key in ['video_id','fps','duration','feat_stride','feat_num_frames']:
        video[key]=video['metadata'][key]
    video['use_temporal_grid_feats']=False
    model=SADA(cfg,'cpu');model.eval();captured=[]
    hook=model.reg_head.register_forward_hook(lambda m,args,out:captured.append([v.detach().clone() for v in out[0]]))
    with torch.no_grad():
        model(video_list_src=[video],video_list_tgt=[video],mode='test',domain='source',curr_epoch=5)
        before=torch.get_rng_state().clone()
        observation,_=source_dense_observation(model,video)
    hook.remove();assert torch.equal(before,torch.get_rng_state())
    for reference,actual in zip(captured[0],observation['offsets']):
        torch.testing.assert_close(reference.permute(0,2,1),actual,rtol=0,atol=0)


def test_calibration_single_video_transfer_and_gt_copy():
    from tools.calibrate_prototype_geometry import prepare_source_video
    video=dict(feats=torch.ones(4,64),segments=torch.tensor([[10.,20.]]),labels=torch.tensor([0]),
               metadata={'video_id':'source'})
    clean,gt=prepare_source_video(video,'cpu')
    assert clean['segments'].numel()==clean['labels'].numel()==0
    assert torch.equal(gt,video['segments']) and gt.data_ptr()!=video['segments'].data_ptr()
    assert video['labels'].tolist()==[0] and video['segments'].shape==(1,2)


@pytest.mark.skipif(__import__('os').name=='nt',reason='Linux实际进程组停止')
def test_cleanup_only_stops_owned_process_group():
    import subprocess,sys
    from tools import run_prototype_geometry as queue
    class Context:
        released=False
        def __exit__(self,*args):self.released=True
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],start_new_session=True)
    context=Context();queue.ACTIVE['test']=dict(process=child,context=context)
    try:
        queue.cleanup()
        assert child.poll() is not None and context.released and not queue.ACTIVE
    finally:
        if child.poll() is None:child.kill();child.wait()
