from copy import deepcopy
import numpy as np
import torch
from tools.background_risk import BackgroundRisk,video_split,calibrated,fit_calibration,negative_loss,balanced_negative_BCE
from tools.background_risk_signals import weighted_metrics,aggregate,group_readiness
from tools.background_risk_source import result_record,restore_source


def test_background_head_detaches_detector_and_has_one_output():
    head=BackgroundRisk(4);x=torch.randn(1,4,12,requires_grad=True);p=torch.randn(1,10,12,requires_grad=True)
    y=head([x],[p],[torch.ones(1,1,12,dtype=torch.bool)])[0]
    assert y.shape==(1,12,1)
    y.square().mean().backward()
    assert x.grad is None and p.grad is None and all(v.grad is not None for v in head.parameters())


def test_padding_values_do_not_leak():
    head=BackgroundRisk(4);x=torch.randn(1,4,9);p=torch.randn(1,10,9)
    m=torch.tensor([[[1,1,1,1,0,0,0,0,0]]],dtype=torch.bool)
    a=head([x],[p],[m])[0];x[:,:,4:]=1e7;p[:,:,4:]=-1e7
    assert torch.equal(a,head([x],[p],[m])[0])


def test_source_video_split_is_disjoint_complete_and_order_independent():
    ids=['v%d'%i for i in range(51)];a=video_split(ids);b=video_split(ids[::-1])
    assert a==b and len(a['calibration'])==10
    assert not set(a['fit'])&set(a['calibration']) and set(a['fit'])|set(a['calibration'])==set(ids)


def test_calibration_fits_source_errors_and_preserves_ranking_rng():
    x=torch.linspace(-3,3,100);bg=(x>1).float();rng=torch.get_rng_state().clone()
    c=fit_calibration([dict(video='source',logits=x,background=bg)])
    assert c['slope']>0 and c['after_NLL']<c['before_NLL']
    assert torch.equal(rng,torch.get_rng_state())
    q=calibrated(x,c);assert (q[1:]>=q[:-1]).all() and c['videos']==['source']


def test_no_finite_threshold_required_weights_smooth_and_bounded():
    c=dict(slope=.5,bias=-1);x=torch.tensor([-100.,0.,100.])
    q=calibrated(x,c).square();assert (q>=0).all() and (q<=1).all() and 0<q[1]<1


def test_negative_loss_updates_only_candidate_classes_and_detaches_weights():
    z=torch.zeros(3,10,requires_grad=True);w=torch.tensor([.2,.8,.5],requires_grad=True)
    loss=negative_loss(z,w,torch.tensor([1,0,1],dtype=torch.bool),2.)
    loss.backward();assert (z.grad[[0,2]]>0).all() and torch.count_nonzero(z.grad[1])==0 and w.grad is None


def test_empty_negative_set_zero_loss_and_gradient():
    z=torch.randn(4,10,requires_grad=True);loss=negative_loss(z,torch.ones(4),torch.zeros(4,dtype=torch.bool),1.)
    loss.backward();assert float(loss)==0 and torch.count_nonzero(z.grad)==0


def test_equal_weight_distribution_control_catches_real_order_gain():
    p=np.full((4,10),.001,np.float32);p[:,0]=[.01,.02,.03,.04]
    raw=np.array([-2.,-1.,1.,2.],np.float32);occupied=np.array([1,1,0,0],bool)
    classes=np.zeros((4,10));classes[occupied,0]=1
    result=weighted_metrics(p,raw,np.ones(4,bool),~occupied,occupied,classes,dict(slope=1.,bias=0.))
    assert result['gain_vs_uniform']>0 and result['gain_vs_same_weight_confidence']>0
    assert 0<result['weight_mass']<1 and 0<result['effective_fraction']<=1


def test_constant_weights_do_not_fake_quality_improvement():
    p=np.full((10,10),.01,np.float32);occ=np.arange(10)%2==0
    r=weighted_metrics(p,np.zeros(10,np.float32),np.ones(10,bool),~occ,occ,np.zeros((10,10)),dict(slope=1.,bias=0.))
    assert r['gain_vs_uniform']==r['gain_vs_same_weight_confidence']==0
    a=aggregate([dict(r,video='v')]);assert not group_readiness([dict(validation=a)]*6)


def test_group_readiness_uses_all_six_not_best_shift_or_zero_mass():
    v=dict(candidate_videos=1,weight_mass=.3,effective_fraction=.5,gain_vs_uniform=.02,gain_vs_same_weight_confidence=.01)
    reports=[dict(validation=deepcopy(v)) for _ in range(6)]
    assert group_readiness(reports) and not group_readiness(reports[:5])
    reports[0]['validation']['weight_mass']=0
    assert not group_readiness(reports)


def test_source_result_separates_calibration_quality_from_calibrator():
    metrics=dict(fit={'loss':.2},calibration={'error':.1},validation={'error':.15})
    calibration=dict(slope=1.,bias=-.2)
    result=result_record(metrics,calibration,target_GT_used=False)
    assert result['calibration']==calibration and result['calibration_quality']==metrics['calibration']
    import json
    assert json.loads(json.dumps(result))['validation']==metrics['validation']


def test_source_resume_preserves_adamw_shuffle_and_cpu_random_streams(monkeypatch):
    import random
    monkeypatch.setattr(torch.cuda,'set_rng_state',lambda *args,**kwargs:None)
    torch.manual_seed(3);random.seed(3);np.random.seed(3)
    head=BackgroundRisk(4);optimizer=torch.optim.AdamW(head.parameters(),lr=1e-3)
    generator=torch.Generator().manual_seed(6)
    def step():
        order=torch.randperm(5,generator=generator)
        x=torch.randn(1,4,9)*(random.random()+float(np.random.rand()))
        optimizer.zero_grad();value=head([x],[torch.zeros(1,10,9)],[torch.ones(1,1,9,dtype=torch.bool)])[0].square().mean()
        value.backward();optimizer.step();return value.detach(),order
    step();saved=deepcopy(dict(state_dict=head.state_dict(),optimizer=optimizer.state_dict(),generator=generator.get_state(),
        cpu_rng=torch.get_rng_state(),cuda_rng=torch.zeros(8,dtype=torch.uint8),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
    expected=[step() for _ in range(3)];weights=deepcopy(head.state_dict())
    restore_source(head,optimizer,generator,saved)
    actual=[step() for _ in range(3)]
    assert all(torch.equal(x[0],y[0]) and torch.equal(x[1],y[1]) for x,y in zip(expected,actual))
    assert all(torch.equal(v,head.state_dict()[k]) for k,v in weights.items())


def test_rare_oracle_positive_points_have_nonvanishing_training_weight():
    raw=torch.zeros(100,requires_grad=True);negative=torch.ones(100,dtype=torch.bool);negative[0]=False
    loss=balanced_negative_BCE(raw,negative,torch.ones(100,dtype=torch.bool));loss.backward()
    assert raw.grad[0]>0 and (raw.grad[1:]<0).all()
    assert torch.allclose(raw.grad[0].abs(),raw.grad[1:].abs().sum())


def test_physical_background_gain_cannot_override_oracle_positive_harm():
    # 点0是检测正点，点1为动作边缘（物理前景但原检测负点）。
    p=np.full((4,10),.001,np.float32);p[:,0]=[.09,.01,.02,.03]
    classes=np.zeros((4,10));classes[0,0]=1
    occupancy=np.array([1,1,0,0],bool);raw=np.array([2.,-2.,2.,2.],np.float32)
    r=weighted_metrics(p,raw,np.ones(4,bool),~classes.any(-1),occupancy,classes,dict(slope=1.,bias=0.))
    assert r['physical_gain_vs_uniform']>0 and r['gain_vs_uniform']<0
