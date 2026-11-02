from copy import deepcopy
import numpy as np
import torch
from tools.background_risk import BackgroundRisk,video_split,calibrated,fit_calibration,negative_loss
from tools.background_risk_signals import weighted_metrics,aggregate,group_readiness
from tools.background_risk_source import result_record


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
    result=weighted_metrics(p,raw,np.ones(4,bool),~occupied,occupied,np.zeros((4,10)),dict(slope=1.,bias=0.))
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
