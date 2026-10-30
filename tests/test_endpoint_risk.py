from copy import deepcopy
import pytest
import torch
from libs.models.SADA.endpoint_risk import (EndpointRisk,SCHEMA,POLICY,risk_nll,
    calibrated_gate,endpoint_loss,decode_at)
from tools.endpoint_risk_protocol import calibrate,assert_formal_ready


def options(mode='head_only'):
    return dict(schema=SCHEMA,policy=deepcopy(POLICY),mode=mode,head_learning_rate=.001,
                head_clip=1.,target_weight=.1,warmup_epochs=5)


def certificate():
    return dict(schema=SCHEMA,policy=deepcopy(POLICY),status='passed',target_gt_used=False,
        fit_videos=['fit1','fit2'],audit_videos=['audit1','audit2','audit3'],margin=0.,
        factors={'0':dict(available=[True,True],student=[1.,1.],teacher=[1.,1.])})


def test_head_initialization_preserves_cpu_rng():
    torch.manual_seed(123);before=torch.get_rng_state().clone();EndpointRisk(8,options())
    assert torch.equal(before,torch.get_rng_state())


def test_source_nll_learns_residual_without_shared_gradient():
    head=EndpointRisk(4,options());feature=torch.randn(2,4,6,requires_grad=True)
    mean=torch.randn(2,6,2,requires_grad=True);target=torch.randn(2,6,2,requires_grad=True)
    variance=head([feature])[0];loss=risk_nll(variance,mean,target,torch.ones(2,6,dtype=torch.bool))
    loss.backward()
    assert feature.grad is None and mean.grad is None and target.grad is None
    assert head.head.weight.grad.abs().sum()>0


def test_nll_cannot_escape_by_infinite_variance_and_ignores_background():
    mean=torch.tensor([[[2.,2.],[999.,999.]]]);target=torch.zeros_like(mean)
    mask=torch.tensor([[True,False]])
    a=risk_nll(torch.full_like(mean,4.),mean,target,mask)
    b=risk_nll(torch.full_like(mean,1e9),mean,target,mask)
    assert b>a
    assert torch.equal(a,risk_nll(torch.tensor([[[4.,4.],[.01,.01]]]),mean,target,mask))


def test_empty_source_has_differentiable_zero():
    v=torch.ones(1,3,2,requires_grad=True)
    zero=risk_nll(v,v,torch.zeros_like(v),torch.zeros(1,3,dtype=torch.bool));zero.backward()
    assert zero==0 and v.grad.abs().sum()==0


def test_calibrated_relative_decision_is_per_endpoint_and_detached():
    s=torch.tensor([[2.,1.]],requires_grad=True);t=torch.tensor([[1.,3.]],requires_grad=True)
    gate,gain=calibrated_gate(s,t,torch.tensor([0]),torch.tensor([10.]),certificate())
    assert gate.tolist()==[[True,False]] and not gain.requires_grad
    cert=certificate();cert['factors']['0']['teacher'][0]=3.
    assert not calibrated_gate(s,t,torch.tensor([0]),torch.tensor([10.]),cert)[0].any()


def test_missing_source_level_has_no_automatic_fallback():
    gate,_=calibrated_gate(torch.ones(1,2)*2,torch.ones(1,2),torch.tensor([3]),torch.tensor([10.]),certificate())
    assert not gate.any()


def test_single_selected_side_keeps_original_denominator_and_no_teacher_grad():
    student=torch.tensor([[1.,4.]],requires_grad=True);teacher=torch.tensor([[0.,8.]],requires_grad=True)
    duration=torch.tensor([2.],requires_grad=True)
    loss=endpoint_loss(student,teacher,torch.tensor([[True,False]]),duration,1)
    assert float(loss)==pytest.approx(.0625)
    loss.backward();assert student.grad[0,0]!=0 and student.grad[0,1]==0
    assert teacher.grad is None and duration.grad is None


def test_offset_and_sigma_stride_units_are_decoded_once():
    obs=dict(points=[torch.tensor([[12.,0.,100.,4.]])],
        offsets=[torch.tensor([[[2.,3.]]])],variance=[torch.tensor([[[4.,9.]]])])
    boundary,sigma=decode_at(obs,0,0,0)
    assert boundary.tolist()==[4.,24.] and sigma.tolist()==[8.,12.]


def source_rows():
    rows=[]
    for video in ['fit1','fit2']:
        rows.append(dict(video_id=video,candidate_id='fit',level=0,duration=10.,
            student_sigma=[1.,1.],teacher_sigma=[1.,1.],student_error=[1.,1.],teacher_error=[1.,1.]))
    for video in ['audit1','audit2','audit3']:
        rows.extend([dict(video_id=video,candidate_id='good',level=0,duration=10.,
            student_sigma=[2.,2.],teacher_sigma=[.5,.5],student_error=[4.,4.],teacher_error=[1.,1.]),
            dict(video_id=video,candidate_id='bad',level=0,duration=10.,
            student_sigma=[.5,.5],teacher_sigma=[2.,2.],student_error=[1.,1.],teacher_error=[3.,3.])])
    return rows


def test_source_gate_requires_real_teacher_advantage_over_matched_random():
    rows=source_rows();cert=calibrate(rows,['fit1','fit2'],['audit1','audit2','audit3'],['train'],0.)
    assert cert['status']=='passed' and cert['source_audit']['mean_advantage_over_random']>0
    for row in rows[2:]:row['student_error'],row['teacher_error']=row['teacher_error'],row['student_error']
    assert calibrate(rows,['fit1','fit2'],['audit1','audit2','audit3'],['train'],0.)['status']!='passed'


def test_source_calibration_does_not_read_audit_errors():
    rows=source_rows();a=calibrate(rows,['fit1','fit2'],['audit1','audit2','audit3'],['train'],0.)
    for row in rows[2:]:row['teacher_error']=[100.,100.]
    b=calibrate(rows,['fit1','fit2'],['audit1','audit2','audit3'],['train'],0.)
    assert a['factors']==b['factors'] and b['status']!='passed'


@pytest.mark.parametrize('fit,audit,train',[(['fit1'],['fit1'],[]),(['fit1'],['audit1'],['audit1'])])
def test_video_isolation_rejects_crop_style_split(fit,audit,train):
    with pytest.raises(ValueError):calibrate(source_rows(),fit,audit,train,0.)


@pytest.mark.parametrize('key,value',[('head_learning_rate',0.),('target_weight',float('nan')),
                                      ('mode','mystery'),('warmup_epochs',-1)])
def test_unfrozen_invalid_options_rejected(key,value):
    spec=options();spec[key]=value
    with pytest.raises(ValueError):EndpointRisk(4,spec)


def test_target_fixture(tmp_path):
    # 一个真实grid点对应一个原名单proposal，不需要GT。
    from tools.baseline_protocol import save
    head=EndpointRisk(4,options('relative'));head.install_calibration(certificate())
    head.options['warmup_epochs']=0
    x=torch.arange(64.).reshape(-1,1)
    grid=torch.cat((x,x*0,x*0+10000,x*0+1),1)
    offsets=torch.ones(1,64,2,requires_grad=True)*5
    student=dict(points=[grid],offsets=[offsets],masks=[torch.ones(1,1,64,dtype=torch.bool)],
        variance=[torch.ones(1,64,2,requires_grad=True)*4],logits=[torch.zeros(1,64,10)],
        metadata=[dict(video_id='t',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64)])
    teacher=deepcopy({k:v for k,v in student.items() if k not in ('offsets','variance')})
    teacher.update(offsets=[torch.ones(1,64,2,requires_grad=True)*7],variance=[torch.ones(1,64,2,requires_grad=True)])
    teacher['logits'][0][:,:,0]=10
    video=dict(feats=torch.randn(4,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long),metadata=student['metadata'][0])
    row=dict(proposal_start_sec=16.,proposal_end_sec=36.,source_fpn_levels=[0],candidate_id='a',
        semantic_label='TAKE',class_probabilities=[.9]+[.1/9]*9,participation_selected=True)
    cache={'t':[row]};before=deepcopy(cache)
    loss=head.target_loss([video],cache,0.,student,teacher,6)
    assert loss>0 and head.statistics['selected_endpoints']==2 and cache==before
    loss.backward();assert teacher['offsets'][0].grad is None
    assert head.head.weight.grad is None and head.head.bias.grad is None
    assert head.statistics['pseudo_gradient_norm']>0
    teacher['logits'][0][:,:,1]=20
    assert head.target_loss([video],cache,0.,student,teacher,6)==0
    video['labels']=torch.tensor([0])
    with pytest.raises(ValueError):head.target_loss([video],cache,0.,student,teacher,6)
