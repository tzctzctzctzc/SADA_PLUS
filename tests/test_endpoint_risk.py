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
    offsets=(torch.ones(1,64,2)*5).requires_grad_()
    student=dict(points=[grid],offsets=[offsets],masks=[torch.ones(1,1,64,dtype=torch.bool)],
        variance=[torch.ones(1,64,2,requires_grad=True)*4],logits=[torch.zeros(1,64,10)],
        metadata=[dict(video_id='t',fps=1.,feat_stride=1.,feat_num_frames=2,crop_start=0,crop_feat_len=64)])
    teacher=deepcopy({k:v for k,v in student.items() if k not in ('offsets','variance')})
    teacher.update(offsets=[(torch.ones(1,64,2)*7).requires_grad_()],variance=[torch.ones(1,64,2,requires_grad=True)])
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


def cpu_context(tmp_path, risk):
    from runpy import run_path
    from pathlib import Path
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer,make_scheduler
    from libs.trainer.da_trainer import Trainer
    from libs.utils import ModelEma2
    from libs.evaluation.metrics import MetricTracker
    from tools.endpoint_risk_engine import attach
    code=Path(__file__).resolve().parents[1]
    cfg=run_path(str(code/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['prototype_capacity']=run_path(str(code/'tests/test_prototype_capacity.py'))['configuration'](tmp_path,'control')
    cfg.update(seed=123);cfg['baseline']=dict(epoch_budget=dict(iterations_per_epoch=1))
    if risk:cfg['endpoint_risk']=options()
    ctx=object.__new__(Trainer);ctx.cfg=deepcopy(cfg);ctx.ema_active=True
    torch.manual_seed(123);ctx.model=SADA(cfg,'cpu');ctx.model_ema=ModelEma2(SADA(cfg,'cpu'),decay=.9)
    s,t=run_path(str(code/'tests/test_prototype_capacity.py'))['videos']()
    ctx.train_loader_src=[[s]];ctx.train_loader_tgt=[[t]]
    excludes=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings']
    for key,optname,schedname in [('source','optimizer_gen_source','lr_scheduler_gen_source'),
            ('alignment','optimizer_gen_source_alignment','lr_scheduler_gen_source_alignment'),
            ('discriminator','optimizer_disc','lr_scheduler_disc')]:
        disc=key=='discriminator'
        optimizer=make_optimizer(ctx.model,cfg['opt'],cfg['opt']['disc_learning_rate' if disc else 'learning_rate'],
            include_params=excludes if disc else [],exclude_params=[] if disc else excludes+(['endpoint_risk.'] if risk else []))
        setattr(ctx,optname,optimizer);setattr(ctx,schedname,make_scheduler(optimizer,cfg['opt'],1))
    ctx.train_metrics=MetricTracker('t_loss')
    if risk:attach(ctx,cfg)
    return ctx


@pytest.mark.parametrize('epoch',[1,5])
def test_true_original_training_loop_head_only_preserves_primary_and_rng(tmp_path,monkeypatch,epoch):
    from tools.baseline_engine import tensor_digest,optimizers
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        v=native(*args,**kwargs);return v.clone() if v.requires_grad and v.device.type=='cpu' else v
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    old=cpu_context(tmp_path,False);new=cpu_context(tmp_path,True)
    # 两次构造后固定同一输入，避免合成样本创建消耗的随机流差异。
    new.train_loader_src=deepcopy(old.train_loader_src);new.train_loader_tgt=deepcopy(old.train_loader_tgt)
    before=tensor_digest(new.model.endpoint_risk.state_dict())
    torch.manual_seed(777);old._train_epoch(epoch);rng=torch.get_rng_state().clone()
    torch.manual_seed(777);new._train_epoch(epoch)
    assert torch.equal(rng,torch.get_rng_state())
    for name,value in old.model.state_dict().items():
        assert torch.equal(value,new.model.state_dict()[name]),name
    for name,value in old.model_ema.module.state_dict().items():
        assert torch.equal(value,new.model_ema.module.state_dict()[name]),name
    assert tensor_digest([x.state_dict() for x in optimizers(old)])==tensor_digest([x.state_dict() for x in optimizers(new)])
    assert before!=tensor_digest(new.model.endpoint_risk.state_dict())
    assert all(p.grad is None for p in new.model_ema.module.parameters())
    assert new.train_metrics.result()['t_endpoint_head_gradient_norm']>0


def test_risk_optimizer_scheduler_independent_rng_strict_roundtrip(tmp_path,monkeypatch):
    import io
    from tools.endpoint_risk_engine import capture_extra,restore_extra
    from tools.baseline_engine import tensor_digest
    native=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *a,**kw:(lambda v:v.clone() if v.requires_grad and v.device.type=='cpu' else v)(native(*a,**kw)))
    ctx=cpu_context(tmp_path,True);ctx._train_epoch(1)
    saved=deepcopy(capture_extra(ctx));stream=io.BytesIO();torch.save(saved,stream);stream.seek(0);saved=torch.load(stream)
    torch.randperm(20,generator=ctx.endpoint_risk_generator);ctx.endpoint_risk_optimizer.param_groups[0]['lr']=.5
    restore_extra(ctx,saved)
    assert tensor_digest(capture_extra(ctx))==tensor_digest(saved)
    saved['options']['target_weight']=2.
    with pytest.raises(ValueError):restore_extra(ctx,saved)


def test_teacher_observation_matches_real_head_and_never_reads_gt(tmp_path):
    from tools.calibrate_prototype_geometry import source_dense_observation,prepare_source_video
    ctx=cpu_context(tmp_path,True);teacher=ctx.model_ema.module;teacher.eval()
    s=ctx.train_loader_src[0][0];clean,_=prepare_source_video(s,'cpu')
    before=torch.get_rng_state().clone()
    obs=teacher.endpoint_observation([clean],mode='test')
    dense,logits=source_dense_observation(teacher,clean)
    for a,b in zip(obs['offsets'],dense['offsets']):assert torch.equal(a,b)
    assert torch.equal(before,torch.get_rng_state())
    assert all(not x.requires_grad for x in obs['offsets']+obs['variance'])
    with pytest.raises(ValueError):teacher.endpoint_observation([s],mode='test')


def test_strict_model_restore_rejects_missing_risk_head(tmp_path):
    old=cpu_context(tmp_path,False);new=cpu_context(tmp_path,True)
    with pytest.raises(RuntimeError):new.model.load_state_dict(old.model.state_dict(),strict=True)
    new.model.load_state_dict(new.model.state_dict(),strict=True)


def test_reference_pair_rejects_budget_lr_seed_or_cache_change(tmp_path):
    from tools.endpoint_risk_protocol import config_for,assert_pair
    base=dict(seed=123,loader=dict(batch_size=2),opt=dict(learning_rate=.01),
        baseline=dict(code={'commit':'old'},tmpdir='/old',target_gt_used=False),
        prototype_capacity=dict(variant='control',prepared_cache_sha256='fixed'))
    cfg=config_for(base,options(),{'commit':'new'},tmp_path);assert_pair(cfg,base)
    for key,value in [('seed',6),('opt',dict(learning_rate=.02)),('loader',dict(batch_size=4))]:
        bad=deepcopy(cfg);bad[key]=value
        with pytest.raises(ValueError):assert_pair(bad,base)
