import copy
import pytest
import torch
from torch import nn
from libs.models.SADA.phase_alignment import (role_probabilities, conditioned,
    canonical_initial_state, expand_reference_state, expand_discriminators, balanced_domain_losses, SCHEMA, POLICY)


def test_roles_have_temporal_order_and_are_detached():
    offsets=torch.tensor([[0.,2.],[1.,3.],[2.,2.],[3.,1.],[2.,0.],[0.,0.]],requires_grad=True)
    q,u=role_probabilities(offsets)
    assert torch.equal(q,torch.tensor([[1.,0.,0.],[.5,.5,0.],[0.,1.,0.],[0.,.5,.5],[0.,0.,1.],[0.,1.,0.]]))
    assert not q.requires_grad
    assert torch.equal(q.sum(-1),torch.ones(6))
    assert torch.allclose(u,torch.tensor([0.,.25,.5,.75,1.,.5]))


def test_repeat_initialization_preserves_function_feature_gradient_and_rng():
    model=nn.Module();model.cfg={'phase_alignment':dict(schema=SCHEMA,policy=POLICY),'model':{'weight_sharing':True}}
    model.use_trident_head=False
    class Disc(nn.Module):
        def __init__(self):
            super().__init__();self.initial_layers=nn.Sequential(nn.Linear(8,5),nn.ReLU(),nn.Dropout(0))
        def forward(self,x):return self.initial_layers(x)
    model.disc_dom_conditional_adv=nn.ModuleList([Disc()]);old=copy.deepcopy(model)
    before=torch.get_rng_state();expand_discriminators(model)
    assert torch.equal(before,torch.get_rng_state())
    canonical=canonical_initial_state(model.state_dict())
    assert all(torch.equal(v,old.state_dict()[k]) for k,v in canonical.items())
    x=torch.randn(7,8,requires_grad=True);y=x.detach().clone().requires_grad_()
    q,_=role_probabilities(torch.rand(7,2))
    a=old.disc_dom_conditional_adv[0](x);b=model.disc_dom_conditional_adv[0](conditioned(y,q))
    assert torch.allclose(a,b,atol=1e-6)
    a.sum().backward();b.sum().backward();assert torch.allclose(x.grad,y.grad,atol=1e-6)
    opt=torch.optim.AdamW(model.parameters(),lr=.01);opt.step()
    w=model.disc_dom_conditional_adv[0].initial_layers[0].weight
    assert not torch.equal(w[:,:8],w[:,8:16])
    assert not torch.allclose(model.disc_dom_conditional_adv[0](conditioned(y,q)),
                              model.disc_dom_conditional_adv[0](conditioned(y,q.flip(1))))
    restored=copy.deepcopy(model);restored.load_state_dict(model.state_dict(),strict=True)


def test_empty_and_invalid():
    q,u=role_probabilities(torch.empty(0,2));assert conditioned(torch.empty(0,8),q).shape==(0,24)
    with pytest.raises(ValueError):role_probabilities(torch.tensor([[-1.,2.]]))
    with pytest.raises(ValueError):role_probabilities(torch.tensor([[float('nan'),2.]]))


def test_reference_expansion_only_changes_conditional_first_weights():
    state={'disc_dom_conditional_adv.0.initial_layers.0.weight':torch.randn(3,4),'backbone.weight':torch.randn(3,3)}
    expanded=expand_reference_state(state)
    assert expanded['backbone.weight'] is state['backbone.weight']
    assert torch.equal(canonical_initial_state(expanded)['disc_dom_conditional_adv.0.initial_layers.0.weight'],state['disc_dom_conditional_adv.0.initial_layers.0.weight'])


def test_role_balance_removes_frequency_only_domain_shortcut():
    # 两域特征完全相同，仅role频率不同；每个role条件loss必须不依赖频率。
    s=torch.ones(4,2,requires_grad=True);t=torch.ones(4,2,requires_grad=True)
    qs=torch.tensor([[1.,0.,0.]]*3+[[0.,0.,1.]])
    qt=torch.tensor([[1.,0.,0.]]+[[0.,0.,1.]]*3)
    predict=lambda x:x[:,:2].sum(-1,keepdim=True)-x[:,4:].sum(-1,keepdim=True)
    a,b,n=balanced_domain_losses(predict,s,t,qs,qt)
    c,d,m=balanced_domain_losses(predict,s,t,qt,qs)
    assert n==m==2 and torch.equal(a,c) and torch.equal(b,d)
    (a+b).backward();assert s.grad.abs().sum()>0 and t.grad.abs().sum()>0


def test_empty_domain_falls_back_without_nan():
    s=torch.ones(2,2,requires_grad=True);t=torch.empty(0,2,requires_grad=True)
    qs=torch.tensor([[1.,0.,0.],[0.,1.,0.]]);qt=torch.empty(0,3)
    a,b,n=balanced_domain_losses(lambda x:x.sum(-1,keepdim=True),s,t,qs,qt)
    assert n==0 and a>0 and b==0 and torch.isfinite(a+b)


def test_real_sada_forward_backward_optimizers_strict_restore(tmp_path,monkeypatch):
    from runpy import run_path
    from tools.baseline_protocol import CODE
    from libs.models.SADA.meta_archs import SADA
    from libs.models.SADA.optimizer import make_optimizer
    helper=run_path(str(CODE/'tests/test_prototype_capacity.py'))
    cfg=run_path(str(CODE/'tests/test_corrected_baseline.py'))['small_config']()
    cfg['prototype_capacity']=helper['configuration'](tmp_path,'control')
    cfg['model']['pl_threshold']=0.
    native=torch.tensor
    def cpu_copy(*args,**kwargs):
        value=native(*args,**kwargs)
        return value.clone() if value.requires_grad and value.device.type=='cpu' else value
    monkeypatch.setattr(torch,'tensor',cpu_copy)
    torch.manual_seed(123);base=SADA(copy.deepcopy(cfg),'cpu');before=torch.get_rng_state()
    new_cfg=copy.deepcopy(cfg);new_cfg['phase_alignment']=dict(schema=SCHEMA,policy=POLICY)
    torch.manual_seed(123);model=SADA(new_cfg,'cpu')
    assert torch.equal(before,torch.get_rng_state())
    canonical=canonical_initial_state(model.state_dict())
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in canonical.items())
    for m in [base,model]:
        with torch.no_grad():
            m.cls_head.cls_head.conv.bias.fill_(-10);m.cls_head.cls_head.conv.bias[0]=10
    source,target=helper['videos']()
    outputs=[]
    for m in [base,model]:
        torch.manual_seed(456)
        outputs.append(m(video_list_src=[source],video_list_tgt=[target],mode='train',curr_epoch=5)[0])
    old,new=outputs
    for name in ['cls_loss','reg_loss','rq3_loss','downstream_loss']:
        assert torch.equal(old[name],new[name])
    assert sum(float(v) for k,v in new.items() if k.startswith('phase_common_roles_'))>0
    # 新DA不能通过回归预测的条件坐标反传，改变reg head以逃避D。
    reg=model.reg_head.offset_head.conv.weight
    assert torch.autograd.grad(new['loss_conditional_adv_rev'],reg,allow_unused=True,retain_graph=True)[0] is None
    opt=make_optimizer(model,cfg['opt'],.0001,include_params=['disc_dom_adv','disc_dom_conditional_adv','learnable_embeddings'])
    d=model.disc_dom_conditional_adv[0].initial_layers[0].weight;initial=d.detach().clone()
    (new['downstream_loss']+new['loss_conditional_adv_rev']).backward();opt.step()
    assert not torch.equal(d,initial)
    assert model.backbone_source.embd[0].conv.weight.grad.abs().sum()>0
    restored=SADA(new_cfg,'cpu');restored.load_state_dict(model.state_dict(),strict=True)


def test_pair_rejects_method_parameters_or_old_schema():
    from tools.phase_alignment_protocol import assert_pair,source_decision
    base=dict(seed=123,loader={'batch_size':2},prototype_capacity={'variant':'control'},
              baseline=dict(code={'commit':'old'},tmpdir='/old',target_gt_used=False))
    cfg=copy.deepcopy(base);cfg['baseline'].update(code={'commit':'new'},tmpdir='/new')
    cfg['phase_alignment']=dict(schema=SCHEMA,policy=copy.deepcopy(POLICY))
    assert_pair(cfg,base)
    bad=copy.deepcopy(cfg);bad['loader']['batch_size']=4
    with pytest.raises(ValueError):assert_pair(bad,base)
    bad=copy.deepcopy(cfg);bad['phase_alignment']['policy']['target']='target_gt'
    with pytest.raises(ValueError):assert_pair(bad,base)
    reports=[dict(shift='S'+str(i),rows=[{},{}],positive_points=100,true_phase_std=.1,
                  video_equal_constant_mae=.1,video_equal_mae=.05,side_role_support_videos=[2,2]) for i in range(1,7)]
    assert source_decision(reports)['status']=='passed'
    reports[0]['positive_points']=0
    assert source_decision(reports)['status']=='rejected_uninformative_source_phase'


def test_reference_target_evaluation_drift_is_rejected(tmp_path):
    from tools.phase_alignment_protocol import prepare
    from tools import baseline_protocol as p
    shared=tmp_path/'input.json';shared.write_text('{}');digest=p.sha(shared)
    control=tmp_path/'control';results=[]
    for shift in ['S'+str(i) for i in range(1,7)]:
        ref=control/'formal/control'/shift;ref.mkdir(parents=True)
        base=dict(seed=123,loader={'batch_size':2},dataset={'json_file':str(shared)},
            prototype_capacity=dict(variant='control',prepared_cache=str(shared),prepared_cache_sha256=digest),
            baseline=dict(code={'commit':'old'},tmpdir='/old',target_gt_used=False,
                source_manifest_sha256=digest,target_manifest=str(shared),target_manifest_sha256=digest))
        p.save(ref/'config.json',base);p.save(ref/'evaluation/target_val.json',{'map_0.1_to_0.5':.5})
        p.save(ref/'detector_initialization.json',{'sha256':'cold'});(ref/'best.pt').write_text('weights')
        paths=[ref/n for n in ['config.json','evaluation/target_val.json','detector_initialization.json','best.pt']]
        p.save(ref/'complete.json',dict(artifacts={str(q):p.sha(q) for q in paths}))
        results.append(dict(shift=shift,variant='control',complete=str(ref/'complete.json'),complete_sha256=p.sha(ref/'complete.json')))
    p.save(control/'final_audit.json',dict(status='passed',count=12,results=results))
    assert len(prepare(tmp_path/'new',control,{'commit':'new'}))==6
    (control/'formal/control/S1/evaluation/target_val.json').write_text('drift')
    with pytest.raises(ValueError,match='冻结原输入改变'):prepare(tmp_path/'bad',control,{'commit':'new'})


def test_formal_admission_uses_measured_peak_and_same_lock(tmp_path,monkeypatch):
    from tools import run_phase_alignment as runner
    from tools import baseline_protocol as p
    e=tmp_path/'entry';e.mkdir();p.save(e/'probe_result.json',dict(peak_reserved_mib=9000))
    lease=tmp_path/'lease.json';p.save(lease,dict(lock='/data/.rq2_gpu0.lock'))
    task=dict(entry=e,lease=dict(index=0,uuid='gpu',lease=str(lease)))
    monkeypatch.setattr(runner,'snapshot',lambda index:dict(uuid='gpu',free_mib=10000))
    required=runner.formal_admission(task,{'peak_reserved_mib':9000},tmp_path,{'commit':'clean'},8192)
    record=p.read(task['lease']['lease'])
    assert required==11048 and record['minimum_free_mib']==11048
    assert record['lock']=='/data/.rq2_gpu0.lock' and p.sha(task['lease']['lease'])==task['lease']['lease_sha256']


def test_stop_cleans_real_owned_process_groups():
    import os,sys,subprocess
    if os.name!='posix':pytest.skip('真实进程组只在Linux验证')
    from tools import run_phase_alignment as runner
    children=[subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True) for _ in range(2)]
    saved=runner.ACTIVE
    try:
        runner.ACTIVE={str(i):dict(process=p) for i,p in enumerate(children)}
        runner.terminate_owned()
        assert all(p.poll() is not None for p in children)
    finally:
        runner.ACTIVE=saved
        for p in children:
            if p.poll() is None:p.kill();p.wait()


@pytest.mark.parametrize('drift',[None,'report','receipt','decision'])
def test_source_gate_recomputes_six_reports_and_receipts(tmp_path,drift):
    from tools.phase_alignment_protocol import source_decision,verify_source_gate
    from tools import baseline_protocol as p
    reports=[];paths={};receipts={}
    for i in range(1,7):
        e=tmp_path/('S'+str(i));e.mkdir()
        report=dict(shift='S'+str(i),rows=[{},{}],positive_points=100,true_phase_std=.1,
            video_equal_constant_mae=.1,video_equal_mae=.05,side_role_support_videos=[2,2])
        p.save(e/'source_report.json',report);reports.append(report)
        p.save(e/'config.json',{});p.save(e/'lease.json',{})
        p.save(e/'guard.json',dict(lease=str(e/'lease.json'),lease_sha256=p.sha(e/'lease.json')))
        p.save(e/'sourcecheck_gpu.json',dict(guard=str(e/'guard.json'),guard_sha256=p.sha(e/'guard.json'),config_sha256=p.sha(e/'config.json')))
        paths[str(e/'source_report.json')]=p.sha(e/'source_report.json')
        receipts[str(e/'sourcecheck_gpu.json')]=p.sha(e/'sourcecheck_gpu.json')
    gate=source_decision(reports);gate.update(reports=paths,gpu_receipts=receipts)
    if drift=='decision':gate['equal_shift_mae_advantage']=1.
    p.save(tmp_path/'gate.json',gate)
    if drift=='report':p.save(tmp_path/'S1/source_report.json',{})
    if drift=='receipt':p.save(tmp_path/'S1/sourcecheck_gpu.json',{})
    if drift is None:assert verify_source_gate(tmp_path/'gate.json')['status']=='passed'
    else:
        with pytest.raises(ValueError):verify_source_gate(tmp_path/'gate.json')
