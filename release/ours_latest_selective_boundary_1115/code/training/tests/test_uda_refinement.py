"""检查真实机制、梯度隔离与状态恢复，不把小测试当检测收益。"""
import importlib.util,sys,types
from pathlib import Path
import torch
from types import SimpleNamespace

pkg=types.ModuleType('refinement_test_pkg');pkg.__path__=[str(Path(__file__).parents[1]/'libs/models/SADA')]
sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.uda_refinement',Path(pkg.__path__[0])/'uda_refinement.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_rotation_bounds_coverage_and_rng():
    state=torch.get_rng_state().clone()
    starts=[s for step in range(32) for s in m.rotating_starts(2304,32,2,step)]
    assert min(starts)>=0 and max(starts)<=2272
    assert any(800<s<1400 for s in starts)
    assert torch.equal(state,torch.get_rng_state())

def test_semantic_transport_rejects_unknown_and_detaches_teacher():
    x=torch.randn(4,6,requires_grad=True);y=torch.randn(5,6,requires_grad=True)
    s=torch.zeros(4,3);s[:,0]=1
    t=torch.zeros(5,3);t[:,0]=.8;t.requires_grad_()
    loss,mass=m.semantic_transport(x,y,s,t,m.TF_V3_SPEC);loss.backward()
    assert 0<mass<=.8+1e-5 and x.grad.norm()>0 and y.grad.norm()>0 and t.grad is None
    zero,mass=m.semantic_transport(x,y,s,torch.zeros_like(t),m.TF_V3_SPEC)
    assert zero==0 and mass==0
    wrong=torch.zeros_like(t);wrong[:,1]=1
    wrong_loss,mass=m.semantic_transport(x,y,s,wrong,m.TF_V3_SPEC)
    assert wrong_loss==0 and mass==0

def test_tf_real_forward_and_restore_counter(tmp_path):
    branch=m.SemanticTimeFrequencyAlignment(8,12,3,dict(width=4,length=16,modes=4))
    model=SimpleNamespace(rq3_capacity=SimpleNamespace(strides=[1,2]))
    videos=[dict(feats=torch.randn(12,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))]
    sf=[torch.randn(1,8,64//s,requires_grad=True) for s in (1,2)]
    tf=[torch.randn_like(x,requires_grad=True) for x in sf]
    semantics=[torch.ones(1,x.shape[-1],3)*.8 for x in sf]
    loss,stats=branch(model,videos,videos,sf,tf,5,semantics,semantics);loss.backward()
    assert all(x.grad is not None and x.grad.isfinite().all() and x.grad.norm()>0 for x in tf)
    assert branch.calls==1 and stats['transport_mass']>0
    p=tmp_path/'branch.pt';torch.save(branch.state_dict(),p)
    restored=m.SemanticTimeFrequencyAlignment(8,12,3,dict(width=4,length=16,modes=4))
    restored.load_state_dict(torch.load(p))
    assert restored.calls==branch.calls
    a=branch.semantic_windows(model,videos,sf,semantics)[0]
    b=restored.semantic_windows(model,videos,sf,semantics)[0]
    assert torch.equal(a,b)

def test_teacher_selection_requires_separate_validation():
    assert m.choose_teacher(torch.tensor([1.,.2]),torch.tensor([.8,.4]),0,.01)==1
    assert m.choose_teacher(torch.tensor([1.,.2]),torch.tensor([.4,.8]),0,.01) is None
    assert m.choose_teacher(torch.tensor([.1,.2]),torch.tensor([.1,.2]),0,.01) is None

def test_mda_distills_only_student_and_never_mixes_parameters():
    ds=torch.nn.ModuleList([torch.nn.Linear(1,1),torch.nn.Linear(1,1)])
    with torch.no_grad():
        ds[0].weight.fill_(-1);ds[1].weight.fill_(2)
        for d in ds:d.bias.zero_()
    branch=m.SelectiveDomainDistillation(2,dict(distillation_interval=1))
    x0=torch.full((18,1),-1.,requires_grad=True);x1=torch.ones(18,1,requires_grad=True)
    for l in range(2):
        branch.observe(l,x0,0,1.);branch.observe(l,x1,1,1.)
    before=[p.detach().clone() for p in ds.parameters()]
    loss=branch.distillation_loss(ds,5);loss.backward()
    assert loss>0 and branch.last_selected==1 and ds[0].weight.grad.abs().sum()>0
    assert ds[1].weight.grad is None and x0.grad is None and x1.grad is None
    branch.after_step(ds,5)
    assert all(torch.equal(a,b) for a,b in zip(before,ds.parameters()))
    assert branch.calls==1 and not branch.cache and branch.aggregations==0

def test_no_valid_teacher_no_intervention():
    ds=torch.nn.ModuleList([torch.nn.Linear(1,1),torch.nn.Linear(1,1)])
    ds[1].load_state_dict(ds[0].state_dict())
    branch=m.SelectiveDomainDistillation(2,dict(distillation_interval=1))
    for l in range(2):
        branch.observe(l,-torch.ones(18,1),0,1.);branch.observe(l,torch.ones(18,1),1,1.)
    loss=branch.distillation_loss(ds,5)
    assert loss==0 and branch.last_selected==0
