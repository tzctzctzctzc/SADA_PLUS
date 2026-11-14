import importlib.util
import sys
import types
from pathlib import Path
import torch
from types import SimpleNamespace

pkg=types.ModuleType('class_path_tests');pkg.__path__=[str(Path(__file__).parents[1]/'libs/models/SADA')]
sys.modules[pkg.__name__]=pkg
spec=importlib.util.spec_from_file_location(pkg.__name__+'.class_path_alignment',Path(pkg.__path__[0])/'class_path_alignment.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def test_projection_does_not_consume_rng_and_no_trainable_escape():
    state=torch.get_rng_state().clone()
    b=m.ClassConditionalPathAlignment(8,8,2,{})
    assert torch.equal(state,torch.get_rng_state()) and not list(b.parameters())

def test_path_distinguishes_sequential_and_overlap():
    x=torch.arange(32).float()[None,None].repeat(1,3,1)
    q=torch.zeros(1,2,32);q[:,0,:16]=1;q[:,1,16:]=1
    a,wa=m.path_features(x,q,torch.eye(3))
    b,wb=m.path_features(x,torch.ones_like(q),torch.eye(3))
    assert not torch.equal(a[1],b[1]) and (wa==.5).all() and (wb==1).all()

def test_no_target_semantic_gradient_and_no_unknown_alignment():
    s=torch.randn(5,4,32,requires_grad=True);t=torch.randn(6,4,32,requires_grad=True)
    q=torch.rand(6,2,32,requires_grad=True)
    sx,sw=m.path_features(s,torch.ones(5,2,32),torch.eye(4))
    tx,tw=m.path_features(t,q,torch.eye(4))
    loss,_,_=m.conditional_discrepancy(sx,tx,sw,tw,[.5,1.,2.]);loss.backward()
    assert q.grad is None and s.grad.isfinite().all() and t.grad.norm()>0
    zero,_,n=m.conditional_discrepancy(sx,tx,sw,torch.zeros_like(tw),[1.])
    assert zero==0 and n==0

def test_same_distribution_zero_and_state_roundtrip():
    x=torch.randn(3,5,32);q=torch.rand(3,2,32)
    b=m.ClassConditionalPathAlignment(5,5,2,{})
    features,weights=m.path_features(x,q,b.projection)
    loss,_,_=m.conditional_discrepancy(features,features,weights,weights,b.spec['bandwidths'])
    assert abs(float(loss))<1e-6
    b.calls.fill_(27);c=m.ClassConditionalPathAlignment(5,5,2,{})
    c.load_state_dict(b.state_dict());assert c.calls==27 and torch.equal(b.projection,c.projection)

def test_actual_forward_source_target_gradients():
    b=m.ClassConditionalPathAlignment(5,5,2,{})
    video=dict(feats=torch.randn(5,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))
    s=[torch.randn(1,5,64,requires_grad=True)];t=[torch.randn(1,5,64,requires_grad=True)]
    q=[torch.ones(1,64,2)*.5]
    loss,metrics=b(SimpleNamespace(rq3_capacity=SimpleNamespace(strides=[1])),[video],[video],s,t,10,q,q)
    loss.backward()
    assert s[0].grad.norm()>0 and t[0].grad.norm()>0 and metrics['reconstruction']==0

def test_source_anchor_only_stops_alignment_source_gradient():
    b=m.ClassConditionalPathAlignment(5,5,2,dict(source_anchor=True))
    video=dict(feats=torch.randn(5,64),segments=torch.empty(0,2),labels=torch.empty(0,dtype=torch.long))
    s=[torch.randn(1,5,64,requires_grad=True)];t=[torch.randn(1,5,64,requires_grad=True)]
    q=[torch.ones(1,64,2)*.5]
    loss,_=b(SimpleNamespace(rq3_capacity=SimpleNamespace(strides=[1])),[video],[video],s,t,10,q,q)
    loss.backward()
    assert s[0].grad is None and t[0].grad.norm()>0
    s[0].square().mean().backward()
    assert s[0].grad.norm()>0  # 源检测监督仍能正常进入共享特征。
