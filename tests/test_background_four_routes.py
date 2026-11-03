import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from tools.background_four_routes_math import (official_pauc,official_sopa,scan_reference,
    mamba_inner_reference,attention_reference,hybrid_block,ASLAdapter,knn_distance,SOURCES)

torch.set_num_threads(2)


def test_vendor_identity():
    for name in ('opentad','libauc','asl'):
        record=json.loads((SOURCES/(name+'_provenance.json')).read_text())
        for p,digest in record['files'].items():
            assert hashlib.sha256((SOURCES/name/p).read_bytes()).hexdigest()==digest
    record=json.loads((SOURCES/'mamba_provenance.json').read_text())
    assert hashlib.sha256((SOURCES/'selective_scan_interface.py').read_bytes()).hexdigest()==record['sha256']


def test_pauc_official_update_and_gradient():
    y=torch.tensor([.8,.3,.6,.1],requires_grad=True);target=torch.tensor([1.,1.,0.,0.]);idx=torch.arange(4)
    loss=official_pauc(4,2,'cpu');actual=loss(y,target,idx)
    pair=(1-(y[:2,None]-y[None,2:])).clamp_min(0).square()
    expected=pair.mean()/.01
    assert torch.equal(actual,expected)
    assert torch.allclose(loss.u_pos[:2],torch.full((2,1),.099))
    ga=torch.autograd.grad(actual,y,retain_graph=True)[0];ge=torch.autograd.grad(expected,y)[0]
    assert torch.equal(ga,ge)


def test_sopa_matches_adam_first_update():
    p=torch.nn.Parameter(torch.tensor([.2,-.4]));q=torch.nn.Parameter(p.detach().clone())
    a=official_sopa([p],'cpu');b=torch.optim.Adam([q],lr=3e-4,weight_decay=.05)
    p.grad=torch.tensor([.7,-.1]);q.grad=p.grad.clone();a.step();b.step()
    assert torch.equal(p,q)


def test_pauc_auxiliary_restore():
    loss=official_pauc(4,2,'cpu');x=torch.tensor([.8,.3,.6,.1],requires_grad=True);y=torch.tensor([1.,1.,0.,0.]);i=torch.arange(4)
    loss(x,y,i);saved=loss.u_pos.clone();fresh=official_pauc(4,2,'cpu');fresh.u_pos.copy_(saved)
    assert torch.equal(loss(x,y,i),fresh(x,y,i)) and torch.equal(loss.u_pos,fresh.u_pos)


def test_scan_matches_independent_scalar_recurrence_and_gradients():
    torch.manual_seed(7);u=torch.randn(2,3,5,requires_grad=True);dt=torch.randn_like(u,requires_grad=True)
    A=-torch.arange(1,5).float().repeat(3,1);B=torch.randn(2,4,5);C=torch.randn(2,4,5);D=torch.randn(3);z=torch.randn_like(u);bias=torch.randn(3)
    out=scan_reference()(u,dt,A,B,C,D,z,bias,True)
    state=torch.zeros(2,3,4);rows=[]
    for t in range(5):
        step=F.softplus(dt[:,:,t]+bias)
        state=(step[:,:,None]*A).exp()*state+step[:,:,None]*B[:,None,:,t]*u[:,:,t,None]
        rows.append(((state*C[:,None,:,t]).sum(-1)+u[:,:,t]*D)*F.silu(z[:,:,t]))
    independent=torch.stack(rows,-1)
    assert torch.allclose(out,independent,atol=1e-6,rtol=1e-6)
    a=torch.autograd.grad(out.sum(),(u,dt),retain_graph=True);b=torch.autograd.grad(independent.sum(),(u,dt))
    assert all(torch.allclose(x,y,atol=2e-6,rtol=2e-6) for x,y in zip(a,b))


def test_causal_attention_forbids_future_values():
    torch.manual_seed(3);x=torch.randn(2,8,3,2,4)
    y=attention_reference(x);changed=x.clone();changed[:,5:]+=3
    assert torch.equal(y[:,:5],attention_reference(changed)[:,:5])
    assert not torch.equal(attention_reference(x,causal=False)[:,:5],attention_reference(changed,causal=False)[:,:5])


def test_complete_official_hybrid_gradients():
    h=hybrid_block(width=8);x=torch.randn(2,8,8);valid=torch.ones(2,8,dtype=torch.bool)
    out=h(x,valid)[0];out.square().mean().backward()
    assert out.shape==x.shape
    assert h.block.A_log.grad.abs().sum()>0 and h.block.qkv.weight.grad.abs().sum()>0
    assert h.block.dt_proj.weight.grad.abs().sum()>0 and h.block.out_proj.weight.grad.abs().sum()>0


def test_asl_full_auxiliary_and_matched_control():
    torch.manual_seed(3);a=ASLAdapter(True);b=ASLAdapter(False);b.load_state_dict(a.state_dict())
    seq=torch.randn(4,24,512);valid=torch.ones(4,24,dtype=torch.bool);ctx=torch.ones(4,8,dtype=torch.bool);labels=torch.tensor([0,0,1,2])
    la,stats=a.auxiliary(seq,valid,ctx,labels);lb,_=b.auxiliary(seq,valid,ctx,labels)
    assert torch.allclose(la-lb,stats['contrast'],atol=1e-6)
    la.backward();assert a.centers.grad.abs().sum()>0 and a.sensitivity[0].weight.grad.abs().sum()>0
    assert a.project.weight.grad.abs().sum()>0 and a.offsets.weight.grad.abs().sum()>0
    ctx.zero_();l,_=a.auxiliary(seq,valid,ctx,labels);assert torch.isfinite(l)


def test_knn_exact_kth_distance_and_partition_invariance():
    torch.manual_seed(2);q=torch.randn(7,9);b=torch.randn(53,9)
    actual=knn_distance(q,b,k=50,chunk=3)
    expected=torch.cdist(F.normalize(q,dim=1).double(),F.normalize(b,dim=1).double()).square().sort(1).values[:,49]
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert torch.equal(actual,knn_distance(q,b,k=50,chunk=7))
