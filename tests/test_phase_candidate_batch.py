from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest
from tools.phase_boundary import PhaseBoundary
from tools.phase_candidate_components import endpoint_error_many, source_candidates_loss


@pytest.fixture
def double_default():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.double)
    yield
    torch.set_default_dtype(old)


def test_ragged_batch_preserves_values_and_feature_parameter_derivatives(double_default):
    torch.manual_seed(7)
    a=PhaseBoundary(torch.randn(10,4,512)).double();b=deepcopy(a)
    with torch.no_grad():
        a.error_model[-1].weight.normal_(0,.02);b.load_state_dict(a.state_dict())
    x=torch.randn(512,40,dtype=torch.double,requires_grad=True);y=x.detach().clone().requires_grad_()
    requests=[(torch.tensor([5.,18.],dtype=torch.double),torch.linspace(2,9,13,dtype=torch.double),0,0),
              (torch.tensor([2.,30.],dtype=torch.double),torch.linspace(28,33,7,dtype=torch.double),3,1)]
    expected=[a.endpoint_error(x,*r,2,80) for r in requests]
    actual=endpoint_error_many(b,y,requests,2,80)
    assert all(torch.allclose(u,v,atol=1e-10,rtol=1e-9) for u,v in zip(expected,actual))
    sum(v.square().mean() for v in expected).backward();sum(v.square().mean() for v in actual).backward()
    assert torch.allclose(x.grad,y.grad,atol=1e-10,rtol=1e-8)
    for p,q in zip(a.parameters(),b.parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:assert torch.allclose(p.grad,q.grad,atol=1e-9,rtol=1e-7)


def test_source_batch_preserves_candidate_counts_loss_and_update():
    torch.manual_seed(8);a=PhaseBoundary(torch.randn(10,4,512));b=deepcopy(a)
    a.batch_candidate_errors=False;b.batch_candidate_errors=True
    f=[torch.randn(1,512,40)];m=[torch.ones(1,1,40,dtype=torch.bool)]
    r=[torch.full((1,2,40),4.)];c=[torch.full((1,97,40),-20.)];c[0][0,0,10]=5.
    v=[dict(feats=torch.zeros(512,40),segments=torch.tensor([[6.,16.],[24.,32.]]),labels=torch.tensor([0,1]))]
    oa=torch.optim.AdamW(a.parameters(),lr=.0003);ob=torch.optim.AdamW(b.parameters(),lr=.0003)
    for _ in range(2):
        oa.zero_grad();ob.zero_grad()
        la,sa=source_candidates_loss(SimpleNamespace(fpn_strides=[1]),a,f,m,r,v,c)
        lb,sb=source_candidates_loss(SimpleNamespace(fpn_strides=[1]),b,f,m,r,v,c)
        assert sa==sb and torch.allclose(la,lb,atol=1e-6)
        la.backward();lb.backward();oa.step();ob.step()
    assert max((p-q).abs().max() for p,q in zip(a.parameters(),b.parameters()))<2e-5


def test_prefix_partition_matches_original_recurrence_and_derivatives(double_default):
    from tools.phase_boundary import path_partition
    torch.manual_seed(11);x=torch.randn(3,10,24,3,requires_grad=True);y=x.detach().clone().requires_grad_()
    a=x[...,0,0];b=torch.full_like(a,-1e4);c=b.clone()
    for t in range(1,x.shape[-2]):a,b,c=a+x[...,t,0],torch.logaddexp(a,b)+x[...,t,1],torch.logaddexp(b,c)+x[...,t,2]
    actual=path_partition(y)
    assert torch.allclose(c,actual,atol=1e-11,rtol=1e-10)
    c.sum().backward();actual.sum().backward()
    assert torch.allclose(x.grad,y.grad,atol=1e-10,rtol=1e-9)
