import torch
import pytest
from tools.uncertainty_distribution import DistributionStage,objective,soft_bin_loss,support_from_source,history_decision


def test_zero_distribution_is_exact_identity_and_always_legal():
    h=DistributionStage(2,.5);x=torch.randn(4,20);b=torch.tensor([[1.,3.],[2.,10.],[.1,.2],[3.,4.]])
    p,_,v=h(x,b);assert torch.equal(p,b) and (v>0).all()
    with torch.no_grad():h.out.bias[:17].fill_(-100);h.out.bias[16]=100;h.out.bias[17:]= -100;h.out.bias[17]=100
    p,_,_=h(x,b);assert (p[:,1]>p[:,0]).all()


def test_soft_bin_labels_at_endpoints_and_interpolation():
    logits=torch.zeros(2,2,17,requires_grad=True);target=torch.tensor([[-.5,.5],[0.,.03125]])
    loss=soft_bin_loss(logits,target,.5);assert torch.allclose(loss,torch.ones(2)*torch.tensor(17.).log())
    loss.mean().backward();assert logits.grad.isfinite().all() and logits.grad.abs().sum()>0


def test_source_supervised_distribution_learns_and_restores():
    torch.manual_seed(123);h=DistributionStage(2,.5);x=torch.randn(8,20)
    b=torch.tensor([[2.,8.]]).repeat(8,1);g=torch.tensor([[3.,9.]]).repeat(8,1);opt=torch.optim.AdamW(h.parameters(),lr=.01)
    first=float(objective(h,x,b,g)[0].mean())
    for _ in range(35):
        loss=objective(h,x,b,g)[0].mean();opt.zero_grad();loss.backward();opt.step()
    assert float(loss)<first*.5
    copy=DistributionStage(2,.5);copy.load_state_dict(h.state_dict(),strict=True);assert torch.equal(h(x,b)[0],copy(x,b)[0])


def test_support_uses_source_only_and_narrows():
    b=torch.tensor([[2.,8.],[4.,10.]]);g=b+2
    r=support_from_source(b,g);assert .25<=r<=2
    assert support_from_source(b,g,maximum=r/2)<=r/2


def test_epoch_zero_history_first_stop():
    h=[dict(epoch=i,source_validation=1+i*.1) for i in range(16)]
    assert history_decision(h)==(0,15)
    with pytest.raises(ValueError):history_decision(h[:-1])


def test_second_stage_context_changes_with_updated_box():
    from tools.uncertainty_refiner import context
    f=[torch.arange(40.).view(1,2,20)];m=[torch.ones(1,1,20,dtype=torch.bool)]
    first,valid=context(f,m,torch.tensor([[5.,10.]]),torch.tensor([0]),[1])
    updated,valid2=context(f,m,torch.tensor([[7.,12.]]),torch.tensor([0]),[1])
    assert valid.all() and valid2.all() and not torch.equal(first,updated)
