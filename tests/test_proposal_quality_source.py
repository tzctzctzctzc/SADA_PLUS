import torch
from tools.proposal_quality_source import same_class_quality, region_features, Readout


def test_quality_needs_matching_class_and_complete_interval():
    boxes=torch.tensor([[0.,2.],[0.,1.],[0.,2.],[4.,5.]])
    labels=torch.tensor([1,1,0,1])
    got=same_class_quality(boxes,labels,torch.tensor([[0.,2.]]),torch.tensor([1]))
    torch.testing.assert_close(got,torch.tensor([1.,.5,0.,0.]))
    torch.testing.assert_close(same_class_quality(boxes,labels,torch.empty(0,2),torch.empty(0)),torch.zeros(4))


def test_pooling_preserves_boundary_and_context_positions():
    features=torch.arange(5.).reshape(1,5)
    points=torch.stack([torch.arange(5.),torch.zeros(5),torch.ones(5)*100,torch.ones(5)],-1)
    got=region_features(features,torch.ones(5,dtype=torch.bool),points,torch.tensor([[1.,3.]]),
        torch.tensor([.8]),torch.tensor([1]),torch.tensor([2.]),3)
    torch.testing.assert_close(got[0,:5],torch.tensor([.5,1.,2.,3.,3.5]))
    torch.testing.assert_close(got[0,-3:],torch.tensor([0.,1.,0.]))


def test_padding_does_not_enter_proposal_features():
    features=torch.tensor([[0.,1.,2.,999.,999.]])
    points=torch.stack([torch.arange(5.),torch.zeros(5),torch.ones(5)*100,torch.ones(5)],-1)
    got=region_features(features,torch.tensor([1,1,1,0,0],dtype=torch.bool),points,torch.tensor([[1.,4.]]),
        torch.tensor([.8]),torch.tensor([0]),torch.tensor([3.]),2)
    assert torch.isfinite(got).all() and got[0,2]==1.5 and got[0,3]==2 and got[0,4]==0


def test_quality_readout_has_real_trainable_gradient():
    torch.manual_seed(123);model=Readout(14);x=torch.randn(8,14);y=torch.linspace(0,1,8)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.square().sum() for p in model.parameters())>0
