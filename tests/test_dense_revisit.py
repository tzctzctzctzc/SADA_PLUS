import torch
from tools.dense_revisit_math import occupancy,sample_sequence,dense_loss,DenseEvidence,utility_loss,temporal_ssl


def test_union_short_and_overlap():
    y=occupancy(4,torch.tensor([[.1,.3],[.2,.4],[1.,2.],[1.5,2.5]]),torch.tensor([10,10,0,1]))
    assert abs(float(y[0,10])-.3)<1e-6
    assert y[1,0]>0 and y[2,0]>0 and y[2,1]>0
    assert y[:,10].sum()>0 and y.max()<=1


def test_interpolation_and_outside():
    x=torch.arange(5.).view(-1,1);b=torch.tensor([[0.,4.],[-5.,-3.]])
    z,m=sample_sequence(x,b,4)
    assert torch.equal(z[0,:,0],torch.tensor([.5,1.5,2.5,3.5]))
    assert m[0].all() and not m[1].any() and not z[1].any()


def test_all_objectives_real_gradients():
    torch.manual_seed(1);model=DenseEvidence();x=torch.randn(16,2304);logits,z=model(x)
    y=occupancy(16,torch.tensor([[2.,8.]]),torch.tensor([0]))
    bg,s,u=model.candidate(logits,z,torch.tensor([[2.,7.],[9.,13.]]),torch.tensor([0,2]),torch.tensor([.8,.7]))
    loss=dense_loss(logits,y)+utility_loss(u,torch.tensor([0.,1.]),torch.tensor([-.001,.001]))+temporal_ssl(z,z+.01)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.project.weight.grad.abs().sum()>0 and model.blocks[0][0].weight.grad.abs().sum()>0
    assert (bg>=0).all() and (bg<=1).all() and s.shape==(2,)


def test_dense_balancing_and_ssl():
    z=torch.zeros(3,2,requires_grad=True);y=torch.tensor([[0.,0.],[.1,0.],[1.,0.]])
    loss=dense_loss(z,y);assert torch.allclose(loss,torch.log(torch.tensor(2.)))
    a=torch.eye(4);assert temporal_ssl(a,a)<temporal_ssl(a,a.roll(1,0))
