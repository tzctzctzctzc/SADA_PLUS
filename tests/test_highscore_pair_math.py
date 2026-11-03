import torch
from tools.highscore_pair_math import build_batch,objective,make_model

torch.set_num_threads(2)


def data():
    return dict(seq=torch.randn(8,24,512),valid=torch.ones(8,24,dtype=torch.bool),metadata=torch.randn(8,12),bg=torch.tensor([1.,0.]*4),quality=torch.rand(8))


def test_shared_conditions_are_identical_and_original_primary_unchanged():
    d=data();b=build_batch(d,torch.tensor([0,1]),torch.tensor([[2,3],[4,5]]))
    assert torch.equal(b['metadata'][:2],d['metadata'][:2])
    assert torch.equal(b['metadata'][2:4],b['metadata'][4:])
    assert torch.equal(b['seq'][2:4],d['seq'][torch.tensor([2,4])])


def test_only_pair_objective_changes_and_has_visual_gradient():
    torch.manual_seed(3);a=make_model();b=make_model();b.load_state_dict(a.state_dict());d=data()
    x=build_batch(d,torch.tensor([0,1]),torch.tensor([[2,3],[4,5]]));w=torch.tensor([.001,.003])
    la,ta=objective(a,x,2,w,True);lb,tb=objective(b,x,2,w,False)
    assert torch.equal(ta['ordinary'],tb['ordinary']) and torch.allclose(la-lb,.25*ta['pair'])
    ga=torch.autograd.grad(la,a.project.weight)[0];gb=torch.autograd.grad(lb,b.project.weight)[0]
    assert torch.isfinite(ga).all() and not torch.equal(ga,gb)
