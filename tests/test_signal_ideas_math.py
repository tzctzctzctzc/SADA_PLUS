import torch
from tools.signal_ideas_math import make_model,select_sequence,MODES
from tools.highscore_pair_math import make_model as original


def test_reference_loss_grad_and_update_are_exact():
    torch.manual_seed(311104);old=original();torch.manual_seed(311104);new=make_model('reference')
    x=torch.randn(3,24,512);v=torch.ones(3,24,dtype=torch.bool);meta=torch.randn(3,12)
    a=old(x,v,meta);b=new(x,v,meta);assert torch.equal(a,b)
    a.square().mean().backward();b.square().mean().backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(old.parameters(),new.parameters()))
    oa=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=.05);ob=torch.optim.AdamW(new.parameters(),lr=3e-4,weight_decay=.05);oa.step();ob.step()
    assert all(torch.equal(p,q) for p,q in zip(old.parameters(),new.parameters()))


def test_dual_preserves_parameter_count_and_uses_both_streams():
    model=make_model('dual');assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in original().parameters())
    x=torch.randn(3,24,1024,requires_grad=True);model(x,torch.ones(3,24,dtype=torch.bool),torch.zeros(3,12)).square().mean().backward()
    assert x.grad[...,:512].abs().sum()>0 and x.grad[...,512:].abs().sum()>0


def test_centering_keeps_samples_and_metadata_outside_selection():
    a=torch.randn(2,24,512);b=torch.randn(2,24,512);ga=torch.randn(512);gb=torch.randn(512)
    r=dict(seq_fpn=b,seq_early=a,global_fpn=gb,global_early=ga)
    assert torch.equal(select_sequence(r,'centered'),b-gb)
    assert torch.equal(select_sequence(r,'centerdual'),torch.cat((a-ga,b-gb),-1))
