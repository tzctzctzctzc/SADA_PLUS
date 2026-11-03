import torch
from tools.native_joint_math import critical_anchor,proposal_loss


def test_anchor_protects_only_harmful_foreground():
    seq=torch.randn(2,3,24,4,requires_grad=True);teacher=torch.zeros_like(seq);v=torch.ones(2,3,24,dtype=torch.bool)
    loss=critical_anchor(seq,teacher,v,torch.tensor([1.,0.,0.]),torch.tensor([.1,-.2,.1]));loss.backward()
    assert seq.grad[:,0].abs().sum()==0 and seq.grad[:,2].abs().sum()==0
    assert seq.grad[:,1,4:20].abs().sum()>0 and seq.grad[:,:,:4].abs().sum()==0
    assert critical_anchor(teacher,teacher,v,torch.tensor([1.,0.,0.]),torch.tensor([.1,-.2,.1]))==0


def test_zero_anchor_and_loss():
    z=torch.zeros(3,4,24,5,requires_grad=True);v=torch.ones(3,4,24,dtype=torch.bool)
    assert critical_anchor(z,z.detach(),v,torch.ones(4),torch.ones(4))==0
    logits=torch.zeros(3,4,2,requires_grad=True);loss=proposal_loss(logits,torch.tensor([0.,0.,1.,1.]),torch.tensor([.8,.4,0.,0.]));loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum()>0


def test_full_entrypoint_import():
    from tools.run_native_joint import SPEC
    assert SPEC['epochs']==20 and SPEC['warmup_epochs']==1 and SPEC['target_train_GT_read'] is False


def test_cost_zero_falls_back_to_plain():
    from tools.native_joint_math import cost_proposal_loss
    z=torch.randn(3,4,2);b=torch.tensor([0.,0.,1.,1.]);q=torch.tensor([.8,.4,0.,0.]);d=torch.zeros(4)
    assert torch.equal(proposal_loss(z,b,q),cost_proposal_loss(z,b,q,d))
