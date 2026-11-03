import torch
from tools.target_anchor_teacher import target_loss,mask_channels


def test_unselected_and_padding_have_no_gradient_and_teacher_is_detached():
    c=torch.zeros(3,2,requires_grad=True);r=torch.ones(3,2,requires_grad=True)
    tc=torch.tensor([[4.,-3.],[0.,0.],[4.,-3.]],requires_grad=True)
    tr=torch.full((3,2),2.,requires_grad=True)
    loss,stats=target_loss([c],[r],[tc],[tr],[torch.tensor([True,True,False])]);loss.backward()
    assert stats['selected']==1 and c.grad[0].abs().sum()>0 and r.grad[0].abs().sum()>0
    assert c.grad[1:].count_nonzero()==0 and r.grad[1:].count_nonzero()==0
    assert tc.grad is None and tr.grad is None


def test_no_positive_is_zero_not_background_supervision():
    c=torch.zeros(2,3,requires_grad=True);r=torch.ones(2,2,requires_grad=True)
    loss,s=target_loss([c],[r],[c],[r],[torch.ones(2,dtype=torch.bool)])
    loss.backward();assert loss.item()==0 and s['selected']==0 and c.grad.count_nonzero()==0


def test_diou_identity_and_classification_stationary_at_teacher():
    c=torch.tensor([[4.,-3.]],requires_grad=True);r=torch.tensor([[1.,3.]],requires_grad=True)
    loss,s=target_loss([c],[r],[c.detach()],[r.detach()],[torch.tensor([True])]);loss.backward()
    assert s['reg']==0 and torch.allclose(c.grad,torch.zeros_like(c.grad),atol=1e-7)


def test_augmentation_replay_keeps_global_rng_and_time_coordinates():
    x=torch.arange(100.).reshape(10,10);g=torch.Generator().manual_seed(31)
    state=g.get_state();global_state=torch.get_rng_state();a=mask_channels(x,g)
    g.set_state(state);b=mask_channels(x,g)
    assert torch.equal(a,b) and torch.equal(global_state,torch.get_rng_state()) and a.shape==x.shape


def test_original_sada_threshold_is_used_without_hardcoding_image_confidence():
    c=torch.zeros(1,2,requires_grad=True);r=torch.ones(1,2,requires_grad=True)
    tc=torch.tensor([[1.,-2.]])
    _,image=target_loss([c],[r],[tc],[r.detach()],[torch.tensor([True])],threshold=.9)
    loss,adapted=target_loss([c],[r],[tc],[r.detach()],[torch.tensor([True])],threshold=.6)
    assert image['selected']==0 and adapted['selected']==1
    loss.backward();assert c.grad.abs().sum()>0
