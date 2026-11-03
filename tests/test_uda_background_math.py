import numpy as np
import torch
from tools.uda_background_math import UDAHead,semantic_targets,conditional_coral,weighted_supervision,ema_update,consistency_loss
from tools.highscore_pair_math import make_model


def test_semantic_non_task_never_overrides_any_task_foreground():
    boxes=torch.tensor([[0.,10.],[20.,30.],[50.,55.]])
    gt=torch.tensor([[0.,1.],[0.,10.],[20.,30.]]);labels=torch.tensor([2,34,11])
    assert semantic_targets(boxes,gt,labels).tolist()==[2,11,-1]


def test_conditional_coral_matches_numpy_and_has_both_domain_gradients():
    a=torch.randn(8,4,requires_grad=True);b=torch.randn(8,4,requires_grad=True);classes=torch.zeros(8,dtype=torch.long)
    loss=conditional_coral(a,b,classes,classes);ref=((np.cov(a.detach().numpy().T)-np.cov(b.detach().numpy().T))**2).sum()/64
    assert abs(float(loss)-ref)<1e-6;loss.backward();assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0


def test_new_core_preserves_original_forward_and_weighted_ones_match():
    torch.manual_seed(311104);old=make_model();torch.manual_seed(311104);new=UDAHead();x=torch.randn(3,24,512);v=torch.ones(3,24,dtype=torch.bool);m=torch.randn(3,12)
    assert torch.equal(old(x,v,m),new.all_outputs(x,v,m)[0])
    z=torch.randn(10,2);bg=(torch.arange(10)%2).float();q=torch.rand(10)
    assert torch.allclose(weighted_supervision(z,bg,q),weighted_supervision(z,bg,q,torch.ones(10)),atol=1e-7)


def test_teacher_targets_detached_and_ema_first_step_copies_student():
    a=torch.randn(3,2,requires_grad=True);b=torch.randn(3,2,requires_grad=True);consistency_loss(a,b).backward();assert a.grad is not None and b.grad is None
    m=torch.nn.Linear(3,2);t=torch.nn.Linear(3,2);ema_update(m,t,0)
    assert all(torch.equal(a,b) for a,b in zip(m.parameters(),t.parameters()))


def test_target_stream_restore_preserves_wrap_order_and_visit_counts():
    from copy import deepcopy
    from tools.run_uda_background import TargetStream
    a=TargetStream(7);a.draw(11);saved=deepcopy(a.state());x=a.draw(19);seen=a.seen.clone()
    b=TargetStream(7);b.restore(saved);assert torch.equal(x,b.draw(19)) and torch.equal(seen,b.seen)
    assert int(b.seen.sum())==30 and (b.seen>0).all()
