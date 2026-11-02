from copy import deepcopy
import torch
from tools.hard_background_modules import (targets,balanced_bg,rank_loss,context_swap,HardBackgroundHead,step_loss,MODES)


def item():
    torch.manual_seed(12)
    boxes=torch.tensor([[1.,4.],[7.,9.],[9.,11.]])
    truth=torch.tensor([[1.,4.]])
    labels=torch.tensor([0,0,0]);bg,q=targets(boxes,labels,truth,torch.tensor([0]))
    return dict(feature=torch.randn(512,16),valid=torch.ones(16,dtype=torch.bool),boxes=boxes,
        labels=labels,scores=torch.tensor([.8,.6,.4]),bg=bg,quality=q,truth=truth,truth_labels=torch.tensor([0]))


def test_wrong_class_action_is_not_strict_background():
    bg,q=targets(torch.tensor([[1.,3.],[6.,7.]]),torch.tensor([2,2]),torch.tensor([[1.,3.]]),torch.tensor([1]))
    assert bg.tolist()==[0.,1.] and q.tolist()==[0.,0.]


def test_rank_gradient_direction_and_single_class_pairing():
    logits=torch.tensor([0.,0.],requires_grad=True)
    loss=rank_loss(logits,torch.tensor([1.,0.]),torch.tensor([2,2]),torch.tensor([.8,.6]));loss.backward()
    assert logits.grad[0]<0 and logits.grad[1]>0
    assert rank_loss(logits,torch.tensor([1.,0.]),torch.tensor([2,3]),torch.tensor([.8,.6]))==0


def test_context_swap_preserves_interior_and_input():
    x=item();h=HardBackgroundHead();reps=h.encode(x['feature'],x['valid'],x['boxes']);original=reps.detach().clone()
    other,used=context_swap(reps,x['bg'],x['labels'])
    assert torch.equal(other[:,64:128],reps[:,64:128]);assert used.all()
    assert torch.equal(original,reps.detach()) and not torch.equal(other,reps)


def test_all_four_modes_finite_gradients_no_input_mutation():
    torch.set_num_threads(1)
    for mode in MODES:
        x=item();before=x['feature'].clone();h=HardBackgroundHead()
        loss,stats=step_loss(h,x,mode,torch.zeros(3,2));loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None)
        assert h.projection.weight.grad.norm()>0 and torch.equal(before,x['feature'])
        assert stats['changed_labels']==0


def test_jitter_label_changes_are_recomputed_not_assumed_invariant():
    boxes=torch.tensor([[.9,1.1]])
    a,_=targets(boxes,torch.tensor([0]),torch.tensor([[1.,2.]]),torch.tensor([0]))
    b,_=targets(boxes-.2,torch.tensor([0]),torch.tensor([[1.,2.]]),torch.tensor([0]))
    assert a.item()==0 and b.item()==1


def test_complete_optimizer_and_rng_roundtrip():
    import io
    x=item();h=HardBackgroundHead();opt=torch.optim.AdamW(h.parameters(),lr=3e-4)
    l,_=step_loss(h,x,'rank');l.backward();opt.step()
    state=dict(head=h.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state())
    stream=io.BytesIO();torch.save(state,stream);stream.seek(0);r=torch.load(stream)
    other=HardBackgroundHead();other.load_state_dict(r['head']);second=torch.optim.AdamW(other.parameters());second.load_state_dict(r['optimizer'])
    assert all(torch.equal(h.state_dict()[k],other.state_dict()[k]) for k in h.state_dict())
    assert torch.equal(h(**{k:x[k] for k in ['feature','valid','boxes','labels','scores']}),other(**{k:x[k] for k in ['feature','valid','boxes','labels','scores']}))
    assert len(second.state_dict()['state'])==len(opt.state_dict()['state'])


def test_prediction_only_scores_high_pool_and_preserves_all_original_arrays():
    from tools.run_hard_background_pilots import predict
    x=item();x.update(video='v',segments=x['boxes'].clone(),scores=torch.tensor([.2,.3,.9]))
    h=HardBackgroundHead();out=predict(h,x)
    assert out['pbg'][0]==0 and out['pbg'][1]>0
    assert all(torch.equal(out[k],x[k]) for k in ['segments','scores','labels'])
