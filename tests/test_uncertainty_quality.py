import pytest
import torch
from tools.uncertainty_quality import QualityHead,objective,rescore,evaluated_intervals,same_class_quality,audit_history


def test_quality_time_units_and_clipping():
    meta=dict(use_temporal_grid_feats=True,feat_stride=16,feat_num_frames=16,fps=32,duration=2.,video_length=100)
    result=evaluated_intervals(torch.tensor([[-1.,0.],[0.,10.]]),meta)
    assert torch.equal(result,torch.tensor([[0.,.25],[.25,2.]]))


def test_no_reward_for_other_class_overlap():
    boxes=torch.tensor([[1.,3.],[1.,3.],[5.,7.]])
    q=same_class_quality(boxes,torch.tensor([0,1,0]),torch.tensor([[1.,3.]]),torch.tensor([0]))
    assert q.tolist()==[1.,0.,0.]


def test_monotone_event_probabilities_source_gradient_and_restore():
    torch.manual_seed(123);h=QualityHead(2,2,2);x=torch.randn(8,20);labels=torch.arange(8)%2;levels=labels;score=torch.rand(8);width=torch.ones(8);complete=torch.ones(8,dtype=torch.bool);q=torch.linspace(0,1,8)
    loss,logits,target=objective(h,x,labels,levels,score,width,complete,q)
    assert (logits.sigmoid()[:,1:]<=logits.sigmoid()[:,:-1]).all() and (target[:,1:]<=target[:,:-1]).all()
    loss.mean().backward();assert sum(float(p.grad.abs().sum()) for p in h.parameters() if p.grad is not None)>0
    clone=QualityHead(2,2,2);clone.load_state_dict(h.state_dict(),strict=True);assert torch.equal(h(x,labels,levels,score,width,complete),clone(x,labels,levels,score,width,complete))


def test_off_identity_and_actual_within_class_rank_change():
    score=torch.tensor([.8,.6]);original=score.clone()
    assert rescore(score) is score
    out=rescore(score,torch.tensor([[.1]*5,[.9]*5]));assert out[0]<out[1] and torch.equal(score,original)


def test_quality_early_stop_is_first_legal_boundary():
    history=[dict(epoch=i,source_validation=1.+i*.1) for i in range(1,17)]
    assert audit_history(history)==(1,16)
    with pytest.raises(ValueError):audit_history(history[:-1])
