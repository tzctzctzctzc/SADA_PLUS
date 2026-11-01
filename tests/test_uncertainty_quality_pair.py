import numpy as np
import torch
from tools.uncertainty_quality_pair import (
    SPEC, ResidualQualityHead, rescore, pair_pools, sample_pairs,
    corrected_logodds, pair_objective,
)


def test_pairs_only_compare_same_class_matched_threshold_eligibility():
    labels=np.array([0,0,0,0,1,1,1,1])
    quality=np.array([0.,.15,.35,.6,.05,.15,.35,.6])
    cells=pair_pools(labels,quality,['a','a','b','b','a','a','b','b'])
    assert cells
    for cell in cells:
        assert (labels[cell['positive']]==cell['label']).all()
        assert (labels[cell['negative']]==cell['label']).all()
        assert (quality[cell['positive']]>=cell['threshold']).all()
        assert (quality[cell['negative']]<cell['threshold']).all()
        assert (quality[cell['negative']]>=.1).all()
    a=sample_pairs(cells,torch.Generator().manual_seed(123),30)
    b=sample_pairs(cells,torch.Generator().manual_seed(123),30)
    assert torch.equal(a,b) and len(a)==30*len(cells)
    assert (labels[a[:,0]]==labels[a[:,1]]).all()


def test_video_weights_follow_matched_count_not_total_count():
    labels=np.zeros(5,dtype=int);q=np.array([.15,.25,.6,.6,0.])
    cells=pair_pools(labels,q,['a','a','a','b','b'])
    c=next(c for c in cells if c['threshold']==.5)
    assert torch.equal(c['positive'],torch.tensor([2,3]))
    assert torch.allclose(c['positive_weight'],torch.tensor([1/3,1.],dtype=torch.float64))


def test_logistic_sign_rewards_higher_qualification_and_ignores_shared_shift():
    good=torch.tensor([1.,-.5],requires_grad=True)
    bad=torch.tensor([0.,.5],requires_grad=True)
    loss=pair_objective(good,bad)
    assert loss[0]<loss[1]
    loss.mean().backward()
    assert (good.grad<0).all() and (bad.grad>0).all()
    assert torch.allclose(pair_objective(good+5,bad+5),loss)


def test_real_head_zero_identity_gradient_update_and_strict_restore():
    torch.manual_seed(123);h=ResidualQualityHead(2,2,2)
    x=torch.randn(16,20);labels=torch.arange(16)%2;levels=labels
    p=torch.full((16,),.3);width=torch.ones(16);complete=torch.ones(16,dtype=torch.bool)
    args=(x,labels,levels,p,width,complete)
    delta=h(*args);assert torch.equal(rescore(p,delta),p)
    before={k:v.clone() for k,v in h.state_dict().items()}
    z=corrected_logodds(p,delta).reshape(-1,2)
    loss=pair_objective(z[:,0],z[:,1]).mean();loss.backward()
    assert torch.isfinite(loss) and sum(float(v.grad.abs().sum()) for v in h.parameters() if v.grad is not None)>0
    opt=torch.optim.AdamW(h.parameters(),lr=.01);opt.step()
    assert any(not torch.equal(before[k],v) for k,v in h.state_dict().items())
    other=ResidualQualityHead(2,2,2);other.load_state_dict(h.state_dict(),strict=True)
    assert torch.equal(h(*args),other(*args)) and h(*args).abs().max()<=SPEC['logodds_bound']


def test_empty_qualification_pairs_are_rejected():
    import pytest
    with pytest.raises(ValueError):pair_pools(np.zeros(3),np.array([0.,0.,.6]),['a','b','c'])
