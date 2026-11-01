import numpy as np
import torch
from tools.uncertainty_quality_residual import ResidualQualityHead,objective,rescore,weighted_auc,matched_ranking_auc


def test_zero_odds_correction_is_exact_identity_and_bounded():
    torch.manual_seed(123);p=torch.rand(10000);d=torch.zeros_like(p)
    assert torch.equal(rescore(p,d),p)
    assert torch.equal(rescore(torch.tensor([0.,1.]),torch.ones(2)),torch.tensor([0.,1.]))
    x=rescore(p,torch.ones_like(p));y=rescore(p,-torch.ones_like(p));assert (x>=p).all() and (y<=p).all() and (x<=1).all() and (y>=0).all()


def test_residual_source_gradient_updates_and_strict_restore():
    torch.manual_seed(123);h=ResidualQualityHead(2,2,2);x=torch.randn(8,20);l=torch.arange(8)%2;p=torch.linspace(.1,.9,8);w=torch.ones(8);mask=torch.ones(8,dtype=torch.bool);q=torch.linspace(0,1,8)
    loss,pred,d=objective(h,x,l,l,p,w,mask,q);assert torch.equal(pred,p) and d.abs().max()==0
    opt=torch.optim.AdamW(h.parameters(),lr=.01);opt.zero_grad();loss.mean().backward();assert sum(float(z.grad.abs().sum()) for z in h.parameters() if z.grad is not None)>0;opt.step()
    d=h(x,l,l,p,w,mask);assert d.abs().max()<=1 and d.abs().sum()>0
    copy=ResidualQualityHead(2,2,2);copy.load_state_dict(h.state_dict(),strict=True);assert torch.equal(copy(x,l,l,p,w,mask),d)


def test_weighted_auc_ties_perfect_reverse_and_missing_class():
    y=np.array([False,False,True,True]);w=np.array([1.,2.,3.,4.])
    assert weighted_auc([1,1,1,1],y,w)==.5
    assert weighted_auc([0,1,2,3],y,w)==1
    assert weighted_auc([3,2,1,0],y,w)==0
    assert weighted_auc([.2,.4,.4,.9],[False,True,False,True],[1,2,3,4])==.875
    assert weighted_auc([1,2],[True,True],[1,1]) is None


def test_matched_source_ranking_ignores_other_class_background():
    q=np.array([.15,.25,.35,.45,.55,0.]);labels=np.zeros(6,dtype=int);vids=['a','a','b','b','c','c']
    assert matched_ranking_auc(q,labels,q,vids)==1
    score=q.copy();score[-1]=100
    assert matched_ranking_auc(score,labels,q,vids)==1
