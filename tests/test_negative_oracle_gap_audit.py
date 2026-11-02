import numpy as np
import torch
from torch.nn import functional as F
from tools.audit_negative_oracle_gap import focal,auc


def test_focal_logit_gradient_matches_autograd():
    x=torch.tensor([[-9.,-2.,0.,2.,9.]],dtype=torch.double,requires_grad=True)
    y=torch.tensor([[1.,0.,1.,0.,1.]],dtype=torch.double)
    p=x.sigmoid();pt=p*y+(1-p)*(1-y)
    a=.25*y+.75*(1-y)
    loss=F.binary_cross_entropy_with_logits(x,y,reduction='none')*(1-pt)**2*a
    loss.sum().backward()
    value,g,_=focal(p.detach().numpy(),y.numpy().astype(bool))
    np.testing.assert_allclose(value,loss.detach().numpy(),rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(g,x.grad.numpy(),rtol=1e-10,atol=1e-12)


def test_auc_ties_and_order():
    assert auc(np.array([1.,1.,1.,1.]),np.array([True,False,True,False]))==.5
    assert auc(np.array([1.,2.,3.,4.]),np.array([False,False,True,True]))==1.
    assert auc(np.array([1.,2.,3.,4.]),np.array([True,True,False,False]))==0.
