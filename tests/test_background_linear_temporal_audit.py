import numpy as np
import torch
from tools.audit_background_linear_temporal import numpy_descriptors
from tools.background_linear_temporal import descriptors


def test_independent_descriptor_full_missing_and_fractional():
    torch.manual_seed(91)
    feature=torch.randn(9,37)*3;valid=torch.rand(37)>.15
    boxes=torch.tensor([[-5.,-1.],[-.5,3.2],[3.1,27.8],[35.2,40.],[9.,9.05]])
    labels=torch.arange(5);scores=torch.tensor([.3,.4,.6,.9,1.])
    actual=descriptors(feature,valid,boxes,labels,scores)
    expected=numpy_descriptors(feature.numpy(),valid.numpy(),boxes.numpy(),labels.numpy(),scores.numpy())
    for mode in actual:assert np.allclose(actual[mode].numpy(),expected[mode],atol=2e-5,rtol=1e-6)


def test_independent_empty():
    f=np.ones((3,12),dtype=np.float32);v=np.ones(12,dtype=bool);b=np.zeros((0,2),dtype=np.float32)
    d=numpy_descriptors(f,v,b,np.zeros(0,dtype=np.int64),np.zeros(0,dtype=np.float32))
    assert all(len(x)==0 for x in d.values())
