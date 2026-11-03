import numpy as np
from tools.audit_semantic_memory import descriptor_numpy,distances_numpy


def test_reference_order_and_metric():
    x=np.stack((np.linspace(1,3,20),np.linspace(3,1,20)),1)
    z,v=descriptor_numpy(x,np.array([[1.,17.]]));rev,_=descriptor_numpy(x[::-1],np.array([[1.,17.]]))
    assert v.all() and not np.allclose(z,rev)
    bank=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    out=distances_numpy(np.array([[1.,0.]]),bank,np.array([0,10,0]),np.array([0,1,2]),0)
    assert np.allclose(out,[[4.,2.]])


def test_large_time_grid_matches_native_precision():
    import torch
    from tools.semantic_memory_math import ordered_descriptor
    rng=np.random.default_rng(17);x=rng.normal(size=(2048,8)).astype(np.float32);boxes=np.array([[1500.1234,1501.2345],[1940.7,1946.1]],np.float32)
    a,v=descriptor_numpy(x,boxes);b,m=ordered_descriptor(torch.from_numpy(x),torch.from_numpy(boxes))
    assert np.array_equal(v,m.numpy()) and np.allclose(a,b.numpy(),atol=3e-6,rtol=0)
