import numpy as np
import torch
from tools.semantic_memory_math import semantic_distances,ordered_descriptor


def test_exclusion_and_class_balance():
    q=torch.tensor([[1.,0.],[0.,1.]])
    b=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[-1.,0.]])
    labels=torch.tensor([0,10,0,97]);videos=torch.tensor([0,1,2,2])
    fg,dual=semantic_distances(q,b,labels,videos,0)
    d=np.maximum(0,(q.numpy()[:,None]-b.numpy()[None]).astype(float)**2).sum(2)
    assert np.allclose(fg.numpy(),d[:,2])
    assert np.allclose(dual.numpy(),d[:,2]-np.minimum(d[:,1],d[:,3]))


def test_order_is_retained():
    x=torch.stack((torch.linspace(1,3,20),torch.linspace(3,1,20)),1)
    b=torch.tensor([[1.,17.]])
    a,v=ordered_descriptor(x,b);r,_=ordered_descriptor(x.flip(0),b)
    assert v.all() and not torch.allclose(a,r)
