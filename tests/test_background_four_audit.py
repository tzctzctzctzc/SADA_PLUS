import numpy as np
import torch
from tools.audit_background_four_routes import numpy_samples
from tools.background_linear_temporal import region_samples


def test_independent_sampler_all_points_and_missing_neighbors():
    feature=torch.arange(64,dtype=torch.float32).reshape(4,16)/3
    valid=torch.ones(16,dtype=torch.bool);valid[5]=False
    boxes=torch.tensor([[-2.,2.],[3.,7.],[14.,18.],[4.,4.5]])
    a,m=region_samples(feature,valid,boxes)
    x,v,p=numpy_samples(feature.numpy(),valid.numpy(),boxes.numpy())
    assert np.allclose(x,torch.cat(a,1),atol=1e-6,rtol=1e-6)
    assert np.array_equal(v,torch.cat(m,1)) and p.shape==(4,24)
    assert not v[0].all() and not v[2].all()
