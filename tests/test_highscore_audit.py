import numpy as np
import torch
from tools.audit_highscore_pairs import np_samples
from tools.background_linear_temporal import region_samples


def test_all_sampling_points_and_invalid_neighbors():
    torch.manual_seed(1);f=torch.randn(5,32);v=torch.ones(32,dtype=torch.bool);v[7]=False;b=torch.tensor([[-3.,4.],[5.,10.],[28.,35.]])
    a,m=region_samples(f,v,b);x,valid=np_samples(f.numpy(),v.numpy(),b.numpy())
    assert np.allclose(x,torch.cat(a,1),atol=2e-6,rtol=1e-6) and np.array_equal(valid,torch.cat(m,1))
