def test_independent_numpy_fusion_matches_sampled_torch_heads():
    import numpy as np
    import torch
    from tools.detector_scale_math import fuse_heads
    from tools.audit_detector_scale import independent_fuse,sigmoid
    torch.manual_seed(7);c=torch.randn(10,33);r=torch.rand(2,33)*4;vc=torch.randn(10,42);vr=torch.rand(2,42)*5;v=torch.arange(42)<35
    a,b=fuse_heads(c,r,[(vc,vr,v,1.25)]);p,z=independent_fuse(c.numpy(),r.numpy(),[(vc.numpy(),vr.numpy(),v.numpy(),1.25)])
    assert np.allclose(p,sigmoid(a.numpy()),atol=2e-7,rtol=0) and np.allclose(z,b.numpy(),atol=3e-6,rtol=1e-6)
