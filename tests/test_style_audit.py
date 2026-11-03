import numpy as np


def test_entrypoints_and_numpy_adversary():
    from tools.audit_style_robust import MODES
    import torch
    from tools.style_robust_math import group_update
    assert len(MODES)==4
    q=np.array([.2,.3,.5],np.float32);loss=np.array([.1,.8,.4],np.float32);ref=q*np.exp(np.float32(.01)*loss);ref/=ref.sum()
    assert np.allclose(ref,group_update(torch.from_numpy(q),torch.from_numpy(loss)).numpy(),atol=5e-7,rtol=0)
