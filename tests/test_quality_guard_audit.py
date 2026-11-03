def test_independent_product_float32_matches_fixed_GPU_readout_math():
    import numpy as np
    import torch
    from tools.audit_quality_guard import independent_risk
    from tools.run_quality_guard import protected_risk
    p=np.array([0,.9,.1234567,1],np.float32);q=np.array([.8,.7,.9876543,1],np.float32)
    assert np.array_equal(independent_risk(p,q),protected_risk(torch.from_numpy(p),torch.from_numpy(q)).numpy())
