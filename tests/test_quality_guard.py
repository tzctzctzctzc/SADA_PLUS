import torch


def test_quality_protection_is_fixed_monotone_and_readonly():
    from tools.run_quality_guard import protected_risk,MODES,BASEMODES,SPEC
    p=torch.tensor([.8,.8,0.,1.]);q=torch.tensor([.9,.1,.7,1.]);before=p.clone()
    r=protected_risk(p,q)
    assert r[0]<r[1] and torch.equal(p,before) and r[2]==0 and r[3]==0
    assert len(MODES)==8 and len(BASEMODES)==4 and SPEC['optimizer_steps']==0


def test_float32_independent_product_exact():
    import numpy as np
    from tools.run_quality_guard import protected_risk
    p=torch.tensor([.1234567,.99991,.0001]);q=torch.tensor([.92345,.9991,.1])
    assert np.array_equal(protected_risk(p,q).numpy(),p.numpy()*(np.float32(1)-q.numpy()))
