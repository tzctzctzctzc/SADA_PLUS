import numpy as np
from tools.hard_background_val_search import transform,grid


def test_low_score_scope_and_no_filter_fallback():
    s=np.array([.2,.4,.8]);p=np.array([.99,.8,.95]);q=np.array([.01,.2,.9])
    k,v=transform(s,p,q,dict(kind='delete',cutoff=.3,beta=0.,threshold=.9))
    np.testing.assert_array_equal(k,[True,True,False]);np.testing.assert_array_equal(v,s)
    assert any(x['kind']=='delete' and x['threshold']==1.01 for x in grid())


def test_zero_soft_strength_is_exact_baseline_and_quality_fusion_finite():
    s=np.array([.2,.4,.8]);p=np.array([0.,1.,.4]);q=np.array([0.,0.,1.])
    k,v=transform(s,p,q,dict(kind='soft',cutoff=.3,alpha=0.,beta=0.))
    np.testing.assert_array_equal(v,s);assert k.all()
    _,v=transform(s,p,q,dict(kind='soft',cutoff=.3,alpha=.5,beta=.5))
    assert np.isfinite(v).all() and v[0]==s[0]
