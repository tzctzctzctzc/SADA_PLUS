import numpy as np
from tools.audit_dense_revisit import union_targets


def test_independent_union():
    y=union_targets(3,[[.1,.3],[.2,.4],[1.,2.],[1.2,2.5]],[10,10,0,1])
    assert abs(y[0,10]-.3)<1e-12 and abs(y[:,0].sum()-1)<1e-12
    assert y[1,0]>0 and y[1,1]>0 and y.max()<=1


def test_empty_union():
    assert np.array_equal(union_targets(3,[],[]),np.zeros((3,97)))
