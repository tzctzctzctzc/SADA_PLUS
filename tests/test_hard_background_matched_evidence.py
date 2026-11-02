import numpy as np
from tools.hard_background_matched_evidence import mask_at_budget


def test_count_matching_low_score_protection_and_ranking_only():
    labels=np.array([0]*10+[1]*10);scores=np.array([.2]+[.6]*19);risk=np.arange(20,dtype=float)
    a=mask_at_budget(labels,scores,risk,.2);b=mask_at_budget(labels,scores,-risk,.2)
    assert (~a).sum()==(~b).sum()==3 and a[0] and b[0]
    assert not np.array_equal(a,b)


def test_confidence_control_removes_lowest_eligible_score_with_stable_ties():
    labels=np.zeros(6,dtype=int);scores=np.array([.1,.3,.4,.4,.8,.9])
    keep=mask_at_budget(labels,scores,-scores,.4)
    assert np.flatnonzero(~keep).tolist()==[1,2]
