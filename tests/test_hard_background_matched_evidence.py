import numpy as np
from tools.hard_background_matched_evidence import mask_at_budget


def test_count_matching_low_score_protection_and_ranking_only():
    labels=np.array([0]*10+[1]*10);scores=np.array([.2]+[.6]*19);risk=np.arange(20,dtype=float)
    a=mask_at_budget(labels,scores,risk,.2);b=mask_at_budget(labels,scores,-risk,.2)
    assert (~a).sum()==(~b).sum()==3 and a[0] and b[0]
    assert not np.array_equal(a,b)
