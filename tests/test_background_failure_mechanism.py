import numpy as np
from tools.audit_background_failure_mechanism import configured_support


def test_background_interval_can_have_only_action_overlapping_feature_windows():
    boxes=np.array([[4.,4.2],[20.,21.]])
    gt=np.array([[4.5,5.5]])
    x=configured_support(boxes,gt,100,32,16)
    assert x['all_overlap'].tolist()==[True,False] and x['unique_feature_centers'].tolist()==[2,2]


def test_invalid_sampling_is_not_counted_as_clean_evidence():
    x=configured_support(np.array([[-4.,-3.]]),np.array([[1.,2.]]),10)
    assert x['valid_samples'][0]==0 and not x['all_overlap'][0]
