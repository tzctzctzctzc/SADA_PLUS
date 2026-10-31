import pytest
import torch
from tools.check_boundary_support import endpoint_counts,source_decision


def test_support_counts_do_not_hide_zero_base_or_count_zero_GT_as_bad_endpoint():
    base=torch.tensor([[[1.,0.,1.],[1.,1.,1.]]]);truth=torch.tensor([[[1.,2.,0.],[.1,5.,1.]]])
    result=endpoint_counts(base,truth,torch.ones(1,3,dtype=torch.bool),torch.tensor([.5,1.,2.]))
    assert result==dict(endpoints=5,zero_base=1,inside=2,below=1,above=2)


def test_six_direction_source_support_gate_rejects_missing_correspondence_or_bad_support():
    rows=[dict(shift='S'+str(i),metrics=dict(endpoints=100,inside=90,zero_base=0,temporal_points=5),gpu_head_probe=dict(passed=True)) for i in range(1,7)]
    assert source_decision(rows)['status']=='passed'
    rows[0]['metrics']['temporal_points']=0
    assert source_decision(rows)['status']=='rejected'
    with pytest.raises(ValueError):source_decision(rows[:-1])
