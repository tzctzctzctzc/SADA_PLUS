import pytest
import numpy as np
from tools.source_window_stability import (window_plan, assess_prediction, pair_statistics,
                                          retention_statistics, combine)


def test_windows_are_real_same_grid_and_untruncated():
    p=window_plan(1000,[400.,420.],3,408.)
    assert p['valid'] and len(set(p['starts']))==3
    assert all(s%32==0 and s>=0 and s+p['length']<=1000 for s in p['starts'])
    assert all(s+p['margin']<=400 and 420<=s+p['length']-p['margin'] for s in p['starts'])
    assert p['anchor']==408.


def test_short_video_and_boundary_are_not_fake_views():
    assert not window_plan(50,[20,25],0,22)['valid']
    assert not window_plan(500,[0,10],0,5)['valid']
    assert not window_plan(500,[100,450],0,200)['valid']


def test_absolute_prediction_coordinates_not_local_offsets():
    p=window_plan(1000,[400,420],0,410)
    result=assess_prediction([400,420],[[400,420]]*3,[[400,420]],[2],p)
    assert result['uncertainty']==0 and result['iou']==1 and result['boundary_error']==0


def test_quality_uses_independent_middle_view_and_fixed_instance():
    p=dict(common=[0,100])
    r=assess_prediction([10,20],[[10,20],[30,40],[10,20]],[[10,20],[30,40]],[0,0],p)
    assert r['uncertainty']==0 and r['iou']==0 and r['reference_gt_index']==0
    assert not r['offline_gt_identity_consistent']  # 稳定也可以错，不能按新预测换GT。


def test_start_end_separated_and_no_gt_candidate():
    r=assess_prediction([10,20],[[10,20],[10,20],[10,24]],[[10,20]],[0],dict(common=[0,100]))
    assert r['start_uncertainty']==0 and r['end_uncertainty']==pytest.approx(.4)
    assert not assess_prediction([10,20],[[10,20]]*3,[],[],dict(common=[0,100]))['gt_matched']


def rows():
    return [dict(video_id='v',level=0,duration=10,confidence=.6,candidate_id=str(i),
                 uncertainty=float(i),iou=1-i*.1,predicted_class=0) for i in range(6)]


def test_pair_direction_confidence_ties_and_matching():
    r=rows();s=pair_statistics(r)
    assert s['per_video']['v']['concordance']==1
    assert s['per_video']['v']['confidence_concordance']==.5
    r[-1]['level']=1;assert pair_statistics(r)['pairs']==10
    r[-2]['duration']=100;assert pair_statistics(r)['pairs']==6


def test_retention_exact_budget_and_random_expectation():
    r=rows();s=retention_statistics(r)
    assert s['selected']==3 and s['stable_iou_by_video']['v']==pytest.approx(.9)
    assert s['random_expected_iou_by_video']['v']==pytest.approx(.75)


def test_clustering_reused_physical_videos():
    s=dict(primary=dict(per_video={'v':dict(concordance=.9,confidence_concordance=.5)}))
    assert combine({'S1':s,'S2':s})['physical_videos']==1
    assert combine({'S1':s,'S2':s})['status']=='insufficient_matched_source_evidence'


def test_invalid_prediction_rejected():
    with pytest.raises(ValueError):assess_prediction([0,1],[[0,float('nan')]]*3,[],[],dict(common=[0,1]))


def test_uncertain_start_does_not_hide_stable_end():
    r=assess_prediction([10,20],[[8,20],[10,20],[12,20]],[[10,20]],[0],dict(common=[0,100]))
    assert r['start_uncertainty']==pytest.approx(.4) and r['end_uncertainty']==0


def test_prediction_leaving_common_context_is_not_reliable():
    r=assess_prediction([10,20],[[0,20],[10,20],[0,20]],[[10,20]],[0],dict(common=[5,100]))
    assert r['uncertainty']==0 and not r['prediction_fully_observed']


def test_clean_real_window_removes_gt_without_mutating_input():
    import torch
    from tools.validate_source_windows import clean_window
    video=dict(feats=torch.arange(40.).reshape(2,20),segments=torch.tensor([[4.,9.]]),
               labels=torch.tensor([3]),metadata=dict(video_id='v'))
    w=clean_window(video,4,8)
    assert w['segments'].numel()==0 and w['labels'].numel()==0
    assert video['segments'].tolist()==[[4.,9.]] and video['labels'].tolist()==[3]
    assert torch.equal(w['feats'],video['feats'][:,4:12])
    w['feats'].zero_();assert video['feats'].sum()>0


def test_dense_anchor_same_absolute_position_stride_multiplied_once():
    import torch
    from tools.validate_source_windows import anchor_prediction
    grid=torch.tensor([[0.,0.,10.,4.],[4.,0.,10.,4.],[8.,0.,10.,4.]])
    observation=dict(points=[grid],offsets=[torch.tensor([[[1.,2.],[2.,3.],[1.,2.]]])],
                     masks=[torch.ones(1,3,dtype=torch.bool)])
    pred=anchor_prediction(observation,[torch.zeros(1,2,3)],0,32,36)
    assert pred['endpoints']==[28.,48.]
    with pytest.raises(ValueError):anchor_prediction(observation,[torch.zeros(1,2,3)],0,32,37)
