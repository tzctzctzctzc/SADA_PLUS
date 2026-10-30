import numpy as np
from tools.background_diagnostics import error_counts,occupancy,routing_counts,iou


def test_error_categories_and_matching_are_disjoint():
    gt={'v':[dict(segment=[0.,10.],label=0),dict(segment=[20.,30.],label=1)]}
    detections=[dict(video_id='v',segment=s,label=c,score=1.-i*.1) for i,(s,c) in enumerate([
        ([0.,10.],0),([0.,10.],0),([20.,30.],0),([8.,18.],0),([28.,38.],0),([50.,60.],0)])]
    result=error_counts(detections,gt,.5)
    for category in ['true_positive','duplicate','classification','localization','classification_and_localization','background']:assert result[category]==1
    assert result['missed_gt']==1 and result['predictions']==6


def test_occupancy_is_not_training_assignment():
    inside=occupancy([0,2,4,6],[[1,5]])
    counts=routing_counts(np.array([[.1],[.1],[.9],[.1]]),[1,1,1,0],[0,0,1,0],inside,.7)
    assert counts['valid']==3 and counts['routed_background']==2
    assert counts['background_inside_action_but_unassigned']==1
    assert counts['background_but_assigned_positive']==0


def test_empty_predictions_and_no_gt_cases():
    assert error_counts([],{'v':[dict(segment=[0,1],label=0)]},.5)['missed_gt']==1
    assert iou([0,1],[]).size==0
    assert not occupancy([0,1],[]).any()
