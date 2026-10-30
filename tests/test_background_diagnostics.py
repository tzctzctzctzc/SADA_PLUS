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


def test_entry_script_compiles_before_remote_launch():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/run_background_diagnostics.py'
    compile(path.read_text(encoding='utf8'),str(path),'exec')


def test_four_routing_arms_preserve_random_coverage_and_source_threshold():
    from tools.background_routing_replay import masks,weighted_lower
    result=masks(np.array([.01,.02,.3,.4,.9]),np.zeros(5),np.ones(5,dtype=bool),.7,.1,123)
    assert result['original'].sum()==4
    assert result['protect_ambiguous'].sum()==result['matched_random'].sum()==2
    assert result['disable_background'].sum()==0
    assert weighted_lower([np.array([.2,.4])]) is None
    assert weighted_lower([np.array([.2,.4]),np.array([.1,.3])])==.1


def test_source_actual_routing_uses_gt_not_classifier_score():
    counts=routing_counts(np.array([[.01],[.01]]),[True,True],[True,False],[True,False],.6,background=[False,True])
    assert counts['routed_background']==1 and counts['background_but_assigned_positive']==0
