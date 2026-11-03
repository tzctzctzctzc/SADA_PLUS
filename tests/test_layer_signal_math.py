import numpy as np
import torch
from tools.layer_signal_math import duration_levels,raw_projection,conditional_auc,pool_signals


def test_projection_is_fixed_and_not_label_dependent():
    assert torch.equal(raw_projection(),raw_projection())
    assert raw_projection().shape==(2304,512)


def test_duration_selects_original_grid_ranges_without_GT():
    boxes=torch.tensor([[0.,6.],[0.,16.],[0.,30.]])
    assert duration_levels(boxes,[(0,4),(4,8),(8,100)]).tolist()==[0,1,2]


def test_constant_feature_has_no_temporal_change_and_stride_coordinates_match():
    x=torch.ones(4,32);v=torch.ones(32,dtype=torch.bool);b=torch.tensor([[8.,16.]])
    mean,globalmean,stats=pool_signals(x,v,b,stride=2)
    assert torch.equal(mean,torch.ones(1,4)) and stats[0,1]==0 and stats[0,2]==0


def test_conditional_auc_excludes_same_video_and_unmatched_bins():
    labels=np.array([True,False,False,True]);s=np.array([.9,.8,.1,.7]);classes=np.zeros(4,int);scores=np.full(4,.8);length=np.ones(4)
    report=conditional_auc(labels,s,classes,scores,length,np.array(['a','a','b','b']),{'0':dict(score=[],length=[])})
    assert report['BG_anchors']==2 and report['pairs']==2 and report['video_macro_conditional_AUC']==.5
