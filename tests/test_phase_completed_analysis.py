import numpy as np
from tools.phase_completed_analysis import error_partition, iou_matrix


def test_tp_duplicate_and_background_partition():
    r=error_partition(np.array([[1.,0.],[1.,0.],[0.,0.]]),np.array([0,0,0]),np.array([0,1]),np.array([.8,.9,.7]),.5)
    assert r['tp']==1 and r['duplicate']==1 and r['background']==1 and sum(r.values())==3


def test_classification_localization_and_overlap_are_distinct():
    matrix=np.array([[0.,.8],[.3,0.],[0.,.3],[.05,0.]])
    r=error_partition(matrix,np.array([0,0,0,0]),np.array([0,1]),np.array([.9,.8,.7,.6]),.5)
    assert r['classification']==r['localization']==r['class_and_localization']==r['near_background']==1


def test_empty_truth_and_seconds_iou():
    m=iou_matrix(np.array([[1.,3.],[4.,5.]]),np.array([[2.,4.]]))
    assert np.allclose(m[:,0],[1/3,0])
    r=error_partition(np.empty((2,0)),np.array([0,1]),np.array([],dtype=int),np.array([.1,.2]),.5)
    assert r['background']==2
