"""机制指标不能把稳定错误、缺失检测或平移当改善。"""
from copy import deepcopy
import pytest
from tools.check_context_stability import iou, summarize_views


def example(label=2,box=(2.,6.),det=True):
    return dict(class_id=label,gt_score=.8,box=list(box),
                detection=dict(class_id=label,box=list(box)) if det else None)


def test_perfect_invariance_and_accuracy():
    r=summarize_views([example() for _ in range(3)],[2.,6.],2)
    assert r['boundary_pair_l1']==r['boundary_error']==r['class_flip_rate']==0
    assert r['class_accuracy']==r['mean_iou']==r['detection_success_05']==1


def test_stable_but_wrong_is_not_success():
    r=summarize_views([example(1,(8.,10.)) for _ in range(3)],[2.,6.],2)
    assert r['class_flip_rate']==r['boundary_pair_l1']==0
    assert r['class_accuracy']==r['mean_iou']==r['detection_success_05']==0
    assert r['boundary_error']>0


def test_translation_and_scale_normalization():
    rows=[example(box=(2.,6.)),example(box=(3.,7.)),example(box=(1.,5.))]
    a=summarize_views(rows,[2.,6.],2)
    shifted=deepcopy(rows)
    for r in shifted:
        r['box']=[3*x+100 for x in r['box']]
        r['detection']['box']=[3*x+100 for x in r['detection']['box']]
    b=summarize_views(shifted,[106.,118.],2)
    assert a==b


def test_missing_detections_are_counted():
    r=summarize_views([example(det=False),example(det=False),example()],[2.,6.],2)
    assert r['detection_miss_rate']==pytest.approx(2/3)
    assert r['detection_success_05']==pytest.approx(1/3)


def test_class_changes_and_boundary_changes_are_detected():
    rows=[example(),example(1,(3.,7.)),example()]
    r=summarize_views(rows,[2.,6.],2)
    assert r['class_flip_rate']==pytest.approx(2/3)
    assert r['boundary_pair_l1']>0


def test_iou_no_overlap_and_exact():
    assert iou([1,2],[3,4])==0
    assert iou([1,2],[1,2])==1
