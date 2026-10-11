from copy import deepcopy
import json
import numpy as np

from tools.rq4_temporal_information import (scores, direct_report, oracle_report,
    geometry, teacher_features, corrected, predict_fold, cross_validate,
    shuffled_features, paired_uncertainty)


def rows():
    return [dict(video_id='v'+str(i//20),candidate_id=str(i),proposal=[float(i+1),float(i+3)],
                 gt=[float(i+1.2),float(i+3.2)],duration=200.,fpn_level=i%3,anchor_sec=i+2.) for i in range(80)]


def test_metrics_perfect_and_unmatched():
    assert scores([1,3],[1,3])==dict(iou=1.,start_error=0.,end_error=0.)
    assert scores([1,3],None)['iou']==0.


def test_abstention_retains_proposal():
    data=rows(); report=direct_report(data,[[None,None] for _ in data])
    for r in report.values():
        assert r['coverage']==0 and r['delta_iou']==0


def test_oracle_includes_identity_and_can_mix_endpoints():
    data=[dict(video_id='v',proposal=[1.,4.],gt=[2.,3.])]
    report=oracle_report(data,[[[2.,4.],[1.,3.]]])
    assert report['endpoint_mix']['mean_iou']==1
    assert report['whole_box']['mean_iou']==.5
    assert not report['deployable']


def test_features_do_not_read_gt_or_selected_flag():
    data=rows();pairs=[[r['proposal'],None] for r in data]
    other=deepcopy(data)
    for r in other:r.update(gt=None,selected=True)
    np.testing.assert_array_equal(geometry(data),geometry(other))
    np.testing.assert_array_equal(teacher_features(data,pairs),teacher_features(other,pairs))


def test_heldout_gt_cannot_change_fold_predictions():
    data=rows();x=np.concatenate([geometry(data),np.zeros((len(data),7))],1)
    ids,boxes,proof=predict_fold(data,x,'v0')
    other=deepcopy(data)
    for r in other:
        if r['video_id']=='v0':r['gt']=[20.,180.]
    assert predict_fold(other,x,'v0')[1]==boxes
    assert 'v0' not in proof['train_videos'] and len(ids)==20


def test_same_information_identical_cross_validation():
    data=rows();blank=np.zeros((len(data),7))
    result=cross_validate(data,blank)
    assert result==cross_validate(data,blank.copy())
    assert json.loads(json.dumps(result,allow_nan=False))==result


def test_shuffle_preserves_video_distribution():
    data=rows();x=np.arange(len(data)*7).reshape(-1,7)
    a=shuffled_features(x,data,6);b=shuffled_features(x,data,6)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(x,a)
    for k in range(4):
        assert sorted(map(tuple,a[k*20:(k+1)*20]))==sorted(map(tuple,x[k*20:(k+1)*20]))


def test_correction_limits_and_invalid_fallback():
    assert corrected([1.,3.],[99.,99.],10.)==[3.,5.]
    assert corrected([1.,3.],[1.,-1.],10.)==[1.,3.]
    assert corrected([1.,3.],[-1.,0.],10.)==[0.,3.]


def test_direction_not_equivalent_to_improvement():
    data=[dict(video_id='v',proposal=[1.,4.],gt=[2.,3.])]
    result=direct_report(data,[[[4.,5.],None]])['2x']['start_correction']
    assert result['correct_direction']==1 and result['worsened']==1


def test_video_pairing_bootstrap():
    result=paired_uncertainty({'a':.4,'b':.6},{'a':.3,'b':.5})
    assert abs(result['mean_delta']-.1)<1e-9
    assert result['videos_improved']==2 and result['video_count']==2
