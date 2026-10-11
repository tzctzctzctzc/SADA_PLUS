from copy import deepcopy
import pytest
from tools.rq4_protocol import digest
from tools.rq4_native_density import (POLICY,select_requests,prediction,actual_times,
    temporal_audit,evaluate,decision)


def candidate(vid='v',cid='S2-v-p0-context3'):
    return dict(candidate_id=cid,video_id=vid,class_id=0,proposal=[12.,16.],window=[8.,20.],
        video_duration=30.,anchor_sec=14.,fpn_level=0)


def test_selection_fixed_no_gt_order_or_teacher_dependence():
    rows=[]
    for i in range(25):
        for j in range(3):
            rows.append(dict(candidate=candidate('v'+str(i),f'S2-v{i}-p{j}-context3')))
    m=dict(shift='S2',split='source_validation',seed=6,requests=rows)
    a=select_requests([m],{'v0'})
    m['requests']=list(reversed(rows))
    assert a==select_requests([m],{'v0'}) and len(a)==40
    assert 'v0' not in {r['candidate']['video_id'] for r in a}
    m['seed']=123
    with pytest.raises(ValueError):select_requests([m],set())


@pytest.mark.parametrize('raw',['not json','[[5,4]]','[[0,99]]','[]'])
def test_invalid_or_empty_predictions_keep_proposal(raw):
    c=candidate();box,status=prediction(c,dict(raw_text=raw,metadata={},error=''),'timelens2_8b')
    assert box==c['proposal'] and status!='matched'


def test_match_does_not_use_gt_and_rejects_ties():
    c=candidate()
    b,s=prediction(c,dict(raw_text='[[4,8]]',metadata={},error=''),'timelens2_8b')
    assert b==[12,16] and s=='matched'
    b,s=prediction(c,dict(raw_text='[[3,7],[5,9]]',metadata={},error=''),'timelens2_8b')
    assert b==c['proposal'] and s=='ambiguous'


def test_native_codec_failure_not_silently_abstained():
    with pytest.raises(RuntimeError):prediction(candidate(),dict(raw_text='<|TIMESPAN|>',metadata={},error=''),'timeple_8b')
    with pytest.raises(RuntimeError):prediction(candidate(),dict(raw_text='',metadata={},error='OOM'),'timeple_8b')
    b,s=prediction(candidate(),dict(raw_text='2,3',metadata={},error=''),'timeple_8b')
    assert b==[12,16] and s=='invalid_output'


def test_actual_frame_indices_not_requested_fps():
    m=dict(video_metadata=[dict(fps=50.,frames_indices=[0,5,10,15])])
    assert actual_times(m)==[0,.1,.2,.3]
    assert actual_times(dict(frame_timestamps=[0,.2,.4]))==[0,.2,.4]
    with pytest.raises(ValueError):actual_times(dict(frame_timestamps=[1,0]))
    with pytest.raises(ValueError):actual_times(dict(frame_timestamps=[float('nan')]))


def test_temporal_audit_uses_clip_origin():
    a=temporal_audit(candidate(),[3.9,4.,4.1,8.],[12,16])
    assert a['gt']['frames_inside']==3 and a['gt']['nearest_start_seconds']==0
    assert a['gt']['nearest_end_seconds']==0


def fixture():
    requests=[];responses=[];truth={}
    for i in range(12):
        c=candidate('v'+str(i),'S2-v'+str(i)+'-p0-context3')
        truth[c['candidate_id']]=dict(scorable=True,gt=[12,16])
        for condition in POLICY['conditions']:
            r=dict(request_id=c['candidate_id']+condition,condition=condition,request=dict(candidate=c))
            requests.append(r);responses.append(dict(request_id=r['request_id'],request_sha256=digest(r),
                teacher='timelens2_8b',raw_text='[]',error='',metadata=dict(frame_timestamps=[0,4,8,12])))
    return dict(requests=requests),responses,truth


def test_all_invalid_still_in_denominator_and_zero_increment():
    m,r,t=fixture();out=evaluate(m,r,t,'timelens2_8b')
    assert out['summary']['fps16']['count']==12
    assert out['summary']['fps16']['teacher_iou']==1
    assert out['summary']['fps16']['mean_delta_iou']==0
    assert out['output_status']['fps8']['no_match']==12
    assert decision(dict(a=out,b=out))=='do_not_expand_current_direct_transfer_route'


def test_full_response_identity_required():
    m,r,t=fixture()
    with pytest.raises(ValueError):evaluate(m,r[:-1],t,'timelens2_8b')
    with pytest.raises(ValueError):evaluate(m,list(reversed(r)),t,'timelens2_8b')


def test_decision_threshold_and_uncertainty_both_required():
    def report(mean,low,high,n=20):return dict(summary=dict(fps16=dict(video_count=n,mean_delta_iou=mean,interval=[low,high])))
    a=report(.03,.005,.055);b=report(-.02,-.05,.01)
    assert decision(dict(a=a,b=b))=='signal_for_separate_mechanism_validation'
    assert decision(dict(a=report(.03,-.01,.07),b=b))=='inconclusive_no_automatic_expansion'
    assert decision(dict(a=report(.01,.001,.015),b=b))=='do_not_expand_current_direct_transfer_route'
    assert decision(dict(a=report(.03,.01,.06,4),b=b))=='insufficient_scorable_videos'


def test_truth_changes_only_scoring_not_frozen_query():
    m,r,t=fixture();old=deepcopy(m)
    evaluate(m,r,t,'timelens2_8b')
    for x in t.values():x['gt']=[11,17]
    evaluate(m,r,t,'timelens2_8b')
    assert m==old
