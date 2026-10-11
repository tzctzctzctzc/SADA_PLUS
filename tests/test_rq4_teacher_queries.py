from copy import deepcopy
import json
import pytest

from tools.rq4_protocol import digest
from tools.rq4_teacher_queries import (choose,tasks,parse_answer,match_gt,answer_for,
                                      geometry_answer,evaluate,POLICY,validate_completion)


def candidate():
    return dict(candidate_id='S1-v-p1-context3',video_id='v',proposal=[12.,16.],
                window=[8.,20.],class_id=0,anchor_sec=14.,fpn_level=0)


def test_fixed_tasks_and_local_time():
    generated=tasks(candidate(),'taking an object')
    assert len(generated)==7 and len({r['task_id'] for r in generated})==7
    assert generated[0]['time']==11. and 'time 3.000' in generated[0]['prompt']
    assert '[4.000, 8.000]' in generated[0]['prompt']
    assert not any(k in r for r in generated for k in ('gt','answer','annotations'))


def test_labels_cannot_change_queries():
    c=candidate();base=tasks(c,'test')
    c.update(gt=[1.,100.],annotations=[dict(segment=[0,1])])
    assert tasks(c,'test')==base


def test_swap_same_boxes_reversed():
    a,b=tasks(candidate(),'test')[-2:]
    assert a['boxes']==list(reversed(b['boxes']))
    assert not a['swapped'] and b['swapped']


@pytest.mark.parametrize('kind,text,expected',[
    ('state','DURING','DURING'),('state','AFTER.','AFTER'),
    ('direction','KEEP<|im_end|>','KEEP'),('ranking','A or B',None),
    ('ranking','[[1,2]]',None),('ranking','<|TIMESPAN|>',None),
    ('state','The answer is DURING',None),('ranking','B\n','B')])
def test_parser_no_timestamp_or_prose_fallback(kind,text,expected):
    assert parse_answer(kind,text)==expected


def test_selection_independent_of_order_and_gt():
    requests=[]
    for v in ('v1','v2'):
        for i in range(5):
            c=candidate();c.update(video_id=v,candidate_id=f'{v}-p{i}-context3')
            requests.append(dict(candidate=c))
    selected=choose(requests)
    assert selected==choose(list(reversed(requests))) and len(selected)==4


def test_matching_same_class_and_ambiguity():
    c=candidate()
    assert not match_gt(c,[dict(label_id=1,segment=[12,16])])['scorable']
    assert not match_gt(c,[dict(label_id=0,segment=[11,15]),dict(label_id=0,segment=[13,17])])['scorable']
    m=match_gt(c,[dict(label_id=0,segment=[12,16]),dict(label_id=0,segment=[17,18])])
    assert m['gt']==[12,16] and m['repeated_same_class_in_clip']


def test_direction_and_state_labels():
    generated=tasks(candidate(),'test')
    assert [answer_for(t,[12,16]) for t in generated[:3]]==['BEFORE','DURING','AFTER']
    assert answer_for(generated[3],[11,16])=='EARLIER'
    assert answer_for(generated[4],[12,17])=='LATER'
    assert answer_for(generated[3],[12.2,16])=='KEEP'
    assert all(geometry_answer(t)=='KEEP' for t in generated[3:5])


def test_rank_ground_truth_and_swap():
    a,b=tasks(candidate(),'test')[-2:]
    aa=answer_for(a,candidate()['proposal']);bb=answer_for(b,candidate()['proposal'])
    assert (aa,bb) in [('A','B'),('B','A')]


def test_score_invalid_counts_and_input_binding():
    task=tasks(candidate(),'test')[0]
    req=dict(request_id='r',task=task,condition='video',clip_path='/tmp/clip',clip_sha256='123')
    response=dict(request_id='r',request_sha256=digest(req),raw_text='[[1,2]]')
    truth={task['task_id']:dict(scorable=True,answer='BEFORE')}
    result=evaluate([req],[response],truth)
    assert result['summary']['state_video']['accuracy']==0
    assert result['summary']['state_video']['valid_rate']==0
    json.dumps(result,allow_nan=False)
    response['request_sha256']='wrong'
    with pytest.raises(ValueError):evaluate([req],[response],truth)


def test_unmatched_not_assumed_absent():
    task=tasks(candidate(),'test')[0]
    req=dict(request_id='r',task=task,condition='video')
    response=dict(request_id='r',request_sha256=digest(req),raw_text='ABSENT')
    result=evaluate([req],[response],{task['task_id']:dict(scorable=False,answer=None)})
    assert result['summary']['state_video']['count']==0
    assert result['all_response_format']['video']==dict(count=1,valid=1)


def test_completion_does_not_accept_missing_or_reordered_answers():
    task=tasks(candidate(),'test')[0]
    requests=[dict(request_id=str(i),task=task,condition='video') for i in range(16)]
    responses=[dict(request_id=r['request_id'],request_sha256=digest(r),raw_text='invalid') for r in requests]
    validate_completion(requests,responses,dict(status='complete',count=16))
    validate_completion(requests,responses[:14],dict(status='unsupported_answer_interface',count=14))
    with pytest.raises(ValueError):validate_completion(requests,responses[:14],dict(status='complete',count=14))
    with pytest.raises(ValueError):validate_completion(requests,list(reversed(responses)),dict(status='complete',count=16))
    responses[0]['raw_text']='DURING'
    with pytest.raises(ValueError):validate_completion(requests,responses[:14],dict(status='unsupported_answer_interface',count=14))
