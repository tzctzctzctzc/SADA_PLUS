import copy
import pytest
from tools.vlm_background_protocol import (score,frame_indices,prompt,option_order,decisions,digest,verify_response)


def row():
    return dict(view_id='physical-clip',label=0,observable=True,visible_interval=[1.,2.])


def answer(r,q,state):
    order=option_order(r['view_id'],q);z=[-3.]*3;z[order.index(state)]=3.
    return dict(logits=z)


def test_other_task_action_is_not_background_and_uncertainty_preserved():
    r=row();responses=dict(presence=answer(r,'presence',0),predicted=answer(r,'predicted',1),
                         others=answer(r,'others',0),static=answer(r,'presence',0))
    d=decisions(r,responses);assert not d['verify']['delete'] and not d['presence']['delete']
    responses['others']=answer(r,'others',2);assert not decisions(r,responses)['verify']['delete']


def test_both_absent_and_logit_contrast_not_probability_subtraction():
    r=row();responses=dict(presence=answer(r,'presence',1),predicted=answer(r,'predicted',1),
                         others=answer(r,'others',1),static=answer(r,'presence',0))
    d=decisions(r,responses);assert all(d[m]['delete'] for m in d)
    assert d['contrast']['BG_score']>d['presence']['BG_score']


def test_ties_missing_evidence_and_insufficient_video_never_delete():
    assert score([2.,2.,0.],[0,1,2])['state']=='uncertain'
    r=row();r['observable']=False;assert all(not x['delete'] for x in decisions(r,{}).values())
    with pytest.raises(ValueError):score([float('nan'),0,1],[0,1,2])


def test_real_pts_and_short_and_outside_intervals():
    times=[i*.02 for i in range(200)]
    d=frame_indices(times,1,2);assert d['observable'] and len(d['indices'])==16
    assert d['times']==sorted(d['times']) and d['inside_unique']==12
    assert not frame_indices(times,1,1.03)['observable']
    assert not frame_indices(times,-.1,1)['observable']
    with pytest.raises(ValueError):frame_indices([0,1,1],0,1)


def test_prompt_never_equates_wrong_class_partial_or_static_state_to_task_presence():
    r=row();text=prompt(r,'presence');assert 'Partial execution counts YES' in text
    assert 'persistent opened/closed/on/off state' in text and 'ANY of the ten' in text
    assert 'TAKE:' not in prompt(r,'others').split('other nine')[1]


def test_resume_rejects_different_input_even_if_logits_exist():
    req=dict(option_order=[0,1,2],frames=['hash1'],prompt='frozen')
    res=dict(request=copy.deepcopy(req),request_id=digest(req),logits=[0.,1.,-1.],logsumexp=2.)
    assert verify_response(req,res)==res
    req['frames']=['different']
    with pytest.raises(ValueError):verify_response(req,res)
