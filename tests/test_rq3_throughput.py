import copy
import pytest
from tools.benchmark_rq3_temporal_concurrency import compare, digest_state


def reports(seconds):
    return [dict(measurements=[dict(seconds=seconds,state_sha256=str(i),metrics={'loss':1.})
                              for i in range(2)]) for _ in range(2)]


def test_requires_both_exact_semantics_and_speed():
    assert compare(reports(10),reports(12))['enable_candidate']
    assert not compare(reports(10),reports(20))['enable_candidate']
    changed=reports(12);changed[1]['measurements'][1]['state_sha256']='different'
    assert not compare(reports(10),changed)['enable_candidate']
    changed=reports(12);changed[0]['measurements'][0]['metrics']['loss']=1.001
    assert not compare(reports(10),changed)['enable_candidate']


def test_hash_covers_optimizer_and_model():
    import torch
    a={'weight':torch.ones(3),'optim':{'step':torch.tensor(1.),'moment':torch.zeros(3)}}
    b=copy.deepcopy(a)
    assert digest_state(a)==digest_state(b)
    b['optim']['moment'][1]=1
    assert digest_state(a)!=digest_state(b)


def test_handoff_excludes_only_exact_completed():
    from tools.rq3_pipeline_handoff import remaining_rows
    t='timeple_8b';done={(t,s,'S1') for s in ('hard','verify','prototype')}
    rows=remaining_rows(t,done)
    assert len(rows)==15
    assert {r['shift'] for r in rows}=={'S2','S3','S4','S5','S6'}
    with pytest.raises(ValueError):remaining_rows(t,{('other','hard','S1')})
