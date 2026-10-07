import json

import pytest

from tools.run_rq3_matrix import load_entry_plan, audit_shifts


def test_intern_remaining_eight_are_fresh_not_fake_old_results(tmp_path):
    rows = [dict(teacher='internvideo2_clip_1b', strategy='prototype', shift='S'+str(i)) for i in range(1,7)]
    rows += [dict(teacher='internvideo2_clip_1b', strategy='verify', shift=s) for s in ('S4','S6')]
    p=tmp_path/'plan.json'
    p.write_text(json.dumps(dict(schema='rq3-revised-fresh-matrix-v1', seed=6, fresh=True, entries=rows)))
    entries=load_entry_plan(p)
    assert len(entries)==8
    assert not set(entries) & {('internvideo2_clip_1b','verify',s) for s in ('S1','S2','S3','S5')}


@pytest.mark.parametrize('change', ['hard','duplicate','seed','fresh','unknown_shift'])
def test_invalid_fresh_plan_is_rejected(tmp_path,change):
    row=dict(teacher='internvideo2_clip_1b',strategy='prototype',shift='S1')
    plan=dict(schema='rq3-revised-fresh-matrix-v1',seed=6,fresh=True,entries=[row])
    if change=='hard':row['strategy']='hard'
    elif change=='duplicate':plan['entries'].append(dict(row))
    elif change=='seed':plan['seed']=7
    elif change=='fresh':plan['fresh']=False
    else:row['shift']='S7'
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    with pytest.raises(ValueError):load_entry_plan(p)


def test_fresh_scope_only_checks_used_shifts_but_old_full_gate_is_preserved():
    entries=[('internvideo2_clip_1b','prototype','S4')]
    assert audit_shifts(entries,True)==['S4']
    assert audit_shifts(entries,False)==['S1','S2','S3','S4','S5','S6']
