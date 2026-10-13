import pytest
from tools.rq3_colocation import choose_student_groups
from tools.resume_rq3_gpu1 import split_remaining, TEACHER
from tools.benchmark_rq3_temporal_concurrency import compare


def test_prototype_only_leaves_hard_and_verify_serial():
    jobs=[dict(strategy=s, identity=s+str(i)) for s in ('hard','verify','prototype') for i in (1,2)]
    groups=choose_student_groups(jobs, ['prototype'])
    assert [len(g) for g in groups] == [1,1,1,1,2]
    assert {j['identity'] for g in groups for j in g} == {j['identity'] for j in jobs}
    with pytest.raises(ValueError): choose_student_groups(jobs, ['verify'])


def test_exact_recovery_boundary_excludes_completed():
    done={(TEACHER,'hard','S1'),(TEACHER,'verify','S1')}
    first, remaining=split_remaining(done)
    assert len(first)==1 and len(remaining)==15
    assert first[0]['strategy']=='prototype'
    assert not done & {(r['teacher'],r['strategy'],r['shift']) for r in first+remaining}
    with pytest.raises(ValueError): split_remaining(done | {(TEACHER,'prototype','S1')})


def test_missing_benchmark_reports_cannot_enable():
    with pytest.raises(ValueError): compare([], [])
