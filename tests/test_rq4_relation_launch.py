import json
import pytest
from tools.rq4_relation_launch import predecessor, validate_acceptance
from tools.run_rq4_soft_relation_matrix import entries


def test_only_two_teachers_six_shifts_no_control_or_legacy_strategy():
    values=entries()
    assert len(values)==len(set(values))==12
    assert {t for t,s in values}=={'timelens2_8b','timeple_8b'}
    assert {s for t,s in values}=={f'S{i}' for i in range(1,7)}


def test_count_only_cannot_release_new_experiment(tmp_path):
    path=tmp_path/'final_audit.json'
    path.write_text(json.dumps(dict(status='passed',count=24,entries=[])))
    with pytest.raises(ValueError,match='exact original'):
        predecessor(path,'old-head')


def test_old_acceptance_cannot_authorize_soft_relation():
    with pytest.raises(ValueError,match='variant'):
        validate_acceptance(dict(status='passed',variant='rq4-relational-boundary-v1'))
