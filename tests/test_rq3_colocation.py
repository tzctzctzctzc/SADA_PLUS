"""共享卡只改变运行控制；拒绝未知进程，策略分组不重复。"""
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.rq3_colocation import check_admission, choose_student_groups, current_lease, owners


@pytest.mark.parametrize('before,current,after,limit', [
    (set(), {7}, {7, 8}, 2), ({7}, {7}, {7, 8, 9}, 2),
    ({7}, {7}, {8}, 2), ({7}, {7}, {7}, 2), ({7}, {7}, {7, 8}, 1)])
def test_reject_unknown_or_ambiguous_cuda_owner(before, current, after, limit):
    with pytest.raises(RuntimeError):
        check_admission(before, current, after, limit)


def test_unique_host_pid_registration():
    assert check_admission(set(), set(), {100001}, 2) == 100001
    assert check_admission({100001}, {100001}, {100001, 100003}, 2) == 100003


def test_only_hard_and_prototype_paired_without_duplicates():
    jobs = [dict(identity='%s_%s' % (s, i), strategy=s)
            for i in range(3) for s in ('hard', 'verify', 'prototype')]
    groups = choose_student_groups(jobs)
    assert [len(g) for g in groups] == [2, 1, 1, 1, 1, 2, 1]
    assert all(len(g) == 1 for g in groups if g[0]['strategy'] == 'verify')
    assert sorted(j['identity'] for g in groups for j in g) == sorted(j['identity'] for j in jobs)
    with pytest.raises(ValueError, match='duplicate'):
        choose_student_groups(jobs + [jobs[0]])
    with pytest.raises(ValueError, match='unknown'):
        choose_student_groups([dict(identity='a', strategy='unknown')])


def test_finished_cuda_record_does_not_authorize_other_process(tmp_path):
    (tmp_path/'cuda_0_train.json').write_text(json.dumps(dict(host_pid=40)))
    assert owners(tmp_path, {40, 50}) == {40}
    assert owners(tmp_path, {50}) == set()


def test_lease_validates_live_parent_and_process_group(tmp_path, monkeypatch):
    path = tmp_path/'lease.json'
    path.write_text(json.dumps(dict(owner_pid=111, owner_start='22', members={'0': 333, '1': 444})))
    monkeypatch.setenv('RQ3_COHORT_LEASE', str(path))
    monkeypatch.setenv('RQ3_COHORT_MEMBER', '0')
    with patch('tools.rq3_colocation.process_start', return_value='22'), patch('os.getpgrp', return_value=333):
        assert current_lease()[2] == '0'
    with patch('tools.rq3_colocation.process_start', return_value='23'), patch('os.getpgrp', return_value=333):
        with pytest.raises(RuntimeError, match='owner'):
            current_lease()
    with patch('tools.rq3_colocation.process_start', return_value='22'), patch('os.getpgrp', return_value=999):
        with pytest.raises(RuntimeError, match='belong'):
            current_lease()


def test_pair_mode_does_not_edit_learning_code():
    root = Path(__file__).resolve().parents[1]
    source = (root/'tools/run_rq3_trusted_queue.py').read_text(encoding='utf-8')
    assert 'fixed batch2, without batch benchmark' in source
    assert "shift_number % 2 == 0" in source
    assert "'cohort_finished.json'" in source
