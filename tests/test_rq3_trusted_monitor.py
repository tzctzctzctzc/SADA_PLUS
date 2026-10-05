import json

import pytest

from tools.run_rq3_trusted_monitor import observe_idle, partition_rows, gpu_start_race
from tools.run_rq3_matrix import matrix_lock_name


def test_thirty_minutes_are_continuous_and_busy_card_resets_timer():
    timers = {}
    assert observe_idle(timers, {0: 'GPU-A'}, 0, 1800) == []
    assert observe_idle(timers, {0: 'GPU-A'}, 1799, 1800) == []
    assert observe_idle(timers, {}, 1800, 1800) == []
    assert observe_idle(timers, {0: 'GPU-A'}, 1900, 1800) == []
    assert observe_idle(timers, {0: 'GPU-A'}, 3699, 1800) == []
    assert observe_idle(timers, {0: 'GPU-A'}, 3700, 1800) == [0]


def test_uuid_change_and_locked_card_clear_previous_idle_time():
    timers = {}
    observe_idle(timers, {0: 'GPU-A', 1: 'GPU-B'}, 0, 1800)
    assert observe_idle(timers, {0: 'GPU-C'}, 1800, 1800) == []
    assert set(timers) == {0}
    assert observe_idle(timers, {0: 'GPU-C'}, 3600, 1800) == [0]


def test_two_partitions_skip_completed_without_overlap_or_hard():
    rows = [dict(teacher='egovideo_hod', strategy=strategy, shift='S' + str(shift))
            for shift in range(1, 7) for strategy in ('verify', 'prototype')]
    done = {('egovideo_hod', 'verify', 'S1'): '/old/complete.json'}
    first, second = partition_rows({'entries': rows}, done)
    keys = lambda group: {(r['teacher'], r['strategy'], r['shift']) for r in group}
    assert not keys(first) & keys(second)
    assert len(first) + len(second) == 11
    assert not set(done) & (keys(first) | keys(second))


def test_managed_matrices_have_distinct_locks_and_legacy_lock_is_preserved(monkeypatch):
    monkeypatch.delenv('RQ3_QUEUE_SLOT', raising=False)
    assert matrix_lock_name(0, 1) == '.rq3_matrix_0_of_1.lock'
    monkeypatch.setenv('RQ3_MONITOR_PID', '123')
    monkeypatch.setenv('RQ3_QUEUE_SLOT', '0')
    first = matrix_lock_name(0, 1)
    monkeypatch.setenv('RQ3_QUEUE_SLOT', '1')
    assert matrix_lock_name(0, 1) != first
    monkeypatch.setenv('RQ3_QUEUE_SLOT', '2')
    with pytest.raises(ValueError):
        matrix_lock_name(0, 1)


def test_only_pre_cuda_collision_can_retry_not_started_training(tmp_path):
    state = dict(status='failed', shift='S2', teacher='egovideo_hod', strategy='prototype')
    (tmp_path / 'state.json').write_text(json.dumps(state))
    log = (tmp_path / (tmp_path.name + '_formal_S2_egovideo_hod_prototype')
           / 'egovideo_hod/prototype/S2_seed6/logs/uda.train.log')
    log.parent.mkdir(parents=True)
    message = 'guard_before_cuda: selected physical GPU is occupied'
    log.write_text(message)
    assert gpu_start_race(tmp_path)
    log.write_text('RQ3_PRE_CUDA_IDLE_CHECK_PASS\nepoch 1\n' + message)
    assert not gpu_start_race(tmp_path)
    log.write_text('CUDA out of memory')
    assert not gpu_start_race(tmp_path)
