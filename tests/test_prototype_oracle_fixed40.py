"""只测试改范围后的身份/保存点边界，不重复模型回归。"""
from runpy import run_path
from pathlib import Path
import pytest

module = run_path(str(Path(__file__).resolve().parents[1]/'tools/run_prototype_oracle_fixed40.py'))


def test_exact_four_remaining_excludes_completed_and_cancelled_groups():
    done = [dict(group='fixed40', shift=s) for s in ('S1', 'S2')]
    done += [dict(group='clean20', shift='S1')]
    assert module['remaining_plan'](done) == [('fixed40', s) for s in ('S3', 'S4', 'S5', 'S6')]
    with pytest.raises(ValueError):
        module['remaining_plan'](done+[done[0]])
    with pytest.raises(ValueError):
        module['remaining_plan'](done[:1])
    with pytest.raises(ValueError):
        module['remaining_plan'](done+[dict(group='fixed40', shift='S3')])


def test_partial_first_epoch_allowed_but_any_saved_or_complete_epoch_rejected(tmp_path):
    (tmp_path/'train.log').write_text('epoch1 partial batches')
    module['assert_unstarted'](tmp_path)
    for name in ('best.pt', 'latest.pt', 'termination.json', 'complete.json', 'metrics.jsonl'):
        p = tmp_path/name
        p.write_text('saved')
        with pytest.raises(ValueError):
            module['assert_unstarted'](tmp_path)
        p.unlink()


def test_live_predecessor_or_wrong_stop_reason_rejected():
    check = module['assert_previous_stopped']
    reason = 'user_keep_fixed40_only_S1_S6_cancel_other_groups'
    check(dict(reason=reason, remaining_pids=[], own_pids=[]))
    with pytest.raises(ValueError):
        check(dict(reason='other', remaining_pids=[], own_pids=[]))
    with pytest.raises(ValueError):
        check(dict(reason=reason, remaining_pids=[1], own_pids=[]))
