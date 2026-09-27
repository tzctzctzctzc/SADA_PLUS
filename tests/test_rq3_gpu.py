import sys

import pytest

from tools import rq3_gpu as gpu


def test_auto_lists_physical_gpus_and_explicit_request_does_not_scan(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *args, **kw: "0\n1\n2\n3\n")
    assert gpu.gpu_indices("auto") == [0, 1, 2, 3]
    assert gpu.gpu_indices("1") == [1]
    with pytest.raises(ValueError):
        gpu.gpu_indices("-1")


@pytest.mark.parametrize("memory,util,processes", [(4096, 0, ""), (4, 95, ""), (4, 0, "GPU-1, 42")])
def test_busy_gpu_rejected_even_when_process_table_is_empty(monkeypatch, memory, util, processes):
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda command, **kw:
                        "GPU-1, %d, %d" % (memory, util) if "-i" in command else processes)
    with pytest.raises(RuntimeError, match="occupied"):
        gpu.idle_gpu_uuid(1)


@pytest.mark.skipif(sys.platform == "win32", reason="production locks are Linux flock")
def test_two_idle_cards_allowed_and_third_rq3_job_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "gpu_indices", lambda request: [0, 1, 2, 3])

    def idle(index):
        if index not in (1, 2):
            raise RuntimeError("occupied")
        return "GPU-%d" % index

    monkeypatch.setattr(gpu, "idle_gpu_uuid", idle)
    with gpu.gpu_slot("auto", tmp_path) as selected:
        assert selected == {"index": 1, "uuid": "GPU-1"}
        with gpu.gpu_slot("auto", tmp_path) as second:
            assert second == {"index": 2, "uuid": "GPU-2"}
            with pytest.raises(BlockingIOError):
                with gpu.gpu_slot("auto", tmp_path):
                    pytest.fail("a third RQ3 GPU must not be acquired")
    with gpu.gpu_slot("auto", tmp_path) as selected:
        assert selected["index"] == 1


@pytest.mark.skipif(sys.platform == "win32", reason="production locks are Linux flock")
def test_wait_rechecks_until_card_is_idle(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "gpu_indices", lambda request: [2])
    calls = []

    def idle(index):
        calls.append(index)
        if len(calls) == 1:
            raise RuntimeError("occupied")
        return "GPU-2"

    monkeypatch.setattr(gpu, "idle_gpu_uuid", idle)
    monkeypatch.setattr(gpu.time, "sleep", lambda _: None)
    with gpu.gpu_slot("auto", tmp_path, wait=True) as selected:
        assert selected["uuid"] == "GPU-2"
    assert calls == [2, 2]


def test_matrix_shards_cover_54_runs_without_duplicates():
    from tools.run_rq3_matrix import matrix
    first, second = matrix(0, 2), matrix(1, 2)
    assert len(first) == len(second) == 27
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(matrix())
    with pytest.raises(ValueError):
        matrix(2, 2)
