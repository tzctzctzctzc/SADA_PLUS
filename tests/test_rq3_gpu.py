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
def test_auto_uses_idle_card_and_global_lock_prevents_second_rq3_job(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "gpu_indices", lambda request: [0, 1, 2, 3])

    def idle(index):
        if index != 1:
            raise RuntimeError("occupied")
        return "GPU-1"

    monkeypatch.setattr(gpu, "idle_gpu_uuid", idle)
    with gpu.gpu_slot("auto", tmp_path) as selected:
        assert selected == {"index": 1, "uuid": "GPU-1"}
        with pytest.raises(BlockingIOError):
            with gpu.gpu_slot("auto", tmp_path):
                pytest.fail("a second RQ3 GPU must not be acquired")
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
