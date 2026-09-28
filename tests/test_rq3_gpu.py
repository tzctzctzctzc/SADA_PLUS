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
    assert calls == [2, 2, 2]  # 取得卡锁后再复核一次。


@pytest.mark.skipif(sys.platform == "win32", reason="production locks are Linux flock")
def test_continuous_idle_hour_resets_on_external_use(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "gpu_indices", lambda request: [0])
    clock = [0]
    waits = []
    monkeypatch.setattr(gpu.time, "monotonic", lambda: clock[0])

    def sleep(seconds):
        waits.append(clock[0])
        clock[0] += 600

    def idle(index):
        if clock[0] == 1800:
            raise RuntimeError("external task entered")
        return "GPU-0"

    monkeypatch.setattr(gpu.time, "sleep", sleep)
    monkeypatch.setattr(gpu, "idle_gpu_uuid", idle)
    with gpu.gpu_slot("auto", tmp_path, wait=True, idle_seconds=3600) as selected:
        assert selected["index"] == 0
        assert clock[0] == 6000  # 2400时恢复空闲，不能沿用最早的0时刻。
        with pytest.raises(RuntimeError, match="no idle permitted"):
            with gpu.gpu_slot("0", tmp_path):
                pytest.fail("physical GPU lock must remain held")
    assert 3600 in waits


@pytest.mark.skipif(sys.platform == "win32", reason="production locks are Linux flock")
def test_lease_reuses_card_without_another_hour_and_releases_locks(monkeypatch, tmp_path):
    monkeypatch.setattr(gpu, "gpu_indices", lambda request: [0])
    monkeypatch.setattr(gpu, "idle_gpu_uuid", lambda index: "GPU-0")
    clock = [0]
    monkeypatch.setattr(gpu.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    lease = gpu.GpuLease("auto", tmp_path, 3600)
    try:
        with lease.stage() as first:
            assert clock[0] == 3600
        with lease.stage() as second:
            assert second == first
            assert clock[0] == 3600
    finally:
        lease.close()
    with gpu.gpu_slot("auto", tmp_path) as selected:
        assert selected == first


def test_cpu_preparation_guard_rejects_card_taken_by_external_task(monkeypatch):
    monkeypatch.setenv("RQ3_GPU_INDEX", "2")
    monkeypatch.setenv("RQ3_GPU_UUID", "GPU-2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-2")
    monkeypatch.setattr(gpu, "idle_gpu_uuid", lambda index: (_ for _ in ()).throw(RuntimeError("occupied")))
    with pytest.raises(RuntimeError, match="occupied"):
        gpu.guard_before_cuda()


def test_resume_only_skips_complete_records_and_rejects_duplicate_identity(tmp_path):
    import json
    from tools.run_rq3_matrix import completed_entries, matrix
    first = tmp_path / "old0/egovideo_hod/hard/S1_seed6"
    first.mkdir(parents=True)
    record = dict(teacher="egovideo_hod", strategy="hard", shift="S1", seed=6)
    (first / "complete.json").write_text(json.dumps(record))
    partial = tmp_path / "old0/egovideo_hod/hard/S2_seed6"
    partial.mkdir(parents=True)
    (partial / "preflight.json").write_text("{}")
    done = completed_entries([tmp_path / "old0"])
    assert len(done) == 1
    assert len([entry for entry in matrix() if entry not in done]) == 53
    duplicate = tmp_path / "old1/complete.json"
    duplicate.parent.mkdir()
    duplicate.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="duplicate"):
        completed_entries([tmp_path / "old0", tmp_path / "old1"])


def test_matrix_shards_cover_54_runs_without_duplicates():
    from tools.run_rq3_matrix import matrix
    first, second = matrix(0, 2), matrix(1, 2)
    assert len(first) == len(second) == 27
    assert not set(first) & set(second)
    assert set(first) | set(second) == set(matrix())
    with pytest.raises(ValueError):
        matrix(2, 2)
