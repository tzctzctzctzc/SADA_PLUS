import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import rq3_external_share as share


def observation():
    return dict(index=2, uuid='GPU-authorized', total_mib=81920, used_mib=57890,
                utilization=99, processes=[dict(pid=100, memory_mib=53000),
                                           dict(pid=200, memory_mib=4200)])


def test_busy_authorized_gpu_and_exiting_external_process_are_allowed():
    original = observation()
    share.validate_admission(original)
    current = copy.deepcopy(original)
    current['processes'].pop()
    share.validate_admission(current, original)


@pytest.mark.parametrize('change', ['memory', 'uuid', 'pid', 'gpu'])
def test_new_process_insufficient_memory_or_changed_identity_is_rejected(change):
    original = observation()
    current = copy.deepcopy(original)
    if change == 'memory':
        current['used_mib'] = 70000
    elif change == 'uuid':
        current['uuid'] = 'GPU-other'
    elif change == 'pid':
        current['processes'].append(dict(pid=300, memory_mib=1))
    else:
        current['index'] = 0
    with pytest.raises((RuntimeError, ValueError)):
        share.validate_admission(current, original)


def test_snapshot_reads_only_nvml_metadata(monkeypatch):
    def output(command, **kwargs):
        if '--query-gpu=uuid,memory.total,memory.used,utilization.gpu' in command:
            return 'GPU-a, 81920, 57890, 99\n'
        return 'GPU-b, 99, 500\nGPU-a, 12, 53000\n'
    monkeypatch.setattr(share.subprocess, 'check_output', output)
    value = share.snapshot(2)
    assert value['processes'] == [dict(pid=12, memory_mib=53000)]
    assert value['utilization'] == 99


@pytest.mark.skipif(sys.platform != 'linux', reason='POSIX GPU locks')
@pytest.mark.parametrize('gpu',[1,2])
def test_real_parent_child_guard_lock_lifetime_and_receipt_audit(tmp_path, monkeypatch,gpu):
    current=observation()
    current['index']=gpu
    monkeypatch.setattr(share, 'snapshot', lambda i:current)
    code = """from tools import rq3_external_share as m
import json
m.snapshot=lambda i:json.loads(%r)
m.shared_guard()
""" % json.dumps(current)
    with share.external_gpu_slot(str(gpu), tmp_path, tmp_path/'receipts', 'commit') as card:
        env = dict(os.environ, RQ3_GPU_INDEX=str(gpu), RQ3_GPU_UUID=card['uuid'],
                   CUDA_VISIBLE_DEVICES=card['uuid'],
                   RQ3_EXTERNAL_GPU_LEASE=card['external_share_lease'],
                   RQ3_EXTERNAL_GPU_LEASE_SHA256=card['external_share_sha256'])
        child = subprocess.Popen([sys.executable, '-c', code], env=env, stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        stdout,stderr=child.communicate(timeout=20)
        assert child.returncode == 0, stderr
        assert 'EXTERNAL_SHARE_CHECK_PASS' in stdout
        share.capture_guard_receipt(card,child.pid)
        share.audit_receipt(card, 'commit')
        with pytest.raises(ValueError):
            share.audit_receipt(card, 'wrong-commit')
    # 父控制器仍在，但阶段锁已归还，不能复用旧lease启动下个CUDA。
    child = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True)
    assert child.returncode != 0 and 'released its lock' in child.stderr
    Path(card['external_share_lease']).write_text('{}')
    with pytest.raises(ValueError, match='receipt changed'):
        share.audit_receipt(card, 'commit')


@pytest.mark.skipif(sys.platform!='linux',reason='POSIX process locks')
def test_dynamic_external_pid_churn_is_revalidated_with_memory_and_sha_proof(tmp_path,monkeypatch):
    original=observation()
    monkeypatch.setattr(share,'snapshot',lambda i:original)
    changed=copy.deepcopy(original)
    changed['processes'].append(dict(pid=300,memory_mib=2000))
    changed['used_mib']+=2000
    code="""import json
from tools import rq3_external_share as s
s.snapshot=lambda i:json.loads(%r)
s.shared_guard()
""" % json.dumps(changed)
    with share.external_gpu_slot(2,tmp_path,tmp_path/'receipts','head') as card:
        env=dict(os.environ,RQ3_GPU_INDEX='2',RQ3_GPU_UUID=card['uuid'],CUDA_VISIBLE_DEVICES=card['uuid'],
                 RQ3_EXTERNAL_GPU_LEASE=card['external_share_lease'],RQ3_EXTERNAL_GPU_LEASE_SHA256=card['external_share_sha256'])
        p=subprocess.Popen([sys.executable,'-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        out,error=p.communicate(timeout=20)
        assert p.returncode==0,error
        share.capture_guard_receipt(card,p.pid)
        share.audit_receipt(card,'head')
        proof=json.loads(Path(card['pre_cuda_receipt']).read_text())
        assert proof['observed']['processes'][-1]['pid']==300
        Path(card['pre_cuda_receipt']).write_text('{}')
        with pytest.raises(ValueError,match='receipt changed'):
            share.audit_receipt(card,'head')


def test_external_slot_rejects_other_gpu_without_creating_receipts(tmp_path):
    with pytest.raises(ValueError):
        with share.external_gpu_slot('0', tmp_path, tmp_path/'receipts', 'commit'):
            pass
    assert not (tmp_path/'receipts').exists()


def test_default_idle_guard_still_rejects_occupied_card(monkeypatch):
    from tools import rq3_gpu
    monkeypatch.setenv('RQ3_GPU_INDEX', '2')
    monkeypatch.setenv('RQ3_GPU_UUID', 'GPU-authorized')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-authorized')
    monkeypatch.delenv('RQ3_EXTERNAL_GPU_LEASE', raising=False)
    monkeypatch.delenv('RQ3_COHORT_LEASE', raising=False)
    def occupied(index):
        raise RuntimeError('occupied')
    monkeypatch.setattr(rq3_gpu, 'idle_gpu_uuid', occupied)
    with pytest.raises(RuntimeError, match='occupied'):
        rq3_gpu.guard_before_cuda()


def test_cohort_cannot_be_combined_with_external_sharing(monkeypatch):
    from tools.rq3_gpu import guard_before_cuda
    monkeypatch.setenv('RQ3_GPU_INDEX', '2')
    monkeypatch.setenv('RQ3_EXTERNAL_GPU_LEASE', 'lease')
    monkeypatch.setenv('RQ3_COHORT_LEASE', 'cohort')
    with pytest.raises(RuntimeError, match='cannot be combined'):
        guard_before_cuda()
