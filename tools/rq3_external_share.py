"""用户明确授权的GPU1/2外部共享；只改变准入，不改变训练或随机数设置。"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

SCHEMA = 'rq3-authorized-external-gpu-share-v1'
MIN_FREE_MIB = 16384
AUTHORIZED_GPUS = (1, 2)


def snapshot(index):
    fields = subprocess.check_output([
        'nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid, total, used, utilization = [v.strip() for v in fields]
    rows = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True)
    # NVML可能返回宿主机PID；只记录，不用容器/proc推断外部任务身份。
    processes = []
    for line in rows.splitlines():
        parts = [v.strip() for v in line.split(',')]
        if len(parts) == 3 and parts[0] == uuid:
            processes.append(dict(pid=int(parts[1]), memory_mib=int(parts[2])))
    return dict(index=int(index), uuid=uuid, total_mib=int(total), used_mib=int(used),
                utilization=int(utilization), processes=sorted(processes, key=lambda p:p['pid']))


def validate_admission(current, admitted=None):
    if current['index'] not in AUTHORIZED_GPUS:
        raise ValueError('external sharing is authorized only on physical GPU1/2')
    if current['total_mib'] - current['used_mib'] < MIN_FREE_MIB:
        raise RuntimeError('GPU sharing requires at least 16GiB free memory')
    if admitted is not None:
        if (current['index'], current['uuid']) != (admitted['index'], admitted['uuid']):
            raise RuntimeError('shared GPU identity changed')
        old = {p['pid'] for p in admitted['processes']}
        if not {p['pid'] for p in current['processes']} <= old:
            raise RuntimeError('a new external CUDA process appeared during CPU loading')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def external_gpu_slot(index, lock_root, receipts, code_commit, poll_seconds=30):
    import fcntl
    index = int(index)
    if index not in AUTHORIZED_GPUS:
        raise ValueError('external sharing is authorized only on GPU1/2')
    lock_root, receipts = Path(lock_root), Path(receipts)
    receipts.mkdir(exist_ok=True)
    announced = False
    while True:
        capacity = card_lock = None
        try:
            for name in ('.rq3_single_gpu.lock', '.rq3_second_gpu.lock'):
                handle = (lock_root/name).open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    capacity = handle
                    break
                except BlockingIOError:
                    handle.close()
            if capacity is None:
                raise BlockingIOError('RQ3 capacity occupied')
            card_lock = (lock_root/('.rq2_gpu%d.lock' % index)).open('a')
            fcntl.flock(card_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            admitted = snapshot(index)
            validate_admission(admitted)
        except (BlockingIOError, RuntimeError):
            if card_lock is not None:
                card_lock.close()
            if capacity is not None:
                capacity.close()
            if not announced:
                print('RQ3_WAITING_FOR_SHARED_GPU%d min_free_mib=16384' % index, flush=True)
                announced = True
            time.sleep(poll_seconds)
            continue
        except BaseException:
            if card_lock is not None:
                card_lock.close()
            if capacity is not None:
                capacity.close()
            raise
        try:
            path = receipts/('%d_%d.json' % (os.getpid(), time.time_ns()))
            lease = dict(schema=SCHEMA, owner_pid=os.getpid(), code_commit=code_commit,
                         minimum_free_mib=MIN_FREE_MIB, admitted=admitted,
                         locks=[str(Path(capacity.name).absolute()), str(Path(card_lock.name).absolute())])
            with path.open('x') as stream:
                json.dump(lease, stream, sort_keys=True)
            print('RQ3_SHARED_GPU_ADMITTED index=%d free_mib=%d external_pids=%s' % (
                index, admitted['total_mib']-admitted['used_mib'],
                [p['pid'] for p in admitted['processes']]), flush=True)
            yield dict(index=index, uuid=admitted['uuid'], external_share_lease=str(path),
                       external_share_sha256=digest(path))
        finally:
            card_lock.close()
            capacity.close()
        return


def shared_guard():
    import fcntl
    path = Path(os.environ['RQ3_EXTERNAL_GPU_LEASE'])
    if digest(path) != os.environ['RQ3_EXTERNAL_GPU_LEASE_SHA256']:
        raise RuntimeError('external sharing lease changed')
    lease = json.loads(path.read_text())
    if lease['schema'] != SCHEMA or lease['minimum_free_mib'] != MIN_FREE_MIB:
        raise ValueError('unrecognized GPU sharing policy')
    if lease['owner_pid'] != os.getppid():
        raise RuntimeError('GPU sharing controller is not the direct parent')
    admitted = lease['admitted']
    if (os.environ.get('RQ3_GPU_INDEX') != str(admitted['index']) or
        os.environ.get('RQ3_GPU_UUID') != admitted['uuid'] or
        os.environ.get('CUDA_VISIBLE_DEVICES') != admitted['uuid']):
        raise RuntimeError('CUDA visibility differs from authorized GPU')
    for lock in lease['locks']:
        with Path(lock).open('a') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            raise RuntimeError('GPU sharing controller released its lock')
    validate_admission(snapshot(admitted['index']), admitted)
    print('RQ3_PRE_CUDA_EXTERNAL_SHARE_CHECK_PASS index=%d min_free_mib=16384' % admitted['index'], flush=True)


def audit_receipt(card, code_commit):
    if 'external_share_lease' not in card:
        return
    path = Path(card['external_share_lease'])
    if digest(path) != card['external_share_sha256']:
        raise ValueError('GPU admission receipt changed')
    lease = json.loads(path.read_text())
    if lease['schema'] != SCHEMA or lease['code_commit'] != code_commit:
        raise ValueError('GPU sharing code identity changed')
    validate_admission(lease['admitted'])
    if card['index'] != lease['admitted']['index'] or card['uuid'] != lease['admitted']['uuid']:
        raise ValueError('GPU admission identity differs from complete')
