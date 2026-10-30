"""所有有余量的卡均可共享；沿用现有物理卡锁，不碰外部进程。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time

from tools.baseline_protocol import read, save, sha


def snapshot(index):
    row = subprocess.check_output(['nvidia-smi', '-i', str(index),
        '--query-gpu=uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    uuid, total, used, util = [x.strip() for x in row]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                         '--format=csv,noheader,nounits'], text=True)
    pids = [int(parts[1]) for line in processes.splitlines()
            if len(parts := [x.strip() for x in line.split(',')]) == 3 and parts[0] == uuid]
    return dict(index=int(index), uuid=uuid, total_mib=int(total), used_mib=int(used),
                free_mib=int(total)-int(used), utilization=int(util), external_pids=pids)


@contextmanager
def slot(index, data, root, minimum_free_mib, commit):
    import fcntl
    lock = (Path(data)/('.rq2_gpu%d.lock' % int(index))).open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = snapshot(index)
        if observed['free_mib'] < minimum_free_mib:
            raise BlockingIOError('显存余量不足')
        lease = Path(root)/'gpu_admissions'/('%d_%d.json' % (os.getpid(), time.time_ns()))
        save(lease, dict(owner_pid=os.getpid(), lock=str(Path(lock.name).absolute()),
                         observed=observed, minimum_free_mib=minimum_free_mib, commit=commit))
        yield dict(index=index, uuid=observed['uuid'], lease=str(lease), lease_sha256=sha(lease))
    finally:
        lock.close()


def first_cuda_guard():
    import fcntl
    path = Path(os.environ['SADA_BASELINE_LEASE'])
    if sha(path) != os.environ['SADA_BASELINE_LEASE_SHA256']:
        raise ValueError('GPU lease SHA不符')
    lease = read(path)
    if lease['owner_pid'] != os.getppid():
        raise ValueError('GPU锁拥有者不是直接父进程')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != lease['observed']['uuid']:
        raise ValueError('GPU UUID不符')
    with Path(lease['lock']).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise ValueError('父进程未持有物理卡锁')
    while True:
        now = snapshot(lease['observed']['index'])
        if now['uuid'] != lease['observed']['uuid'] or os.getppid() != lease['owner_pid']:
            raise ValueError('GPU/父进程改变')
        if now['free_mib'] >= lease['minimum_free_mib']:
            if 'research_share' in lease:
                from tools.research_gpu_share import first_cuda_membership
                sharing=first_cuda_membership(lease)
                # 研究同伴核验可能等待；离开等待后再核显存，不复用旧快照。
                now=snapshot(lease['observed']['index'])
                if now['uuid']!=lease['observed']['uuid'] or now['free_mib']<lease['minimum_free_mib']:continue
            break
        print('WAITING_BEFORE_CUDA', now['free_mib'], flush=True)
        time.sleep(10)
    proof = path.with_name(path.stem+'_guard_%d.json' % os.getpid())
    record=dict(lease=str(path),lease_sha256=sha(path),observed=now,child_pid=os.getpid(),owner_pid=lease['owner_pid'])
    if 'research_share' in lease:record['research_share']=sharing
    save(proof,record)
    print('FIRST_CUDA_GUARD_PASS', now['index'], now['free_mib'], flush=True)
    return str(proof)
