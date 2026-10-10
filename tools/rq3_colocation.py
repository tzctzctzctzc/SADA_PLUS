"""每卡最多两个独立学生进程；父进程持物理/容量锁，评估仍串行。"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True))
    temporary.replace(path)


def process_start(pid):
    # comm字段可能包含空格，不能直接对整个stat用split。
    return Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()[19]


def current_lease():
    path = Path(os.environ['RQ3_COHORT_LEASE'])
    lease = json.loads(path.read_text())
    if process_start(lease['owner_pid']) != lease['owner_start']:
        raise RuntimeError('colocation owner is no longer alive')
    member = os.environ['RQ3_COHORT_MEMBER']
    if member not in lease['members'] or os.getpgrp() != lease['members'][member]:
        raise RuntimeError('process does not belong to this colocation cohort')
    if len(lease['members']) != 2:
        raise ValueError('a shared cohort must contain exactly two students')
    return path.parent, lease, member


def compute_pids(uuid):
    import subprocess
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    return {int(line.split(',')[1]) for line in rows.splitlines()
            if line.split(',')[0].strip() == uuid}


def check_admission(previous, current, after, maximum):
    if current != previous:
        raise RuntimeError('unknown GPU process before CUDA admission')
    added = after - previous
    if len(added) != 1 or not previous.issubset(after) or len(after) > maximum:
        raise RuntimeError('cannot uniquely identify the admitted CUDA process')
    return next(iter(added))


def owners(root, live_pids):
    # 已退出进程的记录不会给新阶段授权。NVML可短暂保留刚退出的上下文。
    return {json.loads(p.read_text())['host_pid'] for p in root.glob('cuda_*.json')} & live_pids


def shared_guard():
    import fcntl
    import torch
    root, lease, member = current_lease()
    selected = lease['selected']
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != selected['uuid'] or
            os.environ.get('RQ3_GPU_INDEX') != str(selected['index'])):
        raise RuntimeError('colocation CUDA visibility differs from physical lease')
    with (root/'admission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = compute_pids(selected['uuid'])
        expected = owners(root, before)
        if before != expected:
            raise RuntimeError('external process entered reserved GPU')
        stage = os.environ['RQ3_COHORT_STAGE']
        maximum = 2 if stage == 'train' else 1
        if len(before) >= maximum:
            raise RuntimeError('colocation stage capacity is occupied')
        if not before:
            from tools.rq3_gpu import idle_gpu_uuid
            if idle_gpu_uuid(selected['index']) != selected['uuid']:
                raise RuntimeError('colocation physical GPU UUID changed')
        # 显式创建当前上下文后登记NVML宿主PID；不使用容器getpgid猜宿主身份。
        torch.cuda.init()
        probe = torch.ones(1, device='cuda')
        torch.cuda.synchronize()
        host_pid = check_admission(expected, before, compute_pids(selected['uuid']), maximum)
        write_json(root/('cuda_%s_%s.json' % (member, stage)),
                   dict(host_pid=host_pid, pid=os.getpid(), start=process_start(os.getpid())))
        del probe
    print('RQ3_SHARED_CUDA_GUARD_PASS member=%s stage=%s' % (member, stage), flush=True)
    # 父队列意外消失时，不能遗留失去物理锁保护的CUDA子任务。
    import signal
    import threading
    def watch_parent():
        while True:
            time.sleep(1)
            try:
                current_lease()
            except (OSError, ValueError, KeyError, RuntimeError):
                os.killpg(os.getpgrp(), signal.SIGTERM)
                return
    threading.Thread(target=watch_parent, daemon=True).start()


@contextmanager
def shared_stage(request, stage):
    import fcntl
    root, lease, member = current_lease()
    if str(request) != str(lease['selected']['index']):
        raise ValueError('shared stage requested another physical GPU')
    old = os.environ.get('RQ3_COHORT_STAGE')
    os.environ['RQ3_COHORT_STAGE'] = stage
    try:
        if stage == 'train':
            yield lease['selected']
            (root/('trained_' + member)).touch()
        else:
            # 两个学生均退出CUDA训练后，才允许串行评估，避免训练/评估互抢。
            while not all((root/('trained_' + key)).exists() for key in lease['members']):
                current_lease()
                time.sleep(1)
            with (root/'evaluation.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                from tools.rq3_gpu import idle_gpu_uuid
                for attempt in range(31):
                    try:
                        if idle_gpu_uuid(int(request)) != lease['selected']['uuid']:
                            raise RuntimeError('physical GPU UUID changed')
                        break
                    except RuntimeError:
                        if attempt == 30:
                            raise
                        time.sleep(1)
                yield lease['selected']
    finally:
        if old is None:
            os.environ.pop('RQ3_COHORT_STAGE', None)
        else:
            os.environ['RQ3_COHORT_STAGE'] = old


def choose_student_groups(jobs):
    """仅同策略Hard/Prototype配对；Verify保持单任务。不读任何性能结果。"""
    result = []
    for strategy in ('hard', 'verify', 'prototype'):
        selected = [job for job in jobs if job['strategy'] == strategy]
        width = 1 if strategy == 'verify' else 2
        result.extend(selected[i:i+width] for i in range(0, len(selected), width))
    if sum(map(len, result)) != len(jobs):
        raise ValueError('unknown strategy in student jobs')
    identities = [job['identity'] for group in result for job in group]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate student entry')
    return result


def cohort_entry():
    """新session先等父进程登记身份，然后运行原matrix，不修改训练参数。"""
    import runpy
    import sys
    lease = Path(os.environ['RQ3_COHORT_LEASE'])
    deadline = time.monotonic() + 30
    while not lease.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('colocation parent did not publish lease')
        time.sleep(.05)
    current_lease()
    sys.argv = ['tools.run_rq3_matrix', *sys.argv[1:]]
    runpy.run_module('tools.run_rq3_matrix', run_name='__main__')


if __name__ == '__main__':
    cohort_entry()
