"""修订RQ3：任意卡连续空闲30分钟后接续，最多两个互斥工作队列。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_confidence import load_calibration
from tools.rq3_gpu import gpu_indices, idle_gpu_uuid
from tools.rq3_protocol import REVISION, load_cache, proposal_evidence
from tools.run_rq3_matrix import CODE_ROOT, baseline_root, completed_baseline, checked_completed_entry, load_entry_plan
from tools.run_rq3_trusted_queue import DATA, ordered_rows
from tools.score_rq3_vlm_candidates import check_code


def row_key(row):
    return row['teacher'], row['strategy'], row['shift']


def partition_rows(plan, completed):
    rows = [row for row in ordered_rows(plan) if row_key(row) not in completed]
    return [rows[0::2], rows[1::2]]


def observe_idle(timers, observations, now, required_seconds):
    """忙、卡锁不可用或UUID变化即清零，不沿用历史快照。"""
    for index in list(timers):
        if index not in observations or timers[index]['uuid'] != observations[index]:
            del timers[index]
    for index, uuid in observations.items():
        timers.setdefault(index, dict(uuid=uuid, since=now))
    return [index for index in sorted(timers)
            if now - timers[index]['since'] >= required_seconds]


def eligible_cards(timers, now, idle_seconds, immediate_first_start):
    return [index for index in sorted(timers)
            if immediate_first_start or now - timers[index]['since'] >= idle_seconds]


def probe_idle(index):
    import fcntl
    with (DATA / ('.rq2_gpu%d.lock' % index)).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return idle_gpu_uuid(index)
        except (BlockingIOError, RuntimeError):
            return None


def audit_complete(path, baselines):
    record = json.loads(path.read_text())
    cfg = json.loads((path.parent / 'preflight.json').read_text())
    aux = cfg['rq3_auxiliary']
    hard = aux['strategy'] == 'hard' and aux['schema'] == 'rq3-three-strategies-v2'
    if (not hard and (aux['schema'] != REVISION or aux['participation_fraction'] != .2)) or aux['target_gt_used']:
        raise ValueError('completed result belongs to another RQ3 protocol')
    payload, _ = load_cache(aux['cache_path'], record['teacher'], record['shift'])
    if not hard:
        manifest = json.loads(Path(aux['manifest_path']).read_text())
        proposal_evidence(payload, aux['manifest_path'], aux['selected_coordinates'])
        load_calibration(Path(aux['calibration_path']), baselines[record['shift']][0], payload, manifest, record['strategy'])
    args = SimpleNamespace(data_root=DATA, baseline_parent=Path('/vepfs/vlms'))
    checked_completed_entry(path, args, baselines,
                            {record['teacher'] + '/' + record['shift'] + '.json': sha256_file(aux['cache_path'])})
    return row_key(record)


def collect_completed(roots, baselines, completed):
    for root in roots:
        for path in sorted(root.rglob('complete.json')):
            record = json.loads(path.read_text())
            key = row_key(record)
            if key in completed:
                if completed[key] != str(path):
                    raise ValueError('duplicate complete experiment: ' + str(key))
                continue
            audit_complete(path, baselines)
            completed[key] = str(path)
            print('RQ3_MONITOR_COMPLETE_AUDITED', key, str(path), flush=True)


def gpu_start_race(root):
    state_path = root / 'state.json'
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text())
    if state.get('status') != 'failed' or not all(key in state for key in ('shift', 'teacher', 'strategy')):
        return False
    name = '{}_{}_{}'.format(state['shift'], state['teacher'], state['strategy'])
    formal = root / (root.name + '_formal_' + name)
    log = formal / state['teacher'] / state['strategy'] / (state['shift'] + '_seed6') / 'logs/uda.train.log'
    if not log.is_file():
        return False
    text = log.read_text(errors='replace')
    # 只重试明确在首个CUDA创建之前被空闲guard拒绝的启动；训练/评估错误保留并停止。
    return ('guard_before_cuda' in text and 'selected physical GPU is occupied' in text
            and 'RQ3_PRE_CUDA_IDLE_CHECK_PASS' not in text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry-plan', type=Path, required=True)
    parser.add_argument('--completed-root', type=Path, action='append', default=[])
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--recovery-root', type=Path, required=True)
    parser.add_argument('--recovery-manifest-sha256', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--idle-minutes', type=float, default=30)
    parser.add_argument('--immediate-gpu', action='store_true',
                        help='用户明确授权本次首次直接启动；启动冲突后的重试仍重新观察30分钟')
    args = parser.parse_args()
    if args.idle_minutes < 30:
        parser.error('continuous idle time must be at least 30 minutes')
    for key in ('entry_plan', 'prepared_root', 'output_root', 'recovery_root'):
        setattr(args, key, getattr(args, key).resolve())
    args.completed_root = [root.resolve() for root in args.completed_root]
    check_code(args.expected_head)
    load_entry_plan(args.entry_plan)
    if sha256_file(args.recovery_root / 'manifest.json') != args.recovery_manifest_sha256:
        raise ValueError('frame recovery manifest differs')
    plan = json.loads(args.entry_plan.read_text())
    children, streams, attempts = {}, {}, []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    import fcntl
    with (DATA / '.rq3_idle_monitor.lock').open('a') as monitor_lock:
        fcntl.flock(monitor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output_root.mkdir(parents=True, exist_ok=False)
        state = dict(pid=os.getpid(), status='cpu-audit', expected_head=args.expected_head,
                     idle_minutes=args.idle_minutes, immediate_first_start=args.immediate_gpu,
                     max_gpus=2, workers={}, gpu_timers={})

        def save(**values):
            state.update(values)
            tmp = args.output_root / 'monitor_state.tmp'
            tmp.write_text(json.dumps(state, sort_keys=True))
            tmp.replace(args.output_root / 'monitor_state.json')

        save()
        try:
            os.chdir(DATA)
            baselines = {shift: completed_baseline(baseline_root(Path('/vepfs/vlms'), shift), shift)
                         for shift in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6')}
            completed = {}
            collect_completed(args.completed_root, baselines, completed)
            partitions = partition_rows(plan, completed)
            state['completed_before_start'] = len(completed)
            state['remaining_before_start'] = sum(map(len, partitions))
            workers = [dict(status='waiting' if rows else 'complete', attempt=0, gpu=None, rows=rows)
                       for rows in partitions]
            timers = {}
            print('RQ3_IDLE_OBSERVATION_STARTED idle_minutes=%g max_gpus=2' % args.idle_minutes, flush=True)
            while not stopping:
                check_code(args.expected_head)
                for slot, child in list(children.items()):
                    code = child.poll()
                    if code is None:
                        continue
                    worker = workers[slot]
                    streams.pop(slot).close()
                    del children[slot]
                    root = Path(worker['output_root'])
                    collect_completed([root], baselines, completed)
                    pending = [row for row in worker['rows'] if row_key(row) not in completed]
                    worker.update(gpu=None, exit_code=code)
                    if code == 0 and not pending:
                        worker['status'] = 'complete'
                        print('RQ3_MONITOR_WORKER_COMPLETE', slot, flush=True)
                    elif gpu_start_race(root):
                        worker.update(status='waiting', rows=pending)
                        timers.clear()
                        print('RQ3_MONITOR_GPU_RACE_REWAIT', slot, str(root), flush=True)
                    else:
                        worker['status'] = 'failed'
                        print('RQ3_MONITOR_TECHNICAL_FAILURE', slot, str(root), flush=True)
                active = {worker['gpu'] for worker in workers if worker['status'] == 'running'}
                observations = {}
                for index in gpu_indices('auto'):
                    if index not in active:
                        uuid = probe_idle(index)
                        if uuid:
                            observations[index] = uuid
                now = time.monotonic()
                observe_idle(timers, observations, now, args.idle_minutes * 60)
                for slot, worker in enumerate(workers):
                    ready = eligible_cards(timers, now, args.idle_minutes * 60,
                                           args.immediate_gpu and worker['attempt'] == 0)
                    if worker['status'] != 'waiting' or not ready:
                        continue
                    gpu = ready.pop(0)
                    if probe_idle(gpu) != timers[gpu]['uuid']:
                        timers.pop(gpu, None)
                        continue
                    worker['attempt'] += 1
                    root = args.output_root / (args.output_root.name + '_slot%d_attempt%d' % (slot, worker['attempt']))
                    subplan = args.output_root / ('slot%d_attempt%d_plan.json' % (slot, worker['attempt']))
                    with subplan.open('x') as stream:
                        json.dump(dict(plan, entries=worker['rows']), stream, sort_keys=True)
                    load_entry_plan(subplan)
                    log = args.output_root / ('slot%d_attempt%d.log' % (slot, worker['attempt']))
                    stream = log.open('x')
                    command = [sys.executable, '-B', '-m', 'tools.run_rq3_trusted_queue',
                               '--entry-plan', str(subplan), '--output-root', str(root),
                               '--expected-head', args.expected_head, '--gpu', str(gpu), '--queue-slot', str(slot),
                               '--prepared-root', str(args.prepared_root), '--recovery-root', str(args.recovery_root),
                               '--recovery-manifest-sha256', args.recovery_manifest_sha256]
                    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                               CUDA_VISIBLE_DEVICES='', RQ3_QUEUE_SLOT=str(slot), RQ3_MONITOR_PID=str(os.getpid()),
                               PYTHONPATH=str(CODE_ROOT) + ':/vepfs/vlms/cache/rq3_preparation_deps_0928')
                    child = subprocess.Popen(command, cwd=CODE_ROOT, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    children[slot], streams[slot] = child, stream
                    attempts.append(str(root))
                    worker.update(status='running', gpu=gpu, pid=child.pid, output_root=str(root), log=str(log))
                    timers.pop(gpu, None)
                    print('RQ3_MONITOR_WORKER_STARTED slot=%d gpu=%d pid=%d' % (slot, gpu, child.pid), flush=True)
                finished = all(worker['status'] in ('complete', 'failed') for worker in workers)
                save(status=('complete' if all(w['status'] == 'complete' for w in workers) else 'failed')
                     if finished else 'running' if children else 'waiting',
                     workers={str(i): w for i, w in enumerate(workers)},
                     gpu_timers={str(i): dict(uuid=timer['uuid'], idle_seconds=now - timer['since'])
                                 for i, timer in timers.items()},
                     completed=len(completed), completed_records={'/'.join(key): path for key, path in completed.items()},
                     attempts=attempts, updated_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
                if finished:
                    break
                time.sleep(30)
            if stopping:
                save(status='stopped')
        except BaseException as error:
            save(status='failed', error=str(error))
            raise
        finally:
            stop(None, None)
            for child in children.values():
                child.wait()
            for stream in streams.values():
                stream.close()


if __name__ == '__main__':
    main()
