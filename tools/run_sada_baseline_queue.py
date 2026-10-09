"""有限的统一基线队列：只跑SlowFast六shift的SO/UDA，不扩展VLM。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from tools.run_rq2_matched_control import CODE_ROOT, check_code


SHARDS = {2: ('S1', 'S3', 'S5'), 3: ('S2', 'S4', 'S6')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, choices=(2, 3), required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--audit-only', action='store_true')
    p.add_argument('--seed', type=int, choices=(6, 123), default=6)
    args = p.parse_args()
    check_code(args.expected_head)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    child = None
    stopping = False
    state = dict(schema='sada-unified-baseline-queue-v1', gpu=args.gpu, seed=args.seed,
                 shifts=list(SHARDS[args.gpu]), code_commit=args.expected_head,
                 pid=os.getpid(), status='preparing', completed=[])

    def save():
        (root / 'state.json').write_text(json.dumps(state, indent=2, sort_keys=True))

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # 即使换了output-root，同一卡的本任务也不能启动重复队列。
    import fcntl
    with Path('/vepfs/vlms/SADA_PLUS/.sada_unified_gpu%d.lock' % args.gpu).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            for shift in SHARDS[args.gpu]:
                if stopping:
                    break
                check_code(args.expected_head)
                command = [sys.executable, '-B', '-m', 'tools.run_rq2_matched_control',
                           '--model', 'slowfast', '--shift', shift, '--seed', str(args.seed),
                           '--gpu', str(args.gpu), '--output-root', str(root / shift),
                           '--expected-head', args.expected_head, '--unified-baseline']
                if args.audit_only:
                    command.append('--audit-only')
                with (root / (shift + '.log')).open('x') as log:
                    child = subprocess.Popen(command, cwd=CODE_ROOT, stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                    if stopping:
                        stop(signal.SIGTERM, None)
                    state.update(status='cpu_audit' if args.audit_only else 'running',
                                 current_shift=shift, child_pid=child.pid)
                    save()
                    print('BASELINE_CHILD', shift, child.pid, flush=True)
                    code = child.wait()
                if stopping:
                    break
                if code:
                    raise RuntimeError('%s exited %d; partial outputs retained' % (shift, code))
                if not args.audit_only and not (root / shift / 'complete.json').is_file():
                    raise RuntimeError('missing complete result for ' + shift)
                state['completed'].append(shift)
                save()
            state['status'] = 'stopped' if stopping else 'audit_passed' if args.audit_only else 'complete'
            state.pop('child_pid', None)
            save()
        except BaseException as exc:
            state.update(status='failed', error=str(exc))
            save()
            raise


if __name__ == '__main__':
    main()
