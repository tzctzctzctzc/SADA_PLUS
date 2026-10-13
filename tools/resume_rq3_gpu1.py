"""仅GPU1接续TimeLens2；真实Prototype对照通过后选择性双学生并行。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from tools.rq3_colocation import write_json
from tools.rq3_pipeline_handoff import remaining_rows
from tools.rq2_dense_protocol import sha256_file
from tools.run_rq3_matrix import baseline_root, completed_baseline
from tools.run_rq3_trusted_monitor import audit_complete
from tools.run_rq3_temporal_pair import managed_environment, RECOVERY, RECOVERY_SHA
from tools.score_rq3_vlm_candidates import check_code

CODE = Path(__file__).resolve().parents[1]
DATA = Path('/vepfs/vlms/SADA_PLUS')
TEACHER = 'timelens2_8b'


def split_remaining(completed):
    rows = remaining_rows(TEACHER, completed)
    first = [r for r in rows if r['shift'] == 'S1']
    if first != [dict(teacher=TEACHER, strategy='prototype', shift='S1')]:
        raise ValueError('expected only S1 Prototype remaining at this recovery boundary')
    return first, [r for r in rows if r['shift'] != 'S1']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--timeple-reference', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    args = p.parse_args()
    args.root = args.root.resolve(); args.previous = args.previous.resolve()
    args.timeple_reference = args.timeple_reference.resolve()
    check_code(args.expected_head)
    import fcntl
    with (DATA / '.rq3_gpu1_resume_1013.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for proc in Path('/proc').glob('[0-9]*/cmdline'):
            try: command = proc.read_bytes().replace(b'\0', b' ').decode()
            except (FileNotFoundError, PermissionError): continue
            if 'tools.run_rq3_trusted_queue' in command and TEACHER in command:
                raise RuntimeError('an existing TimeLens2 queue is active')
        args.root.mkdir(exist_ok=False)
        os.chdir(DATA)
        child = None
        state = dict(pid=os.getpid(), gpu=1, code_commit=args.expected_head,
                     phase='audit_previous', completed=[])
        def save(**values):
            state.update(values); write_json(args.root / 'state.json', state)
        def stop(*unused): raise InterruptedError('single GPU supervisor stopped')
        signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
        def run(command, name):
            nonlocal child
            check_code(args.expected_head)
            with (args.root / (name + '.log')).open('x') as output:
                child = subprocess.Popen([sys.executable, '-u', '-m', *command], cwd=CODE,
                    env=managed_environment(0), stdout=output, stderr=subprocess.STDOUT,
                    start_new_session=True)
                save(phase=name, child=child.pid)
                if child.wait(): raise RuntimeError(name + ' failed; preserve artifacts')
                child = None
        def queue(rows, name, prepared, pair=False):
            plan = args.root / (name + '.json')
            write_json(plan, dict(schema='rq3-expanded-fresh-matrix-v1', seed=6, fresh=True, entries=rows))
            output = args.root / (args.root.name + '_' + name)
            command = ['tools.run_rq3_trusted_queue', '--entry-plan', str(plan),
                '--output-root', str(output), '--expected-head', args.expected_head,
                '--gpu', '1', '--queue-slot', '0', '--batch-size', '2',
                '--early-stop-patience', '15', '--refresh-teacher-caches', '--pipeline-cpu',
                '--prepared-root', str(prepared), '--recovery-root', str(RECOVERY),
                '--recovery-manifest-sha256', RECOVERY_SHA]
            if pair: command += ['--pair-students', '--pair-strategies', 'prototype']
            run(command, name)
            return output
        save()
        try:
            baselines = {'S'+str(i): completed_baseline(baseline_root(Path('/vepfs/vlms'), 'S'+str(i)),
                         'S'+str(i)) for i in range(1,7)}
            completed = {}
            def collect(root):
                for path in root.rglob('complete.json'):
                    key = audit_complete(path, baselines)
                    if key in completed: raise ValueError('duplicate completed identity')
                    completed[key] = dict(path=str(path), sha256=sha256_file(path))
                save(completed=list(completed.values()))
            collect(args.previous)
            first, rest = split_remaining(completed)
            first_root = queue(first, 'timelens2_S1_prototype', args.previous)
            collect(first_root)
            reference = Path(completed[(TEACHER, 'prototype', 'S1')]['path']).with_name('preflight.json')
            # 两教师真实配置，完整训练结果不参与吞吐决策。
            audit_complete(args.timeple_reference.with_name('complete.json'), baselines)
            refs = args.root / 'benchmark_references.json'
            write_json(refs, [str(reference), str(args.timeple_reference)])
            run(['tools.benchmark_rq3_temporal_concurrency', 'parent', '--root', str(args.root/'benchmark'),
                 '--references-json', str(refs), '--expected-head', args.expected_head, '--gpu', '1'], 'benchmark')
            report = json.loads((args.root/'benchmark/result.json').read_text())
            enable = bool(report['enable_candidate'])
            save(prototype_pair_enabled=enable, benchmark_sha256=sha256_file(args.root/'benchmark/result.json'))
            if remaining_rows(TEACHER, completed) != rest: raise ValueError('remaining identity changed')
            rest_root = queue(rest, 'timelens2_remaining15', first_root, pair=enable)
            collect(rest_root)
            if remaining_rows(TEACHER, completed): raise ValueError('incomplete TimeLens2 matrix')
            write_json(args.root/'final_audit.json', dict(status='passed', count=18,
                results=list(completed.values()), gpu=1, prototype_pair_enabled=enable))
            save(phase='complete', child=None)
        except BaseException as error:
            save(phase='failed', error=str(error))
            raise
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait()


if __name__ == '__main__': main()
