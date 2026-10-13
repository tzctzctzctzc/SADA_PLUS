"""补齐S3 Verify评估，审计完整项，仅在GPU3接续剩余TimeLens2条目。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_pipeline_handoff import remaining_rows
from tools.run_rq3_matrix import baseline_root, completed_baseline
from tools.run_rq3_temporal_pair import managed_environment, RECOVERY, RECOVERY_SHA
from tools.run_rq3_trusted_monitor import audit_complete
from tools.score_rq3_vlm_candidates import check_code

CODE = Path(__file__).resolve().parents[1]
DATA = Path('/vepfs/vlms/SADA_PLUS')
TEACHER = 'timelens2_8b'
OLD_HEAD = '09a9127010e513db524b55425761b95d6574a29f'
OLD = Path('/vepfs/vlms/rq3_temporal36_lockfix_015590b_1012/timelens2_8b/formal18')
FIRST = Path('/vepfs/vlms/rq3_gpu1_resume_09a9127_1013/rq3_gpu1_resume_09a9127_1013_timelens2_S1_prototype')
PARTIAL = Path('/vepfs/vlms/rq3_gpu1_resume_09a9127_1013/rq3_gpu1_resume_09a9127_1013_timelens2_remaining15')
RECOVER = PARTIAL / (PARTIAL.name + '_formal_S3_timelens2_8b_verify/timelens2_8b/verify/S3_seed6')
TIMEPLE_REFERENCE = Path('/vepfs/vlms/rq3_temporal36_lockfix_015590b_1012/timeple_8b/formal18/formal18_formal_S1_timeple_8b_prototype/timeple_8b/prototype/S1_seed6/preflight.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    args.root = args.root.resolve()
    check_code(args.expected_head)
    args.root.mkdir(exist_ok=False)
    os.chdir(DATA)
    child = None
    state = dict(pid=os.getpid(), code_commit=args.expected_head, gpu=3,
                 phase='audit_existing', completed=[])

    def save(**changes):
        state.update(changes)
        target = args.root / 'state.json'
        temporary = args.root / 'state.tmp'
        temporary.write_text(json.dumps(state, sort_keys=True))
        temporary.replace(target)

    def stop(*unused):
        raise InterruptedError('GPU3 recovery stopped')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    save()
    try:
        baselines = {shift: completed_baseline(baseline_root(Path('/vepfs/vlms'), shift), shift)
                     for shift in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6')}
        completed = {}

        def collect(roots):
            for root in roots:
                for path in sorted(root.rglob('complete.json')):
                    key = audit_complete(path, baselines)
                    if key in completed and completed[key]['path'] != str(path):
                        raise ValueError('duplicate completed RQ3 identity: ' + str(key))
                    completed[key] = dict(path=str(path), sha256=sha256_file(path))
            save(completed=list(completed.values()))

        collect((OLD, FIRST, PARTIAL))
        expected = {(TEACHER, strategy, 'S1') for strategy in ('hard', 'verify', 'prototype')}
        expected |= {(TEACHER, 'hard', 'S2'), (TEACHER, 'verify', 'S2'),
                     (TEACHER, 'hard', 'S3')}
        if set(completed) != expected or RECOVER.exists() is False:
            raise ValueError('existing completed or partial identity differs')
        if (RECOVER / 'complete.json').exists():
            raise ValueError('S3 Verify unexpectedly complete')
        save(phase='recover_S3_verify_evaluation')
        recover = [sys.executable, '-u', '-m', 'tools.recover_rq3_evaluation',
                   '--root', str(RECOVER), '--expected-head', args.expected_head,
                   '--training-head', OLD_HEAD, '--gpu', '3']
        with (args.root / 'recover.log').open('x') as stream:
            child = subprocess.Popen(recover, cwd=CODE, env=managed_environment(0),
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            save(child=child.pid)
            if child.wait():
                raise RuntimeError('S3 Verify evaluation recovery failed; keep checkpoint')
        child = None
        collect((PARTIAL,))
        if (TEACHER, 'verify', 'S3') not in completed:
            raise RuntimeError('S3 Verify recovery lacks audited complete')
        remaining = remaining_rows(TEACHER, completed)
        if len(remaining) != 11:
            raise ValueError('expected exactly 11 uncompleted TimeLens2 entries')
        timelens_reference = Path(completed[(TEACHER, 'prototype', 'S1')]['path']).with_name('preflight.json')
        if not TIMEPLE_REFERENCE.is_file():
            raise FileNotFoundError(TIMEPLE_REFERENCE)
        refs = args.root / 'benchmark_references.json'
        refs.write_text(json.dumps([str(timelens_reference), str(TIMEPLE_REFERENCE)]))
        save(phase='gpu3_prototype_benchmark')
        with (args.root / 'benchmark.log').open('x') as stream:
            child = subprocess.Popen([sys.executable, '-u', '-m',
                'tools.benchmark_rq3_temporal_concurrency', 'parent', '--root', str(args.root / 'benchmark'),
                '--references-json', str(refs), '--expected-head', args.expected_head, '--gpu', '3'],
                cwd=CODE, env=managed_environment(0), stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True)
            save(child=child.pid)
            if child.wait():
                raise RuntimeError('GPU3 prototype colocation benchmark failed')
        child = None
        benchmark = json.loads((args.root / 'benchmark/result.json').read_text())
        pair = bool(benchmark['enable_candidate'])
        save(prototype_pair_enabled=pair, benchmark_sha256=sha256_file(args.root / 'benchmark/result.json'))
        plan = args.root / 'remaining11.json'
        plan.write_text(json.dumps(dict(schema='rq3-expanded-fresh-matrix-v1',
                                        seed=6, fresh=True, entries=remaining), sort_keys=True))
        queue_root = args.root / 'timelens2_remaining11'
        command = [sys.executable, '-u', '-m', 'tools.run_rq3_trusted_queue',
                   '--entry-plan', str(plan), '--output-root', str(queue_root),
                   '--expected-head', args.expected_head, '--gpu', '3',
                   '--queue-slot', '0', '--batch-size', '2', '--early-stop-patience', '15',
                   '--refresh-teacher-caches', '--pipeline-cpu', '--prepared-root', str(PARTIAL),
                   '--recovery-root', str(RECOVERY),
                   '--recovery-manifest-sha256', RECOVERY_SHA]
        if pair:
            command += ['--pair-students', '--pair-strategies', 'prototype']
        check_code(args.expected_head)
        save(phase='timelens2_remaining11', plan_sha256=sha256_file(plan))
        with (args.root / 'remaining11.log').open('x') as stream:
            child = subprocess.Popen(command, cwd=CODE, env=managed_environment(0),
                                     stdout=stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            save(child=child.pid)
            if child.wait():
                raise RuntimeError('remaining11 queue failed; preserve all artifacts')
        child = None
        collect((queue_root,))
        if remaining_rows(TEACHER, completed):
            raise ValueError('TimeLens2 matrix still has missing entries')
        (args.root / 'final_audit.json').write_text(json.dumps(dict(status='passed', count=18,
             results=list(completed.values()), gpu=3, code_commit=args.expected_head), sort_keys=True))
        save(phase='complete', child=None)
    except BaseException as error:
        save(phase='failed', error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait()


if __name__ == '__main__':
    main()
