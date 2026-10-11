"""有限六项：等待RQ4全终审、六shift真实验收、GPU0/1训练和独立全审。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq3_full_launch import (SHIFTS, read, save, bind, create_launch_proof,
                                   validate_smoke)
from tools.rq3_full_audit import audit_complete
from tools.run_rq2_matched_control import check_code

CODE = Path(__file__).resolve().parents[1]


def stop_children(active):
    for child, _, _ in active.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for child, stream, _ in active.values():
        if child.poll() is None:
            child.wait(timeout=60)
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('output', 'references', 'rq4-root'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--rq4-head', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    args = parser.parse_args()
    check_code(args.expected_head)
    # 保存工件路径仍相对调用目录；冻结配置中的data/...相对原数据仓库。
    # 前序基线审计与每项完成审计均需同一数据cwd，不能只在学生子进程切换。
    for key in ('output', 'references', 'rq4_root', 'data_root'):
        setattr(args, key, getattr(args, key).resolve())
    os.chdir(args.data_root)
    references = read(args.references)
    if set(references) != set(SHIFTS):
        raise ValueError('exact six references required')
    lock = open('/vepfs/vlms/SADA_PLUS/.rq3_qwen100_matrix.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output/'references.json', references)
    active, completed, reports = {}, [], {}
    pending = list(SHIFTS)
    phase = 'waiting_rq4_final'
    state = dict(pid=os.getpid(), code_commit=args.expected_head, completed=completed)
    proof_path, acceptance_path = args.output/'launch_proof.json', args.output/'acceptance.json'

    def status():
        state.update(phase=phase, pending=pending,
                     active={g: dict(pid=v[0].pid, shift=v[2]) for g, v in active.items()})
        temporary = args.output/'state.json.tmp'
        temporary.write_text(json.dumps(state, indent=2))
        temporary.replace(args.output/'state.json')

    def interrupted(*_):
        raise RuntimeError('explicit stop; preserve artifacts and do not auto-restart')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        while not (args.rq4_root/'final_audit.json').is_file():
            rq4_state = read(args.rq4_root/'state.json')
            if rq4_state.get('error'):
                raise RuntimeError('RQ4 failed; no Qwen launch: '+rq4_state['error'])
            status(); time.sleep(30)
        phase = 'predecessor_and_reference_audit'; status()
        save(proof_path, create_launch_proof(args.rq4_root/'final_audit.json',
                                            args.rq4_head, references, args.expected_head))
        phase = 'acceptance'
        while True:
            for gpu, (child, stream, shift) in list(active.items()):
                rc = child.poll()
                if rc is None:
                    continue
                stream.close(); del active[gpu]
                if rc:
                    raise RuntimeError(f'{phase} {shift} failed, exit={rc}')
                output = args.output/phase/shift
                if phase == 'acceptance':
                    validate_smoke(read(output/'smoke.json'), read(output/'preflight.json'))
                    reports[shift] = bind(output/'smoke.json')
                else:
                    audit_complete(output)
                    completed.append(dict(shift=shift, root=str(output)))
                print('QWEN100_ITEM_DONE', phase, shift, flush=True)
            if not pending and not active:
                if phase == 'acceptance':
                    save(acceptance_path, dict(status='passed', code_commit=args.expected_head,
                                              launch_proof=bind(proof_path), reports=reports))
                    phase = 'formal'; pending = list(SHIFTS)
                    print('QWEN100_SIX_REAL_ACCEPTANCE_PASS', flush=True)
                else:
                    if len(completed) != 6 or {x['shift'] for x in completed} != set(SHIFTS):
                        raise ValueError('incomplete six-shift identity set')
                    for row in completed:
                        audit_complete(row['root'])
                    save(args.output/'final_audit.json', dict(status='passed', count=6,
                         code_commit=args.expected_head, entries=completed))
                    phase = 'complete'; status()
                    print('QWEN100_ALL_SIX_FINAL_AUDIT_PASS', flush=True)
                    break
            for gpu in ('0', '1'):
                if gpu in active or not pending:
                    continue
                shift = pending.pop(0)
                output = args.output/phase/shift
                output.parent.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, '-B', '-m', 'tools.run_rq3_full_entry',
                           '--reference', references[shift], '--output', str(output),
                           '--expected-head', args.expected_head, '--gpu', gpu,
                           '--data-root', str(args.data_root),
                           '--launch-proof', str(proof_path)]
                command += ['--smoke-only'] if phase == 'acceptance' else ['--acceptance', str(acceptance_path)]
                stream = (output.parent/(shift+'.log')).open('x')
                child = subprocess.Popen(command, cwd=CODE, env=dict(os.environ,
                    CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(CODE)),
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = (child, stream, shift)
                print('QWEN100_STARTED', phase, shift, 'gpu', gpu, 'pid', child.pid, flush=True)
            status(); time.sleep(10)
    except BaseException as error:
        state['error'] = str(error)
        stop_children(active)
        active.clear(); status()
        raise


if __name__ == '__main__':
    main()
