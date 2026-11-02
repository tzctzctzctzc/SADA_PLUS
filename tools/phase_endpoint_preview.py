"""只提前评估已经合法完成20轮的模型；原训练根和工作代码保持只读。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

FILES = ('config.json', 'runtime_config.json', 'seed_proof.json',
         'mature_initialization.json', 'metrics.jsonl', 'oracle_events.jsonl',
         'termination.json', 'train_gpu.json', 'source_val_unlabeled.json',
         'target_val_unlabeled.json', 'best.pt', 'latest.pt',
         'epoch000.pt', 'epoch005.pt', 'epoch010.pt', 'epoch015.pt', 'epoch020.pt')


def parse_jobs(values):
    jobs = [tuple(v.split(':')) for v in values]
    assert jobs and len(set(jobs)) == len(jobs)
    assert all(len(j) == 2 and j[0] in ('control','gate','grid','gt') and j[1] in ('S1','S2','S3','S4','S5','S6') for j in jobs)
    return jobs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def validate_completed(entry):
    entry = Path(entry)
    for name in FILES:
        assert (entry / name).is_file(), name
    t = read(entry / 'termination.json')
    assert t['reason'] == 'fixed20_extra_budget_complete_no_target_selection'
    assert t['epoch'] == 20
    assert [v[0] for v in t['history']] == [0, 5, 10, 15, 20]
    assert t['best_epoch'] == min(t['history'], key=lambda x: (x[1], x[0]))[0]
    rows = [json.loads(s) for s in (entry / 'metrics.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in rows] == list(range(1, 21))
    for domain in ('source', 'target'):
        assert all('annotations' not in x for x in read(entry / (domain + '_val_unlabeled.json'))['database'].values())
    return t


def link_completed(source, destination):
    source, destination = Path(source), Path(destination)
    terminal = validate_completed(source)
    destination.mkdir(parents=True, exist_ok=False)
    links = []
    for name in FILES:
        old, new = source / name, destination / name
        os.link(old, new)
        assert os.path.samefile(old, new)
        links.append(dict(original=str(old), preview=str(new), size=old.stat().st_size))
    receipt = dict(original=str(source), terminal=terminal, links=links,
                   termination_sha256=sha(source / 'termination.json'),
                   train_receipt_sha256=sha(source / 'train_gpu.json'),
                   new_training=False, immutable_finished_training_only=True)
    save(destination / 'preview.json', receipt)
    return receipt


def baseline_check(entry):
    import numpy as np
    cfg = read(entry / 'config.json')
    reference = Path(cfg['training_oracle']['reference'])
    checks = {}
    for domain in ('source', 'target'):
        result = entry / ('evaluation/%s_e000.json' % domain)
        if not result.exists():
            continue  # target epoch0只在预先冻结的source-best=0时推理。
        old = read(reference / ('evaluation/%s_val.json' % domain))
        current = read(result)
        delta = current['map'] - old['map_0.1_to_0.5']
        assert abs(delta) < 1e-7, (domain, delta)
        candidates = list(reference.rglob(domain + '_val.npz'))
        same = None
        if len(candidates) == 1:
            with np.load(candidates[0], allow_pickle=False) as a, np.load(current['prediction'], allow_pickle=False) as b:
                same = set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
        checks[domain] = dict(AP_delta=delta, prediction_all_arrays_exact=same,
                              reference_evaluation_sha256=sha(reference / ('evaluation/%s_val.json' % domain)))
    assert 'source' in checks
    return checks


def main(a):
    # 所有算法、GPU guard及完整审计均导入原冻结worker源码。
    sys.path.insert(0, str(a.worker_code))
    from tools.baseline_protocol import code_identity
    from tools.phase_runner_gpu import acquire
    from tools.run_phase_endpoint_oracle import audit, SCHEMA
    identity = code_identity()
    assert identity['commit'] == a.expected_worker_commit
    a.output.mkdir(parents=True, exist_ok=False)
    jobs = parse_jobs(a.jobs)
    # 整个预声明集合全部合法完成才创建快照，避免部分创建后混入未完成训练。
    for mode, shift in jobs:
        validate_completed(a.original / 'formal' / mode / shift)
    for mode, shift in jobs:
        link_completed(a.original / 'formal' / mode / shift, a.output / 'formal' / mode / shift)
    controller_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    save(a.output / 'plan.json', dict(original=str(a.original), jobs=jobs, worker_code=identity,
        controller_commit=controller_commit, fixed_epoch=20, new_training=False,
        inference='same frozen worker; all five source epochs and fixed20/source-loss-selected target',
        later_original_prediction_comparison_required=True))
    pending = [(m, s, 'inference') for m, s in jobs]
    active, finished = {}, []
    def stop(sig, frame):
        raise SystemExit(128 + sig)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            for key, task in list(active.items()):
                rc = task['process'].poll()
                if rc is None:
                    continue
                if task.get('context'):
                    task['context'].__exit__(None, None, None)
                del active[key]
                assert rc == 0, (key, rc)
                mode, shift, stage = task['mode'], task['shift'], task['stage']
                entry = a.output / 'formal' / mode / shift
                if stage == 'inference':
                    pending.insert(0, (mode, shift, 'evaluate_cpu'))
                else:
                    artifacts = {str(p): sha(p) for p in entry.rglob('*') if p.is_file() and p.name != 'complete.json'}
                    save(entry / 'complete.json', dict(schema=SCHEMA, mode=mode, shift=shift, code=identity,
                         artifacts=artifacts, scope='same completed Oracle model evaluated early; not new training'))
                    checked = audit(entry)
                    checked['baseline_epoch0'] = baseline_check(entry)
                    for stage_name, owner in [('train', read(a.original / 'state.json')['pid']), ('inference', os.getpid())]:
                        receipt = read(entry / (stage_name + '_gpu.json'))
                        guard = read(receipt['guard'])
                        assert guard['owner_pid'] == owner
                    checked['preview_complete_sha256'] = sha(entry / 'complete.json')
                    save(a.output / 'audits' / (mode + '_' + shift + '.json'), checked)
                    finished.append(checked)
                    save(a.output / 'progress.json', dict(count=len(finished), results=finished))
            original_state = read(a.original / 'state.json')
            original_cpu = sum(t['stage'] == 'evaluate_cpu' for t in original_state['active'].values())
            for mode, shift, stage in list(pending):
                card, context = None, None
                if stage == 'evaluate_cpu':
                    if original_cpu + sum(t['stage'] == stage for t in active.values()) >= 2:
                        continue
                else:
                    if sum(t['stage'] == 'inference' for t in active.values()) >= 2:
                        continue
                    # 优先0/2/3；本preview每卡最多一子进程，其他研究共享允许。
                    cards = [g for g in (0, 2, 3, 1) if not any(t['gpu'] == g for t in active.values())]
                    found = acquire(cards, a.data_root, a.output, 12288, identity['commit'])
                    if not found:
                        continue
                    context, card = found
                entry = a.output / 'formal' / mode / shift
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(a.worker_code),
                           PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='123', OMP_NUM_THREADS='8',
                           MKL_NUM_THREADS='8', CUBLAS_WORKSPACE_CONFIG=':4096:8')
                tmpdir = read(entry / 'config.json')['baseline']['tmpdir']
                env.update(TMPDIR=tmpdir, TMP=tmpdir, TEMP=tmpdir)
                if card:
                    env.update(CUDA_VISIBLE_DEVICES=card['uuid'], SADA_BASELINE_LEASE=card['lease'],
                               SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
                command = [sys.executable, '-u', '-B', str(a.worker_code / 'tools/run_phase_endpoint_oracle.py'),
                           '--output', str(a.output), '--entry', str(entry), '--stage', stage]
                with (entry / (stage + '.log')).open('x') as log:
                    proc = subprocess.Popen(command, cwd=a.worker_code, env=env, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                active[mode + '/' + shift + '/' + stage] = dict(process=proc, mode=mode, shift=shift,
                    stage=stage, gpu=card['index'] if card else None, context=context)
                pending.remove((mode, shift, stage))
            save(a.output / 'state.json', dict(pid=os.getpid(), pending=pending, completed=finished,
                active={k:dict(pid=t['process'].pid, mode=t['mode'], shift=t['shift'], stage=t['stage'], gpu=t['gpu']) for k,t in active.items()}))
            time.sleep(3)
        assert len(finished) == len(jobs) and {(r['mode'],r['shift']) for r in finished} == set(jobs)
        paired = {s: (next(r['primary_map'] for r in finished if r['mode']=='gate' and r['shift']==s)
                     - next(r['primary_map'] for r in finished if r['mode']=='control' and r['shift']==s))*100
                  for s in sorted({s for _,s in jobs}) if ('gate',s) in jobs and ('control',s) in jobs}
        save(a.output / 'final.json', dict(count=len(jobs), results=finished, gate_minus_control_pp=paired,
             group_mean_reported=False, scope='target-train-supervised Oracle; fixed20; predeclared completed entries only'))
    except BaseException as exc:
        save(a.output / 'error.json', dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        for task in active.values():
            p = task['process']
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for task in active.values():
            p = task['process']
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker-code', type=Path, required=True)
    p.add_argument('--expected-worker-commit', default='f3aabecbe3f81563ce9177c6ca3ac153ede2cbb4')
    p.add_argument('--jobs', nargs='+', default=['control:S1','control:S3','gate:S1','gate:S3'])
    p.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    main(p.parse_args())
