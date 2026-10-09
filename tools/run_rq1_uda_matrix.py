"""有限双卡 RQ1 UDA 队列；复用已审计 CLIP-B16/S1，不启动 SO。"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
import yaml
from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_rq1_semantic_uda import MODELS, recipe_from_config, sha, write_json


def audit_item(path, expected_model, expected_shift):
    complete = json.loads((path / 'complete.json').read_text())
    required = {'best.pt', 'latest.pt', 'history.jsonl', 'manifest.json', 'metrics.json',
                'audit.json', 'source_val_predictions.npz', 'target_val_predictions.npz'}
    assert set(complete['sha256']) == required
    for name, digest in complete['sha256'].items():
        assert sha(path / name) == digest, (path, name)
    manifest = json.loads((path / 'manifest.json').read_text())
    repo = Path(__file__).resolve().parents[1]
    cfg = repo / SHIFT_CONFIGS[expected_shift]
    assert manifest['split_config_sha256'] == sha(cfg)
    assert manifest['recipe'] == asdict(recipe_from_config(yaml.safe_load(cfg.read_text()), 6))
    assert (manifest['model'], manifest['shift'], manifest['regime']) == (expected_model, expected_shift, 'uda')
    assert manifest['oracle_action_boundaries'] and manifest['frozen_vlm']
    assert not manifest['target_train_labels_used'] and not manifest['target_val_used_for_training_or_selection']
    history = [json.loads(line) for line in (path / 'history.jsonl').open()]
    latest = torch.load(path / 'latest.pt', map_location='cpu', weights_only=False)
    best = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
    assert latest['recipe'] == best['recipe'] == manifest['recipe']
    assert all(row['target_gt_loss'] == 0 for row in history)
    vals = [row for row in history if 'source_val_loss' in row]
    selected = min(vals, key=lambda row: row['source_val_loss'])
    assert best['epoch'] == selected['epoch'] and best['source_val_loss'] == selected['source_val_loss']
    term = latest['termination']
    assert [row['epoch'] for row in history] == list(range(1, term['epoch'] + 1))
    triggers = [row['epoch'] for row in vals if row['epoch'] - row['best_epoch'] >= manifest['recipe']['patience']]
    assert term['epoch'] == (triggers[0] if triggers else manifest['recipe']['epochs'])
    assert term['reason'] == ('source_val_patience' if triggers else 'max_epochs')
    assert term['updates'] == term['steps_per_epoch'] * term['epoch']
    return {'model': expected_model, 'shift': expected_shift, 'path': str(path),
            'termination': term, 'target_alignment_effective': any(row['target_embedding_gradient'] > 0 for row in history),
            'metrics': json.loads((path / 'metrics.json').read_text()), 'complete_sha256': sha(path / 'complete.json')}


def phase_plan():
    return [[('clip_b16', 'S3,S5'), (','.join(m for m in MODELS if m != 'clip_b16'), 'S1,S3,S5')],
            [(','.join(MODELS), 'S2,S4,S6')]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--pilot', type=Path, default=Path('/vepfs/vlms/rq1_semantic_pilot_dafbf53_1010/clip_b16_S1_uda'))
    p.add_argument('--worker', type=int, choices=[0, 1])
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    gpus = a.gpus.split(',')
    assert len(gpus) == 2 and len(set(gpus)) == 2 and all(g.isdigit() for g in gpus)
    children = []

    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            child.wait()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    if a.worker is not None:
        index = a.worker
        from tools.rq3_gpu import idle_gpu_uuid
        for phase, (models, shifts) in enumerate(phase_plan()[index]):
            # 前一子进程 CUDA 退出后，等 NVML 短暂利用率消退；不抢占外部任务。
            if phase:
                for attempt in range(12):
                    try:
                        idle_gpu_uuid(int(gpus[index]))
                        break
                    except RuntimeError:
                        time.sleep(5)
                else:
                    raise RuntimeError('GPU remained occupied between phases; queue stopped')
            output = a.root / ('gpu' + gpus[index] + '_phase' + str(phase))
            cmd = [sys.executable, '-u', '-m', 'tools.run_rq1_semantic_uda', '--mode', 'train',
                   '--gpu', gpus[index], '--models', models, '--shifts', shifts,
                   '--regimes', 'uda', '--seed', '6', '--output-root', str(output)]
            child = subprocess.Popen(cmd, cwd=repo, start_new_session=True)
            children.append(child)
            write_json(a.root / ('worker' + str(index) + '_state.json'),
                       {'pid': os.getpid(), 'child': child.pid, 'phase': phase, 'status': 'running', 'command': cmd})
            code = child.wait()
            if code:
                write_json(a.root / ('worker' + str(index) + '_state.json'),
                           {'pid': os.getpid(), 'phase': phase, 'status': 'failed', 'returncode': code})
                raise RuntimeError('RQ1 phase failed; partial outputs preserved: ' + str(output))
        write_json(a.root / ('worker' + str(index) + '_state.json'), {'pid': os.getpid(), 'status': 'finished'})
        return
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    pilot = audit_item(a.pilot, 'clip_b16', 'S1')
    expected = {(m, s) for m in MODELS for s in SHIFT_CONFIGS}
    jobs = [(m, s) for phases in phase_plan() for models, shifts in phases for m in models.split(',') for s in shifts.split(',')]
    assert len(jobs) == len(set(jobs)) == 47
    assert set(jobs) | {('clip_b16', 'S1')} == expected
    a.root.mkdir(parents=True, exist_ok=False)
    write_json(a.root / 'plan.json', {'pilot': pilot, 'gpus': gpus, 'phases': phase_plan(),
                                    'new_jobs': jobs, 'scope': '48 UDA, seed6, no new SO'})
    write_json(a.root / 'state.json', {'pid': os.getpid(), 'status': 'starting'})
    for index in (0, 1):
        cmd = [sys.executable, '-u', '-m', 'tools.run_rq1_uda_matrix', '--root', str(a.root),
               '--gpus', a.gpus, '--pilot', str(a.pilot), '--worker', str(index)]
        with (a.root / ('worker' + str(index) + '.log')).open('x') as log:
            children.append(subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    write_json(a.root / 'state.json', {'pid': os.getpid(), 'status': 'running', 'workers': [c.pid for c in children]})
    print('RQ1_UDA48_STARTED', os.getpid(), [c.pid for c in children], flush=True)
    codes = [child.wait() for child in children]
    if any(codes):
        write_json(a.root / 'state.json', {'pid': os.getpid(), 'status': 'failed', 'returncodes': codes})
        raise RuntimeError('worker failed; no automatic overwrite or relaunch')
    try:
        results = [audit_item(a.pilot, 'clip_b16', 'S1')]
        for index, phases in enumerate(phase_plan()):
            for phase, (models, shifts) in enumerate(phases):
                for model in models.split(','):
                    for shift in shifts.split(','):
                        path = a.root / ('gpu' + gpus[index] + '_phase' + str(phase)) / (model + '_' + shift + '_uda')
                        results.append(audit_item(path, model, shift))
        assert len(results) == 48 and {(x['model'], x['shift']) for x in results} == expected
        write_json(a.root / 'final_audit_48.json', {'count': 48, 'results': results,
            'zero_target_alignment': [(x['model'], x['shift']) for x in results if not x['target_alignment_effective']]})
        write_json(a.root / 'state.json', {'pid': os.getpid(), 'status': 'complete', 'audited': 48})
        print('RQ1_UDA48_AUDIT_PASS', flush=True)
    except BaseException:
        write_json(a.root / 'state.json', {'pid': os.getpid(), 'status': 'audit_failed'})
        raise


if __name__ == '__main__':
    main()
