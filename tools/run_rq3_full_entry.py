"""独立Qwen100%单项：CPU配对、真实验收/训练、离线评估和独立终审。"""
import argparse
import hashlib
import os
from pathlib import Path
import sys

from tools.rq3_full_launch import (read, save, bind, validate_launch_proof,
                                   acceptance_ready, validate_smoke, audit_launch_reference)
from tools.rq3_full_participation import VARIANT, prepare, audit_pair
from tools.rq3_full_audit import COMPLETE_SCHEMA, training_audit, audit_complete

CODE = Path(__file__).resolve().parents[1]


def commands(reference_path, root, data_root):
    import yaml
    report = prepare(reference_path)
    cfg = report['planned_config']
    shift, aux = cfg['rq2_formal']['shift'], cfg['rq3_auxiliary']
    registry = data_root/'configs/rq2/shift_registry.yaml'
    config_path = CODE/yaml.safe_load(registry.read_text())['shifts'][shift]['config']
    name = 'rq3_qwen100_'+shift+'_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    common = ['-c', str(config_path), '-s', '6', '-i', '0', '--rq2-shift', shift,
              '--rq2-registry', str(registry), '--rq2-reference-root',
              str(data_root/'data/ek100_slowfast/features')]
    command = [sys.executable, '-B', str(CODE/'train_da.py'), *common,
               '-n', name, '--debug', '--benchmark-audit-log', '--rq2-regime', 'uda',
               '--supervised-upper-bound', 'off', '--rq2-target-manifest',
               cfg['rq2_formal']['target_manifest']['path'],
               '--rq3-strategy', 'prototype', '--rq3-revision', 'revised',
               '--rq3-cache', aux['cache_path'], '--rq3-teacher', 'qwen3_vl_8b',
               '--rq3-batch-size', '2', '--rq3-early-stop-patience',
               '15' if aux.get('early_stopping') else '0',
               '--rq3-candidate-manifest', aux['manifest_path'],
               '--rq3-confidence-calibration', aux['calibration_path'],
               '--rq3-full-reference', str(reference_path)]
    return name, common, command


def main():
    from tools.run_rq2_matched_control import check_code, check_new_protocol, run_command, checked_evaluation
    from tools.rq3_protocol import load_cache, audit_baseline
    from tools.rq3_gpu import gpu_slot
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('reference', 'output', 'expected-head'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--launch-proof')
    parser.add_argument('--acceptance')
    parser.add_argument('--gpu', required=True, choices=('0', '1'))
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    check_code(args.expected_head)
    os.chdir(args.data_root)
    root = Path(args.output).resolve()
    reference = Path(args.reference).resolve()
    root.mkdir(parents=True, exist_ok=False); (root/'logs').mkdir()
    name, common, command = commands(reference, root, args.data_root)
    run_root = args.data_root/'model_results'/name
    if run_root.exists():
        raise FileExistsError('existing training output must not be overwritten')
    run_command(command+['--rq2-audit-output', str(root/'preflight.json')],
                root/'logs/preflight.log', args.data_root)
    cfg = read(root/'preflight.json')
    check_new_protocol(cfg)
    old = read(reference.parent/'preflight.json')
    payload, _ = load_cache(old['rq3_auxiliary']['cache_path'])
    pairing = audit_pair(old, cfg, payload['entries'])
    baseline = Path(read(reference)['baseline_root'])/'uda_preflight.json'
    audit_baseline(read(baseline), cfg)
    save(root/'pairing.json', pairing)
    if args.audit_only:
        print('QWEN100_ENTRY_CPU_PASS', cfg['rq2_formal']['shift'], flush=True)
        return
    if not args.launch_proof:
        raise ValueError('RQ4 final audit launch proof required before any GPU diagnostic')
    proof_path = str(Path(args.launch_proof).resolve())
    proof = validate_launch_proof(proof_path, args.expected_head)
    if proof['references'][cfg['rq2_formal']['shift']] != cfg['rq3_auxiliary']['paired_reference']:
        raise ValueError('reference is outside the authorized six-shift plan')
    os.environ['RQ3_FULL_LAUNCH_PROOF'] = proof_path
    if args.smoke_only:
        diagnostic = [sys.executable, '-B', '-m', 'tools.smoke_rq3_full_training',
                      str(root/'smoke.json'), *command[2:]]
        with gpu_slot(args.gpu, wait=True) as selected:
            check_code(args.expected_head)
            run_command(diagnostic, root/'logs/smoke.log', args.data_root,
                        selected['uuid'], selected['index'])
        validate_smoke(read(root/'smoke.json'), cfg)
        if list(run_root.rglob('*.pt')):
            raise ValueError('diagnostic unexpectedly produced a checkpoint')
        return
    if not args.acceptance:
        raise ValueError('formal training requires six-shift real-data acceptance')
    acceptance_path = str(Path(args.acceptance).resolve())
    acceptance_ready(acceptance_path, cfg, proof_path)
    save(root/'launch_reference.json', dict(proof=bind(proof_path), acceptance=bind(acceptance_path)))
    os.environ['RQ3_FULL_ACCEPTANCE'] = acceptance_path
    with gpu_slot(args.gpu, wait=True) as training_gpu:
        check_code(args.expected_head)
        run_command(command, root/'logs/train.log', args.data_root,
                    training_gpu['uuid'], training_gpu['index'])
    config, best, latest, baseline, termination, report = training_audit(run_root, root/'logs/train.log', cfg)
    evaluation = [sys.executable, '-B', str(CODE/'evaluate_da.py'), *common,
                  '-n', name+'_eval', '-r', str(best), '--rq2-formal',
                  '--rq2-output-dir', str(root/'evaluation')]
    with gpu_slot(args.gpu, wait=True) as evaluation_gpu:
        check_code(args.expected_head)
        run_command(evaluation, root/'logs/evaluate.log', args.data_root,
                    evaluation_gpu['uuid'], evaluation_gpu['index'])
    manifest = checked_evaluation(root/'evaluation', cfg, best)
    if manifest.get('rq3_auxiliary') != cfg['rq3_auxiliary']:
        raise ValueError('offline evaluation lost full-participation identity')
    save(root/'supervision_audit.json', report)
    paths = [config, best, latest, baseline, root/'preflight.json', root/'logs/train.log',
             config.parent/'benchmark_epoch_metrics.jsonl', root/'supervision_audit.json',
             root/'evaluation/manifest.json', root/'evaluation/source_val.json',
             root/'evaluation/target_val.json']
    paths += [Path(item['path']) for item in cfg['rq3_auxiliary']['paired_reference'].values()]
    paths += audit_launch_reference(root, cfg, args.expected_head)
    save(root/'complete.json', dict(schema=COMPLETE_SCHEMA, variant=VARIANT,
         teacher='qwen3_vl_8b', strategy='prototype', shift=cfg['rq2_formal']['shift'],
         seed=6, batch_size=2, code_commit=args.expected_head, run_root=str(run_root),
         termination=termination, training_gpu=training_gpu, evaluation_gpu=evaluation_gpu,
         artifacts={str(path): bind(path)['sha256'] for path in paths}))
    audit_complete(root)
    print('QWEN100_ENTRY_COMPLETE', cfg['rq2_formal']['shift'], flush=True)


if __name__ == '__main__':
    main()
