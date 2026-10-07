"""重审已经结束的训练，只补缺失的离线评估；不覆盖评估或重训。"""
import argparse
import json
import os
from pathlib import Path
import sys
from tools.run_rq3_matrix import CODE_ROOT, baseline_root, completed_baseline, supervision_audit
from tools.run_rq2_matched_control import checked_train_result, checked_evaluation, run_command
from tools.rq3_protocol import audit_baseline
from tools.rq3_gpu import gpu_slot
from tools.rq2_dense_protocol import sha256_file
from tools.score_rq3_vlm_candidates import check_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / 'complete.json').exists() or any((root / 'evaluation').glob('*.json')):
        raise FileExistsError('existing results must not be overwritten')
    check_code(args.expected_head)
    cfg = json.loads((root / 'preflight.json').read_text())
    aux = cfg['rq3_auxiliary']
    teacher, strategy, shift = aux['teacher_model'], aux['strategy'], cfg['rq2_formal']['shift']
    name = 'rq3_{}_{}_{}_seed6_{}'.format(teacher, strategy, shift, root.parents[2].name)
    data = Path('/vepfs/vlms/SADA_PLUS')
    os.chdir(data)
    baseline = baseline_root(Path('/vepfs/vlms'), shift)
    base_cfg, *_ = completed_baseline(baseline, shift)
    config, checkpoint = checked_train_result(data / 'model_results' / name,
        root / 'logs/uda.train.log', cfg, allow_accelerated=True)
    audit_baseline(base_cfg, json.loads(config.read_text()))
    if sha256_file(aux['cache_path']) != aux['cache_sha256']:
        raise ValueError('cache changed')
    registry = data / 'configs/rq2/shift_registry.yaml'
    # 使用训练记录的冻结配置路径，只切换评估实现。
    config_path = cfg['config_path']
    command = [sys.executable, '-B', str(CODE_ROOT / 'evaluate_da.py'), '-c', config_path,
        '-s', '6', '-i', '0', '--rq2-shift', shift, '--rq2-registry', str(registry),
        '--rq2-reference-root', str(data / 'data/ek100_slowfast/features'),
        '-n', name + '_eval_recovered', '-r', str(checkpoint), '--rq2-formal',
        '--rq2-output-dir', str(root / 'evaluation')]
    with gpu_slot('3', data, wait=True) as selected:
        check_code(args.expected_head)
        run_command(command, root / 'logs/uda.eval.recovered.log', data,
                    selected['uuid'], gpu_index=3)
    manifest = checked_evaluation(root / 'evaluation', cfg, checkpoint)
    if manifest.get('rq3_auxiliary') != aux:
        raise ValueError('evaluation auxiliary identity differs')
    coverage = supervision_audit(config.parent / 'benchmark_epoch_metrics.jsonl', cfg)
    if not coverage['effective']:
        raise ValueError('ineffective auxiliary supervision')
    (root / 'supervision_audit.json').write_text(json.dumps(coverage, sort_keys=True))
    record = dict(teacher=teacher, strategy=strategy, shift=shift, seed=6,
        single_seed_stage_result=True, code_commit=args.expected_head,
        training_code_commit='8593dab4ce64317ce7a652c51f1ab0aff0881578',
        batch_size=cfg['loader']['batch_size'], batch_matched_baseline=False,
        evaluation_gpu=selected, baseline_root=str(baseline), cache_sha256=aux['cache_sha256'],
        checkpoint_sha256=sha256_file(checkpoint),
        manifest_sha256=sha256_file(root / 'evaluation/manifest.json'))
    with (root / 'complete.json').open('x') as stream:
        json.dump(record, stream, sort_keys=True)
    print('RECOVERED_EVALUATION_COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
