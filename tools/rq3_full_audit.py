"""Qwen Prototype100%独立终审；不复用旧20%矩阵的完成身份。"""
import json
import math
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_full_participation import VARIANT, audit_pair, validate_runtime

COMPLETE_SCHEMA = 'rq3-qwen-prototype-full-complete-v1'


def read(path):
    return json.loads(Path(path).read_text())


def supervision(path, stop_epoch):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if [row['epoch'] for row in rows] != list(range(1, stop_epoch + 1)):
        raise ValueError('missing full-participation epoch history')
    keys = ('eligible', 'selected', 'loss', 'gradient_norm', 'source_loss', 'alignment_loss',
            'target_gradient_norm', 'aligned_classes', 'candidate_count', 'fully_visible_count')
    totals = {key: 0.0 for key in keys}
    for row in rows:
        for key in ('cls', 'reg', 'downstream'):
            if row['t_target_gt_' + key + '_loss'] != 0:
                raise ValueError('target GT entered full-participation training')
        for key in keys:
            value = row['t_rq3_' + key]
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid full-participation auxiliary metric: ' + key)
            totals[key] += value
        if not (row['t_rq3_selected'] <= row['t_rq3_eligible'] <= row['t_rq3_fully_visible_count']
                <= row['t_rq3_candidate_count']):
            raise ValueError('invalid full-participation routing counts')
    target_keys = ('alignment_loss', 'target_gradient_norm', 'aligned_classes')
    intervened = totals['selected'] > 0
    target_ok = (all(totals[key] > 0 for key in target_keys) if intervened
                 else all(totals[key] == 0 for key in target_keys))
    effective = all(totals[key] > 0 for key in ('source_loss', 'gradient_norm', 'fully_visible_count')) and target_ok
    return dict(epoch_mean_sums=totals, intervened=intervened, effective=effective,
                actual_routed_coverage_guaranteed=False)


def training_audit(run_root, train_log, cfg):
    from tools.rq2_evaluate import load_checkpoint
    from tools.run_rq2_matched_control import checked_train_result
    from tools.rq3_protocol import load_cache, audit_baseline
    if cfg['rq3_auxiliary']['schema'] != VARIANT:
        raise ValueError('not a full-participation result')
    config, best = checked_train_result(Path(run_root), Path(train_log), cfg)
    saved = read(config)
    proof = cfg['rq3_auxiliary']['paired_reference']
    reference = read(proof['preflight']['path'])
    payload, _ = load_cache(reference['rq3_auxiliary']['cache_path'])
    if saved['rq3_auxiliary'] != cfg['rq3_auxiliary']:
        raise ValueError('saved full-participation identity differs from preflight')
    audit_pair(reference, saved, payload['entries'])
    baseline = Path(read(proof['complete']['path'])['baseline_root']) / 'uda_preflight.json'
    audit_baseline(read(baseline), saved)
    latest = list(Path(run_root).glob('*/saved_ckpt/latest_ckpt/*.pt'))
    if len(latest) != 1:
        raise ValueError('missing or ambiguous latest checkpoint')
    final, chosen = load_checkpoint(latest[0]), load_checkpoint(best)
    policy = cfg['rq3_auxiliary'].get('early_stopping')
    if policy:
        history = final['rq3_termination']['source_validation_history']
        if ([row[0] for row in history] != list(range(5, final['epoch'] + 1, 5))
                or chosen['epoch'] != final['rq3_termination']['decision']['best_epoch']):
            raise ValueError('source validation cadence or selected checkpoint differs')
    termination = dict(policy=policy, actual_epoch=final['epoch'], selected_epoch=chosen['epoch'],
        reason=final['rq3_termination']['decision']['reason'] if policy else 'max_epochs')
    report = supervision(config.parent / 'benchmark_epoch_metrics.jsonl', final['epoch'])
    if not report['effective']:
        raise ValueError('full-participation auxiliary gradient audit failed')
    return config, best, latest[0], baseline, termination, report


def audit_complete(root):
    from tools.run_rq2_matched_control import checked_evaluation
    root = Path(root)
    record, cfg = read(root/'complete.json'), read(root/'preflight.json')
    if (record.get('schema') != COMPLETE_SCHEMA or record.get('variant') != VARIANT
            or record.get('teacher') != 'qwen3_vl_8b' or record.get('strategy') != 'prototype'
            or record.get('shift') != cfg['rq2_formal']['shift']
            or record.get('seed') != 6 or record.get('batch_size') != 2):
        raise ValueError('completed identity is not the independent full-participation variant')
    validate_runtime(cfg, 'uda')
    config, best, latest, baseline, termination, report = training_audit(
        record['run_root'], root/'logs/train.log', cfg)
    manifest = checked_evaluation(root/'evaluation', cfg, best)
    if manifest.get('rq3_auxiliary') != cfg['rq3_auxiliary']:
        raise ValueError('offline evaluation lost full-participation identity')
    if report != read(root/'supervision_audit.json') or termination != record['termination']:
        raise ValueError('completed supervision or termination changed')
    required = [config, best, latest, baseline, root/'preflight.json', root/'logs/train.log',
                config.parent/'benchmark_epoch_metrics.jsonl', root/'supervision_audit.json',
                root/'evaluation/manifest.json', root/'evaluation/source_val.json',
                root/'evaluation/target_val.json']
    required += [Path(item['path']) for item in cfg['rq3_auxiliary']['paired_reference'].values()]
    if record.get('artifacts') != {str(path): sha256_file(path) for path in required}:
        raise ValueError('full-participation artifact set or SHA changed')
    return record
