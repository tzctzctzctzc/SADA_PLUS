"""单项RQ4：真实CPU预检、独占GPU训练/评估与可复算complete。"""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import re
import sys

from tools.rq4_protocol import (audit_baseline_pair, audit_termination, assert_launch_ready,
                                 file_sha, load_cache)

CODE = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def acceptance_ready(path, teacher, expected_head, strategy=None):
    record = read(path)
    if record.get('status') != 'passed' or record.get('code_commit') != expected_head:
        raise ValueError('acceptance code/status mismatch')
    strategies = set()
    for item in record['reports']:
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('acceptance report changed')
        report = read(item['path'])
        spec = report['spec']
        cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
        if cache['teacher'] != teacher:
            continue
        if (report['status'] != 'passed' or report['formal_result'] is not False
                or report['real_video_features'] is not True or report['batch_size'] != 2
                or report['seed'] != 6 or spec['weight'] != 0.1):
            raise ValueError('invalid real-data acceptance report')
        if strategy is not None:
            from tools.rq4_relational import STRATEGIES, validate_spec
            if strategy not in STRATEGIES or report['strategy'] != strategy:
                continue
            validate_spec(spec, cache)
            metrics = report['metrics']
            if (any(not math.isfinite(v) for v in metrics.values())
                    or metrics['rq4_accepted'] <= 0 or metrics['rq4_gradient_norm'] <= 0
                    or any(metrics['target_gt_'+k+'_loss'] != 0 for k in ('cls','reg','downstream'))):
                raise ValueError('relational acceptance needs real routing/gradient and zero target GT')
        strategies.add(report['strategy'])
    if strategy is not None and strategies != {strategy}:
        raise ValueError('relational real-data acceptance is missing')
    if strategy == 'relational':
        from tools.rq4_relation_launch import validate_acceptance
        validate_acceptance(record)
    if strategy is None and strategies != {'direct','gated'}:
        raise ValueError('both strategies must pass real-data acceptance for this teacher')
    return record


def supervision(path, stop_epoch, relational=False):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r['epoch'] for r in rows] != list(range(1, stop_epoch+1)):
        raise ValueError('incomplete epoch metric history')
    sums = {}
    for key in ('queried', 'matched', 'routable', 'accepted', 'crop_rejected',
                'anchor_rejected', 'gate_rejected', 'loss', 'raw_loss', 'gradient_norm'):
        values = [r['t_rq4_'+key] for r in rows]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError('invalid RQ4 metric: '+key)
        sums[key] = sum(values)
    for r in rows:
        for key in ('cls', 'reg', 'downstream'):
            if r['t_target_gt_'+key+'_loss'] != 0:
                raise ValueError('target GT entered training')
        if not (r['t_rq4_accepted'] <= r['t_rq4_routable'] <= r['t_rq4_matched'] <= r['t_rq4_queried']):
            raise ValueError('invalid boundary routing counts')
    if sums['accepted'] == 0 and (sums['loss'] != 0 or sums['gradient_norm'] != 0):
        raise ValueError('zero intervention produced a gradient')
    if sums['loss'] > 0 and sums['gradient_norm'] == 0:
        raise ValueError('positive auxiliary loss has no gradient')
    if relational:
        for key in ('core_rejected','multipoint_rejected','relation_points','relation_pairs','weight_sum'):
            values = [r['t_rq4_'+key] for r in rows]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('invalid relational metric: '+key)
            sums[key] = sum(values)
        for row in rows:
            r = {k.removeprefix('t_rq4_'):v for k,v in row.items() if k.startswith('t_rq4_')}
            if (r['relation_points'] < 2*r['accepted']-1e-6
                    or r['relation_points'] > 3*r['accepted']+1e-6
                    or not math.isclose(r['relation_pairs'], r['relation_points']-r['accepted'],abs_tol=1e-6)
                    or not math.isclose(r['routable'], sum(r[k] for k in (
                        'accepted','gate_rejected','core_rejected','multipoint_rejected')),abs_tol=1e-6)):
                raise ValueError('invalid relational multi-point routing counts')
            if not math.isclose(r['loss'], .1*r['raw_loss'], rel_tol=1e-5,abs_tol=1e-8):
                raise ValueError('relational loss weight differs')
            if not 0 <= r['weight_sum'] <= r['accepted']+1e-6:
                raise ValueError('invalid soft relation weight sum')
    return dict(epoch_mean_sums=sums, intervened=sums['accepted'] > 0,
        effective_gradient=sums['gradient_norm'] > 0,
        zero_coverage_is_reported_not_filtered=True)


def training_audit(run_root, cfg, baseline):
    from tools.rq2_evaluate import load_checkpoint, selected_checkpoint_state
    from tools.rq2_formal import validate_formal_resume
    configs = list(run_root.glob('*/config.json'))
    best = list(run_root.glob('*/saved_ckpt/best_ckpt/*.pt'))
    latest = list(run_root.glob('*/saved_ckpt/latest_ckpt/*.pt'))
    if len(configs) != 1 or len(best) != 1 or len(latest) != 1:
        raise ValueError('missing/ambiguous config or best/latest checkpoint')
    saved = read(configs[0])
    audit_baseline_pair(baseline, saved)
    if saved['rq4_boundary'] != cfg['rq4_boundary']:
        raise ValueError('saved RQ4 specification differs from preflight')
    stop = load_checkpoint(latest[0])
    validate_formal_resume(stop, cfg['rq2_formal']['run_signature'])
    audit_termination(stop, cfg['rq4_boundary'])
    history = stop['rq4_termination']['source_validation_history']
    if [r[0] for r in history] != list(range(5, stop['epoch']+1, 5)):
        raise ValueError('source validation cadence changed')
    chosen = load_checkpoint(best[0])
    validate_formal_resume(chosen, cfg['rq2_formal']['run_signature'])
    selected_checkpoint_state(chosen, saved, require_provenance=True)
    if chosen['epoch'] != stop['rq4_termination']['decision']['best_epoch']:
        raise ValueError('best checkpoint differs from source stopping history')
    report = supervision(configs[0].parent/'benchmark_epoch_metrics.jsonl', stop['epoch'],
                         cfg['rq4_boundary']['strategy'] in ('relational','proposal_relation'))
    return configs[0], best[0], latest[0], stop['rq4_termination'], report


def audit_complete(root):
    from tools.run_rq2_matched_control import checked_evaluation
    root = Path(root)
    record = read(root/'complete.json')
    cfg, baseline = read(root/'preflight.json'), read(record['baseline_config'])
    spec = cfg['rq4_boundary']
    if spec['strategy']=='consensus':
        raise ValueError('consensus is a separate variant; use rq4_consensus_audit')
    if spec['strategy'] in ('relational','proposal_relation'):
        from tools.rq4_relational import VARIANT
        cache = load_cache(spec['cache_path'],spec['cache_file_sha256'])
        if (record.get('schema') != 'rq4-relational-complete-v1' or record.get('variant') != VARIANT
                or record['strategy'] != spec['strategy'] or record['teacher'] != cache['teacher']
                or record['shift'] != cache['shift'] or record['seed'] != 6 or record['batch_size'] != 2):
            raise ValueError('relational completion identity mismatch')
        reference = read(root/'acceptance_reference.json')
        if file_sha(reference['path']) != reference['sha256']:
            raise ValueError('relational acceptance changed')
        acceptance_ready(reference['path'],cache['teacher'],record['code_commit'],spec['strategy'])
        required = [root/'preflight.json',root/'spec.json',root/'acceptance_reference.json',
                    root/'baseline_pair.json',root/'supervision_audit.json',Path(record['baseline_config'])]
        required += [Path(spec[k+'_path']) for k in ('cache','regions','evidence')]
        required += [root/'evaluation'/name for name in ('manifest.json','source_val.json','target_val.json')]
        if not {str(x) for x in required} <= set(record['artifacts']):
            raise ValueError('relational completion lacks required artifact hashes')
    assert_launch_ready(cfg)
    config, best, latest, termination, report = training_audit(Path(record['run_root']), cfg, baseline)
    if spec['strategy'] in ('relational','proposal_relation'):
        required = [config,best,latest,config.parent/'benchmark_epoch_metrics.jsonl']
        if not {str(x) for x in required} <= set(record['artifacts']):
            raise ValueError('relational completion lacks checkpoint/history hashes')
        if read(root/'baseline_pair.json') != audit_baseline_pair(baseline,cfg):
            raise ValueError('relational baseline pairing differs')
    manifest = checked_evaluation(root/'evaluation', cfg, best)
    if manifest.get('rq4_boundary') != cfg['rq4_boundary']:
        raise ValueError('evaluation lacks RQ4 provenance')
    for path, sha in record['artifacts'].items():
        if file_sha(path) != sha:
            raise ValueError('completed artifact changed: '+path)
    if report != read(root/'supervision_audit.json') or termination != record['termination']:
        raise ValueError('completed supervision/termination differs')
    return record


def main(*, relational=False, consensus=False):
    from tools.run_rq2_matched_control import check_code, check_new_protocol, checked_evaluation, run_command
    from tools.rq3_gpu import gpu_slot
    audit_training=training_audit
    audit_finished=audit_complete
    if consensus:
        from tools.rq4_consensus_audit import training_audit as audit_training, audit_complete as audit_finished
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('baseline-config', 'spec', 'output', 'expected-head'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--gpu', choices=('0','1'), required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    check_code(args.expected_head)
    os.chdir(args.data_root)  # 真实展开配置中的相对数据路径仍相对原数据仓库。
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=False)
    (root/'logs').mkdir()
    baseline = read(args.baseline_config)
    if baseline.get('rq4_boundary') or baseline.get('rq3_auxiliary'):
        raise ValueError('not a no-teacher baseline')
    spec = read(args.spec)
    spec['protocol_frozen'] = True
    spec['baseline_config_path'] = str(Path(args.baseline_config).resolve())
    spec['baseline_config_sha256'] = file_sha(args.baseline_config)
    cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
    shift,teacher = cache['shift'],cache['teacher']
    allowed_teachers = ('timelens2_8b','timeple_8b','sada_self') if relational else ('timelens2_8b','timeple_8b')
    allowed_strategies = ('consensus',) if consensus else (('relational','proposal_relation') if relational else ('direct','gated'))
    if teacher not in allowed_teachers or spec['strategy'] not in allowed_strategies:
        raise ValueError('entry outside authorized matrix')
    save(root/'spec.json',spec)
    # 原始YAML通过冻结注册表；真正的展开配置随后与已完成基线逐字段配对。
    import yaml
    registry = args.data_root/'configs/rq2/shift_registry.yaml'
    entry = yaml.safe_load(registry.read_text())['shifts'][shift]
    config_path = CODE/entry['config']
    # 诊断/正式与不同恢复根永不共享训练目录。
    name = 'rq4_'+teacher+'_'+spec['strategy']+'_'+shift+'_'+hashlib.sha256(str(root).encode()).hexdigest()[:16]
    run_root = args.data_root/'model_results'/name
    if run_root.exists():
        raise FileExistsError('never overwrite an existing training directory')
    common = ['-c',str(config_path),'-s','6','-i','0','--rq2-shift',shift,
        '--rq2-registry',str(args.data_root/'configs/rq2/shift_registry.yaml'),
        '--rq2-reference-root',str(args.data_root/'data/ek100_slowfast/features')]
    command = [sys.executable,'-B',str(CODE/'train_da.py'),*common,'-n',name,'--debug',
        '--benchmark-audit-log','--rq2-regime','uda','--supervised-upper-bound','off',
        '--rq2-target-manifest',baseline['rq2_formal']['target_manifest']['path'],
        '--rq4-config',str(root/'spec.json')]
    run_command(command+['--rq2-audit-output',str(root/'preflight.json')],root/'logs/preflight.log',args.data_root)
    cfg = read(root/'preflight.json')
    check_new_protocol(cfg)
    pairing = audit_baseline_pair(baseline,cfg)
    if consensus and (args.audit_only or args.smoke_only):
        from tools.rq4_consensus import audit_spec
        audit_spec(spec,cache)
    else:
        assert_launch_ready(cfg)
    save(root/'baseline_pair.json',pairing)
    if args.audit_only:
        print('RQ4_ENTRY_CPU_PREFLIGHT_PASS',teacher,spec['strategy'],shift,flush=True)
        return
    if args.smoke_only:
        smoke_module='tools.smoke_rq4_consensus' if consensus else 'tools.smoke_rq4_training'
        diagnostic = [sys.executable,'-B','-m',smoke_module,str(root/'smoke.json'),*command[2:]]
        with gpu_slot(args.gpu,wait=True) as selected:
            run_command(diagnostic,root/'logs/smoke.log',args.data_root,selected['uuid'],selected['index'])
        if read(root/'smoke.json')['status'] != 'passed':
            raise ValueError('RQ4 real-data acceptance failed')
        return
    # 正式队列必须提供已验收证据，不能把--execute等价于已测试。
    acceptance = spec.get('acceptance_path') if consensus else os.environ.get('RQ4_REAL_ACCEPTANCE')
    if not acceptance:
        raise ValueError('real-data acceptance evidence required before formal training')
    if not consensus:
        acceptance_ready(acceptance, teacher, args.expected_head, spec['strategy'] if relational else None)
    save(root/'acceptance_reference.json',dict(path=acceptance,sha256=file_sha(acceptance)))
    with gpu_slot(args.gpu,wait=True) as training_gpu:
        check_code(args.expected_head)
        run_command(command,root/'logs/train.log',args.data_root,training_gpu['uuid'],training_gpu['index'])
    config_path,best,latest,termination,report = audit_training(run_root,cfg,baseline)
    log = (root/'logs/train.log').read_text(errors='replace')
    if re.search(r'Traceback|CUDA out of memory|\bloss\w*\s*:\s*[-+]?(?:nan|inf)\b',log,re.I):
        raise ValueError('training contains technical error/nonfinite loss')
    evaluation = [sys.executable,'-B',str(CODE/'evaluate_da.py'),*common,'-n',name+'_eval',
        '-r',str(best),'--rq2-formal','--rq2-output-dir',str(root/'evaluation')]
    with gpu_slot(args.gpu,wait=True) as evaluation_gpu:
        run_command(evaluation,root/'logs/evaluate.log',args.data_root,evaluation_gpu['uuid'],evaluation_gpu['index'])
    manifest = checked_evaluation(root/'evaluation',cfg,best)
    if manifest.get('rq4_boundary') != cfg['rq4_boundary']:
        raise ValueError('offline evaluation lost RQ4 identity')
    save(root/'supervision_audit.json',report)
    artifacts = [config_path,best,latest,root/'preflight.json',root/'evaluation/manifest.json',
        root/'evaluation/source_val.json',root/'evaluation/target_val.json',Path(spec['cache_path']),
        Path(spec['evidence_path']),Path(args.baseline_config),root/'spec.json',root/'supervision_audit.json']
    if spec['strategy'] == 'gated': artifacts.append(Path(spec['gate_path']))
    if relational or consensus:
        artifacts += [Path(spec['regions_path']),config_path.parent/'benchmark_epoch_metrics.jsonl',
                      root/'acceptance_reference.json',root/'baseline_pair.json']
    if consensus:
        artifacts += [Path(spec['quality_path']),Path(spec['source_policy_path']),Path(spec['acceptance_path'])]
    record = dict(schema='rq4-consensus-complete-v1' if consensus else ('rq4-relational-complete-v1' if relational else 'rq4-complete-v1'),
        teacher=teacher,strategy=spec['strategy'],
        shift=shift,seed=6,batch_size=2,code_commit=args.expected_head,termination=termination,
        baseline_config=str(Path(args.baseline_config).resolve()),run_root=str(run_root),
        training_gpu=training_gpu,evaluation_gpu=evaluation_gpu,
        artifacts={str(path):file_sha(path) for path in artifacts})
    if relational or consensus: record['variant'] = spec['variant']
    save(root/'complete.json',record)
    audit_finished(root)
    print('RQ4_ENTRY_COMPLETE',teacher,spec['strategy'],shift,flush=True)


if __name__ == '__main__': main()
