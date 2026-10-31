"""六方向原版Prototype：真实无标签目标流、无域对抗、源验证mAP选模。"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import baseline_protocol as protocol
from tools.run_sada_baseline import acquire, spawn, prepare_file_limit
from libs.models.SADA.prototype_capacity import specification

SCHEMA = 'prototype-no-adversary-source-map-v1'
SHIFTS = tuple('S%d' % i for i in range(1, 7))
ACTIVE = {}
read, save, sha = protocol.read, protocol.save, protocol.sha


def paired_config(base, prepared, identity, tmp):
    if (base['seed'] != 123 or base['loader']['batch_size'] != 2 or
            base['baseline']['regime'] != 'uda' or base['baseline']['policy'] != protocol.POLICY):
        raise ValueError('必须与本轮seed123/batch2源mAP基线配对')
    cfg = deepcopy(base)
    cfg['model']['adv_method'] = ['None']
    cfg['baseline'].update(code=identity, tmpdir=str(tmp))
    cfg['prototype_capacity'] = dict(variant='control', specification=specification('control'),
        prepared_cache=str(prepared), prepared_cache_sha256=sha(prepared))
    return cfg


def assert_pair(cfg, base):
    cap = cfg['prototype_capacity']
    if cap['variant'] != 'control' or cap['specification'] != specification('control'):
        raise ValueError('只能使用128维单层单原型、目标权重0.1')
    expected = paired_config(base, Path(cap['prepared_cache']), cfg['baseline']['code'], cfg['baseline']['tmpdir'])
    if cfg != expected:
        raise ValueError('配置除关闭对抗、新增原版分支和运行身份外发生变化')


def validate_aux(aux):
    for key in ('rq3_target_gradient_norm', 'rq3_alignment_loss', 'rq3_source_loss', 'rq3_selected'):
        if not math.isfinite(aux[key]) or aux[key] <= 0:
            raise ValueError('缺少实际有效辅助监督: ' + key)
    expected = aux['rq3_source_loss'] + .1 * aux['rq3_alignment_loss']
    if abs(aux['rq3_loss'] - expected) > 1e-5 * max(1., abs(expected)):
        raise ValueError('不是源CE加0.1目标原型损失')


def validate_probe(report, expected_initial):
    if report['status'] != 'passed' or report['initial_detector_sha'] != expected_initial:
        raise ValueError('真实验收或检测器冷初始化不配对')
    if report['rng_roundtrip']['status'] != 'passed' or report['checked_target_batches'] <= 0:
        raise ValueError('缺少目标隔离或GPU完整状态恢复证据')
    for phase in report['phases'].values():
        if (not phase['branch_parameters_updated'] or not phase['model_updated'] or
                phase['target_gt_losses'] != [0., 0., 0.] or phase['source_gradient_norm'] <= 0):
            raise ValueError('真实参数更新、源监督或GT隔离失败')
    validate_aux(report['phases']['5']['auxiliary'])


def audit(entry):
    import torch
    entry = Path(entry)
    record, cfg = read(entry/'complete.json'), read(entry/'config.json')
    baseline = Path(record['baseline_complete']).parent
    if sha(baseline/'complete.json') != record['baseline_complete_sha256']:
        raise ValueError('配对基线SHA改变')
    assert_pair(cfg, read(baseline/'config.json'))
    result = protocol.audit_entry(entry, expected_schema=SCHEMA)
    initial = read(entry/'detector_initialization.json')
    if (initial['sha256'] != read(baseline/'termination.json')['initial_state_sha'] or
            initial['specification'] != specification('control')):
        raise ValueError('共享检测器初始化或分支结构不符')
    rows = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(x['auxiliary']['rq3_target_gradient_norm'] > 0 for x in rows):
        raise ValueError('目标辅助梯度始终为零')
    for row in rows:
        aux = row['auxiliary']
        if not all(math.isfinite(v) for v in aux.values()):
            raise ValueError('辅助指标非有限')
        if abs(aux['rq3_loss']-aux['rq3_source_loss']-.1*aux['rq3_alignment_loss']) > 1e-5*max(1.,abs(aux['rq3_loss'])):
            raise ValueError('正式辅助权重改变')
    for name in ('best', 'latest'):
        saved = torch.load(entry/(name+'.pt'), map_location='cpu')
        if saved['optimizers'][2]['state']:
            raise ValueError('关闭对抗后仍更新域判别器')
        del saved
    result.update(method='prototype_no_adversary', variant='control')
    result['reference_maps'] = {m: read(baseline.parent/m/'evaluation/target_val.json')['map_0.1_to_0.5']
                                for m in ('so', 'uda')}
    result['delta_pp'] = {m: 100*(result['map']-v) for m,v in result['reference_maps'].items()}
    return result


def complete(entry, root, identity, baseline):
    paths = ('config.json', 'runtime_config.json', 'seed_proof.json', 'detector_initialization.json',
        'metrics.jsonl', 'termination.json', 'best.pt', 'latest.pt', 'train_gpu.json', 'evaluate_gpu.json',
        'evaluation/source_val.json', 'evaluation/target_val.json', 'evaluation/manifest.json')
    save(entry/'complete.json', dict(schema=SCHEMA, shift=entry.parent.name, regime='uda', code=identity,
        acceptance_sha256=sha(root/'acceptance.json'), baseline_complete=str(baseline/'complete.json'),
        baseline_complete_sha256=sha(baseline/'complete.json'), target_gt_used=False,
        adversarial_alignment=False, artifacts={str(entry/p): sha(entry/p) for p in paths}))
    return audit(entry)


def finish(task):
    p = task['process']
    if p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)
    try:
        p.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
    task['context'].__exit__(None, None, None)


def stop(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128+signum)


def state(root, phase, completed, **extra):
    save(root/'state.json', dict(phase=phase, pid=os.getpid(), completed=completed,
        active={k: dict(pid=t['process'].pid, gpu=t['lease']['index'], stage=t['stage'])
                for k,t in ACTIVE.items()}, **extra))


def main(args):
    root, base, prepared, data = args.output.resolve(), args.baseline.resolve(), args.prepared.resolve(), args.data_root.resolve()
    identity = protocol.code_identity()
    indices = [int(x) for x in args.gpus.split(',')]
    if not indices or len(set(indices)) != len(indices) or not set(indices) <= {0,1}:
        raise ValueError('本轮仅新服务器物理GPU0/1，各一个自有CUDA')
    root.mkdir(parents=True, exist_ok=False)
    tmp = root/'tmp'; tmp.mkdir()
    save(root/'runtime_resources.json', prepare_file_limit())
    done = []
    try:
        state(root, 'audit_inputs', done)
        if read(base/'final_audit.json')['status'] != 'passed' or read(base/'final_audit.json')['count'] != 12:
            raise ValueError('当前基线12项未全审')
        evidence = read(prepared/'audit.json')
        if evidence['status'] != 'passed' or set(evidence['entries']) != set(SHIFTS):
            raise ValueError('必须是原六方向教师证据')
        configs, refs, migrations = {}, {}, {}
        for s in SHIFTS:
            # 复用先前独立全审结果，仅在完整凭证SHA不变时放行。
            for m in ('so','uda'):
                entry = base/'formal'/s/m
                previous = read(base/'monitor/audits'/('%s_%s.json'%(s,m)))
                if previous['status'] != 'passed' or previous['result']['complete_sha256'] != sha(entry/'complete.json'):
                    raise ValueError('基线缺少当前有效的独立审计')
            path = prepared/(s+'.json'); row = evidence['entries'][s]
            if sha(path) != row['sha256']:
                raise ValueError('迁移教师工件SHA改变')
            payload = read(path)
            if (payload['target_gt_used'] or payload['selected_count'] != 40 or payload['candidate_count'] != 200
                    or payload['teacher'] != 'qwen3_vl_8b' or payload['shift'] != s):
                raise ValueError('教师、方向或40/200协议不符')
            manifest = read(base/'inputs'/s/'target_unlabeled.json')['database']
            if not set(payload['by_video']) <= set(manifest):
                raise ValueError('教师候选不属于当前目标训练视频')
            cfg = paired_config(read(base/'formal'/s/'uda/config.json'), path, identity, tmp)
            assert_pair(cfg, read(base/'formal'/s/'uda/config.json'))
            configs[s] = cfg
            refs[s] = {m:sha(base/'formal'/s/m/'complete.json') for m in ('so','uda')}
            migrations[s] = dict(original_path=row['path'], local_path=str(path), sha256=sha(path))
        save(root/'inputs.json', dict(code=identity, baseline=str(base), reference_complete_sha=refs,
            configurations=configs, evidence_migration=migrations, evidence_audit_sha256=sha(prepared/'audit.json')))
        save(root/'plan.json', dict(schema=SCHEMA, shifts=SHIFTS, count=6, seed=123, batch=2,
            specification=specification('control'), adversarial_alignment=False, policy=protocol.POLICY,
            physical_gpus=indices, target_gt_used=False))
        probe = root/'probes/control'; probe.mkdir(parents=True)
        save(probe/'config.json', configs['S1'])
        while not (found := acquire(indices, data, root, 12288, identity['commit'])):
            state(root, 'waiting_acceptance_gpu', done); time.sleep(10)
        context, lease = found
        try:
            process = spawn('probe', probe/'config.json', probe, lease, data, probe/'result.json')
        except BaseException:
            context.__exit__(None,None,None); raise
        ACTIVE['acceptance'] = dict(process=process, context=context, lease=lease, stage='probe')
        state(root, 'real_data_acceptance', done)
        if process.wait():
            raise RuntimeError('Prototype真实GPU验收失败，保留probe.log')
        report = read(probe/'result.json')
        validate_probe(report, read(base/'formal/S1/uda/termination.json')['initial_state_sha'])
        finish(ACTIVE.pop('acceptance'))
        minimum = max(12288, math.ceil(report['peak_reserved_mib']+4096))
        save(root/'acceptance.json', dict(status='passed', batch=2, seed=123, code=identity,
            inputs_sha256=sha(root/'inputs.json'), minimum_free_mib=minimum,
            report_sha256=sha(probe/'result.json'), receipt_sha256=sha(probe/'probe_gpu.json')))
        pending, evaluations = list(SHIFTS), []
        while pending or evaluations or ACTIVE:
            for s,t in list(ACTIVE.items()):
                if t['process'].poll() is None: continue
                if t['process'].returncode: raise RuntimeError(s+'/'+t['stage']+'失败')
                finish(t); del ACTIVE[s]
                if t['stage'] == 'train': evaluations.append(s)
                else:
                    done.append(complete(t['entry'], root, identity, base/'formal'/s/'uda'))
                    print('COMPLETE', json.dumps(done[-1]), flush=True)
            for stage, queue in [('evaluate', evaluations), ('train', pending)]:
                for s in list(queue):
                    found = acquire(indices, data, root, minimum, identity['commit'])
                    if not found: break
                    context, lease = found; entry = root/'formal'/s/'prototype'
                    try:
                        if stage == 'train':
                            entry.mkdir(parents=True, exist_ok=False); save(entry/'config.json', configs[s])
                        process = spawn(stage, entry/'config.json', entry, lease, data)
                    except BaseException:
                        context.__exit__(None,None,None); raise
                    ACTIVE[s] = dict(process=process,context=context,lease=lease,stage=stage,entry=entry)
                    queue.remove(s)
            state(root, 'formal', done, pending=pending, evaluation_pending=evaluations)
            time.sleep(3)
        if len(done) != 6 or {x['shift'] for x in done} != set(SHIFTS):
            raise ValueError('不是精确六方向')
        save(root/'final_audit.json', dict(status='passed', count=6, schema=SCHEMA, results=done,
            mean_map=sum(x['map'] for x in done)/6))
        state(root, 'complete', done)
    except BaseException as error:
        save(root/'error.json', dict(type=type(error).__name__, message=str(error)))
        state(root, 'stopped' if isinstance(error,SystemExit) else 'failed', done)
        raise
    finally:
        for task in ACTIVE.values():
            if task['process'].poll() is None: os.killpg(task['process'].pid, signal.SIGTERM)
        for task in ACTIVE.values(): finish(task)
        ACTIVE.clear()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus', default='0,1')
    args = p.parse_args()
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    main(args)
