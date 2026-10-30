"""仅S2层权重×阈值开发搜索：复用一对，新增五对，保留全部结果。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import baseline_protocol as protocol
from tools import run_prototype_capacity as capacity
from tools.run_prototype_wide import assert_alignment_weight, finish_process, memory_policy
from tools.run_s5_paper_params import validate_probe
from tools.run_sada_baseline import acquire
from tools.s2_development import (GRID, REUSED, changed_config, assert_config, freeze_split,
                                   subset_score, select_candidate, PointObserver)

SCHEMA = 's2-target-gt-assisted-development-v1'
REFERENCES = dict(so=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal/S2/so'),
    uda=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal/S2/uda'),
    prototype=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control/S2'))
METHODS = ('uda', 'prototype')
ACTIVE = {}
read, save, sha = protocol.read, protocol.save, protocol.sha


def spawn(mode, entry, lease, data, split_path=None):
    cfg = read(entry/'config.json')
    env = dict(os.environ, PYTHONPATH=str(CODE), PYTHONDONTWRITEBYTECODE='1',
        PYTHONUNBUFFERED='1', PYTHONHASHSEED=str(cfg['seed']), CUDA_VISIBLE_DEVICES=lease['uuid'],
        OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', SADA_BASELINE_LEASE=lease['lease'],
        SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
        TMPDIR=cfg['baseline']['tmpdir'], TMP=cfg['baseline']['tmpdir'], TEMP=cfg['baseline']['tmpdir'])
    for name in ('RQ3_EXTERNAL_GPU_LEASE', 'RQ3_EXTERNAL_GPU_LEASE_SHA256', 'RQ3_COHORT_LEASE'):
        env.pop(name, None)
    cmd = [sys.executable, '-B', str(Path(__file__).resolve()), '--child', mode,
           '--entry', str(entry), '--data-root', str(data)]
    if split_path: cmd += ['--diagnostic-split', str(split_path)]
    with (entry/(mode+'.log')).open('x') as log:
        return subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)


def child(args):
    from tools.baseline_engine import probe, train, evaluate
    from tools.baseline_gpu import first_cuda_guard
    entry = args.entry.resolve(); cfg = read(entry/'config.json')
    if protocol.code_identity() != cfg['baseline']['code']:
        raise ValueError('运行代码与冻结代码不符')
    for path, digest in ((cfg['dataset']['json_file'], cfg['baseline']['source_manifest_sha256']),
                         (cfg['baseline']['target_manifest'], cfg['baseline']['target_manifest_sha256'])):
        if sha(path) != digest: raise ValueError('源/无标注目标manifest改变')
    os.chdir(args.data_root)
    proof = first_cuda_guard()
    if args.child == 'probe': probe(cfg, entry/'result.json')
    elif args.child == 'train': train(cfg, entry)
    else:
        observer = PointObserver(read(args.diagnostic_split)['development']) if args.diagnostic_split else None
        evaluate(cfg, entry, cfg['baseline']['annotation_for_offline_evaluation'],
                 export_predictions=True, observer=observer)
    save(entry/(args.child+'_gpu.json'), dict(guard=proof, guard_sha256=sha(proof),
                                               config_sha256=sha(entry/'config.json')))


def audit(entry):
    entry = Path(entry); record = read(entry/'complete.json'); cfg = read(entry/'config.json')
    method, candidate = record['method'], record['candidate']
    reference = REFERENCES[method]
    if sha(reference/'complete.json') != record['reference_complete_sha256']:
        raise ValueError('原参考完成SHA改变')
    assert_config(cfg, read(reference/'config.json'), candidate, method)
    root = entry.parents[2]
    acceptance = read(root/'acceptance.json')
    inputs = read(root/'inputs.json')
    if (acceptance['inputs_sha256'] != sha(root/'inputs.json') or
            inputs['split_sha256'] != sha(root/'split.json') or
            cfg != inputs['configurations'][candidate][method] or acceptance['code'] != record['code']):
        raise ValueError('配置/开发划分/验收代码绑定改变')
    for path, digest in acceptance['report_sha256'].items():
        if sha(path) != digest: raise ValueError('验收或参考推理证明改变')
    result = protocol.audit_entry(entry, expected_schema=SCHEMA)
    initial = read(REFERENCES['uda']/'termination.json')['initial_state_sha']
    if method == 'prototype':
        cap = cfg['prototype_capacity']
        if sha(cap['prepared_cache']) != cap['prepared_cache_sha256']:
            raise ValueError('固定教师缓存改变')
        init = read(entry/'detector_initialization.json')
        if init['sha256'] != initial or init['specification'] != cap['specification']:
            raise ValueError('实际分支或共享检测器初始化改变')
        rows = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
        for row in rows: assert_alignment_weight(row['auxiliary'])
        if not any(r['auxiliary']['rq3_target_gradient_norm'] > 0 for r in rows):
            raise ValueError('没有真实目标原型梯度')
    elif read(entry/'termination.json')['initial_state_sha'] != initial:
        raise ValueError('UDA冷初始化改变')
    for domain in ('source', 'target'):
        pred = read(entry/'evaluation'/(domain+'_predictions.json'))
        if (pred['config_sha256'] != sha(entry/'config.json') or
                pred['checkpoint_sha256'] != read(entry/'evaluation'/(domain+'_val.json'))['checkpoint_sha256']):
            raise ValueError('预测导出不属于当前配置/checkpoint')
    return dict(result, candidate=candidate, method=method)


def complete(entry, candidate, method, root, identity):
    paths = ['config.json', 'runtime_config.json', 'seed_proof.json', 'metrics.jsonl', 'termination.json',
        'best.pt', 'latest.pt', 'train_gpu.json', 'evaluate_gpu.json', 'evaluation/source_val.json',
        'evaluation/target_val.json', 'evaluation/manifest.json',
        'evaluation/source_predictions.json', 'evaluation/target_predictions.json']
    if method == 'prototype': paths.append('detector_initialization.json')
    save(entry/'complete.json', dict(schema=SCHEMA, shift='S2', regime='uda', method=method, candidate=candidate,
        code=identity, acceptance_sha256=sha(root/'acceptance.json'),
        reference_complete_sha256=sha(REFERENCES[method]/'complete.json'),
        target_gt_used_for_selection=True, target_detection_supervision=False,
        artifacts={str(entry/n): sha(entry/n) for n in paths}))
    return audit(entry)


def export_proof(entry, method):
    reference = REFERENCES[method]
    for domain in ('source', 'target'):
        old = read(reference/'evaluation'/(domain+'_val.json'))
        new = read(entry/'evaluation'/(domain+'_val.json'))
        for field in ('map_0.1_to_0.5', 'map_by_tiou', 'videos', 'prediction_count',
                      'selected_epoch', 'checkpoint_sha256', 'state', 'split'):
            if new[field] != old[field]:
                raise ValueError('旧checkpoint新增预测导出改变评估: '+method+'/'+domain+'/'+field)
    record = dict(status='passed', method=method, trained=False,
        original_complete=str(reference/'complete.json'), original_complete_sha256=sha(reference/'complete.json'),
        original_best_sha256=sha(reference/'best.pt'),
        artifacts={str(p): sha(p) for p in (entry/'evaluation').glob('*.json')},
        gpu_receipt_sha256=sha(entry/'evaluate_gpu.json'))
    save(entry/'export_proof.json', record)
    return record


def stop(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128+signum)


def state(root, phase, completed, **extra):
    save(root/'state.json', dict(phase=phase, pid=os.getpid(), completed=completed,
        active={k: dict(pid=t['process'].pid, stage=t['stage'], gpu=t['lease']['index']) for k,t in ACTIVE.items()},
        **extra))


def run_stage_jobs(jobs, root, data, indices, identity, completed, callback, phase):
    pending = list(jobs)
    while pending or ACTIVE:
        for key, task in list(ACTIVE.items()):
            if task['process'].poll() is None: continue
            if task['process'].returncode:
                raise RuntimeError('阶段失败，保留工件: '+key+'/'+task['stage'])
            # 先释放卡，再做CPU全审，其他卡健康任务不中断。
            task['context'].__exit__(None, None, None); del ACTIVE[key]
            callback(task)
        for job in list(pending):
            found = acquire(indices, data, root, job['minimum'], identity['commit'])
            if not found: break
            context, lease = found
            try:
                process = spawn(job['stage'], job['entry'], lease, data, job.get('diagnostic_split'))
            except BaseException:
                context.__exit__(None, None, None); raise
            ACTIVE[job['key']] = dict(job, process=process, context=context, lease=lease)
            pending.remove(job)
        state(root, phase, completed, pending=[x['key'] for x in pending])
        time.sleep(3)


def finalize_search(root, split, reference_exports, entries):
    dev = {}; scores = {}
    so = subset_score(reference_exports['so']/'evaluation/target_predictions.json', split, split['development'])
    for name in GRID:
        scores[name] = {m: subset_score(entries[name,m]/'evaluation/target_predictions.json', split,
                                        split['development']) for m in METHODS}
        dev[name] = {m: scores[name][m]['map'] for m in METHODS}
    selection = select_candidate(dev, so['map'])
    selection.update(split_sha256=sha(root/'split.json'), development_scores=scores, so_score=so,
                     target_gt_used_for_selection=True, prior_full_target_metric_exposure=True)
    # 先写唯一选择凭证，再打开确认集结果；不得在确认失败后改选。
    path = root/'selection.json'
    with path.open('x') as stream: json.dump(selection, stream, indent=2, allow_nan=False)
    selected_sha = sha(path)
    confirmation = {'so': subset_score(reference_exports['so']/'evaluation/target_predictions.json',
                                       split, split['confirmation']), 'candidates': {}}
    for name in sorted({selection['selected'], selection['best_uda'], REUSED}):
        pair = {m: subset_score(entries[name,m]/'evaluation/target_predictions.json', split,
                                split['confirmation']) for m in METHODS}
        confirmation['candidates'][name] = dict(pair,
            uda_minus_so_pp=100*(pair['uda']['map']-confirmation['so']['map']),
            prototype_minus_uda_pp=100*(pair['prototype']['map']-pair['uda']['map']))
    if sha(path) != selected_sha: raise ValueError('确认阶段修改了开发选择')
    save(root/'confirmation.json', dict(confirmation, selection_sha256=selected_sha,
                                        no_reselection=True, not_an_unseen_benchmark=True))
    return selection


def main(args):
    root = args.output.resolve(); data = args.data_root.resolve(); identity = protocol.code_identity()
    indices = [int(x) for x in args.gpus.split(',')]
    if len(set(indices)) != len(indices) or not indices or not set(indices) <= set(range(4)):
        raise ValueError('物理GPU列表不合法')
    root.mkdir(parents=True, exist_ok=False); tmp = root/'tmp'; tmp.mkdir(); completed = []
    save(root/'plan.json', dict(schema=SCHEMA, shift='S2', internal_id='S2', paper_scenario='S1',
        grid=GRID, reused=REUSED, new_training_count=10, total_paired_entries=12,
        methods=METHODS, no_new_so_training=True, seed=123, batch=2,
        code=identity, physical_gpus=indices, target_gt_used_for_selection=True,
        target_detection_supervision=False, source_only_checkpoint_selection=True,
        prior_full_target_metric_exposure=True))
    try:
        state(root, 'audit_references', completed)
        references = {m: capacity.audit(e) if m == 'prototype' else protocol.audit_entry(e)
                      for m,e in REFERENCES.items()}
        old = {m: read(e/'config.json') for m,e in REFERENCES.items()}
        split = freeze_split(old['uda']); save(root/'split.json', split)
        configs = {}
        for name in GRID:
            for method in METHODS:
                cfg = changed_config(old[method], name, identity, tmp)
                assert_config(cfg, old[method], name, method); configs[name,method] = cfg
            capacity.assert_pair(configs[name,'prototype'], configs[name,'uda'])
        save(root/'inputs.json', dict(code=identity, references=references, split_sha256=sha(root/'split.json'),
            configurations={name: {m: configs[name,m] for m in METHODS} for name in GRID}))
        exports = {}; reports = {}; proofs = {}; jobs = []
        for method, reference in REFERENCES.items():
            entry = root/'reference_exports'/method; entry.mkdir(parents=True, exist_ok=False)
            cfg = deepcopy(old[method]); cfg['baseline'].update(code=identity, tmpdir=str(tmp))
            save(entry/'config.json', cfg)
            for name in ('best.pt', 'termination.json'): (entry/name).symlink_to(reference/name)
            exports[method] = entry
            jobs.append(dict(key='export_'+method, stage='evaluate', entry=entry, method=method,
                minimum=12288, diagnostic_split=root/'split.json' if method in METHODS else None))
        # 代表性两方法验收；shape/损失实现相同，其余只是冻结标量组合。
        for method in METHODS:
            entry = root/'probes'/method; entry.mkdir(parents=True, exist_ok=False)
            save(entry/'config.json', configs['paper_t07',method])
            jobs.append(dict(key='probe_'+method, stage='probe', entry=entry, method=method, minimum=12288))
        initial = read(REFERENCES['uda']/'termination.json')['initial_state_sha']
        def prepared(task):
            method, entry = task['method'], task['entry']
            if task['stage'] == 'probe':
                report = read(entry/'result.json'); validate_probe(report, method, initial); reports[method] = report
            else: proofs[method] = export_proof(entry, method)
        run_stage_jobs(jobs, root, data, indices, identity, completed, prepared, 'reference_exports_and_acceptance')
        if set(reports) != set(METHODS) or set(proofs) != set(REFERENCES):
            raise ValueError('参考推理等价/真实验收不完整')
        policies = [memory_policy(r) for r in reports.values()]
        policy = {k: max(p[k] for p in policies) for k in
                  ('training_minimum_free_mib', 'evaluation_minimum_free_mib')}
        bound = [root/'probes'/m/n for m in METHODS for n in ('result.json', 'probe_gpu.json')]
        bound += [e/'export_proof.json' for e in exports.values()]
        save(root/'acceptance.json', dict(status='passed', batch=2, seed=123, code=identity,
            inputs_sha256=sha(root/'inputs.json'), initial_detector_sha256=initial, resource_policy=policy,
            report_sha256={str(p): sha(p) for p in bound}))
        entries = {(REUSED,m): exports[m] for m in METHODS}
        completed = [dict(references[m], candidate=REUSED, method=m, reused=True) for m in METHODS]
        pending = [(n,m) for n in GRID if n != REUSED for m in METHODS]; evaluations = []
        while pending or evaluations or ACTIVE:
            for key,t in list(ACTIVE.items()):
                if t['process'].poll() is None: continue
                if t['process'].returncode: raise RuntimeError('正式阶段失败: '+key+'/'+t['stage'])
                t['context'].__exit__(None, None, None); del ACTIVE[key]
                if t['stage'] == 'train': evaluations.append((t['candidate'],t['method']))
                else:
                    row = complete(t['entry'],t['candidate'],t['method'],root,identity)
                    completed.append(row); print('COMPLETE',json.dumps(row),flush=True)
            for stage,queue,minimum in [('evaluate',evaluations,policy['evaluation_minimum_free_mib']),
                                        ('train',pending,policy['training_minimum_free_mib'])]:
                for name,method in list(queue):
                    found = acquire(indices,data,root,minimum,identity['commit'])
                    if not found: break
                    context,lease = found; entry = root/'formal'/name/method
                    try:
                        if stage == 'train':
                            entry.mkdir(parents=True,exist_ok=False); save(entry/'config.json',configs[name,method])
                            entries[name,method] = entry
                        process = spawn(stage,entry,lease,data)
                    except BaseException: context.__exit__(None,None,None); raise
                    ACTIVE[name+'/'+method] = dict(process=process,stage=stage,entry=entry,context=context,
                                                   lease=lease,candidate=name,method=method)
                    queue.remove((name,method))
            state(root,'formal',completed,pending=pending,evaluation_pending=evaluations,resource_policy=policy)
            time.sleep(3)
        expected = {(n,m) for n in GRID for m in METHODS}
        if len(completed) != 12 or {(r['candidate'],r['method']) for r in completed} != expected:
            raise ValueError('不是精确六套×两方法/新增10项')
        for name in GRID:
            capacity.assert_pair(configs[name,'prototype'],configs[name,'uda'])
        selection = finalize_search(root,split,exports,entries)
        save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=12,new_training_count=10,reused_count=2,
            results=completed,selection_sha256=sha(root/'selection.json'),confirmation_sha256=sha(root/'confirmation.json'),
            selected=selection['selected'],ordering_found_on_development=selection['ordering_found'],
            target_gt_used_for_selection=True,not_a_strict_uda_benchmark=True))
        state(root,'complete',completed)
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        state(root,'stopped' if isinstance(error,SystemExit) else 'failed',completed); raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None: os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            finish_process(t['process']); t['context'].__exit__(None,None,None)
        ACTIVE.clear()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path); p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus',default='0,2,3,1'); p.add_argument('--child',choices=['probe','train','evaluate'])
    p.add_argument('--entry',type=Path); p.add_argument('--diagnostic-split',type=Path)
    args=p.parse_args(); signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    if args.child: child(args)
    else: main(args)
