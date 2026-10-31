"""三种上下文辅助×S1–S6；复用旧参照，四卡各至多一个自有CUDA阶段。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools import baseline_protocol as protocol
from tools.context_mix import SCHEMA, VARIANTS, specification, prepare_table
from tools.run_prototype_no_adversary import audit as audit_reference
from tools.run_prototype_wide import assert_alignment_weight, finish_process
from tools.run_sada_baseline import acquire, spawn, child

REFERENCE = Path('/vepfs/vlms/prototype_no_adv6_9edee06_1031/formal')
INVENTORY = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/inputs/feature_inventory.json')
SHIFTS = tuple('S%d' % i for i in range(1, 7))
ACTIVE = {}
read, save, sha = protocol.read, protocol.save, protocol.sha


def changed_config(old, variant, table, identity, tmp):
    cfg = deepcopy(old)
    cfg['baseline'].update(code=identity, tmpdir=str(tmp))
    cfg['context_mix'] = dict(variant=variant, specification=specification(variant),
                              table=str(table), table_sha256=sha(table))
    return cfg


def assert_pair(cfg, reference):
    restored = deepcopy(cfg)
    mix = restored.pop('context_mix')
    if mix['specification'] != specification(mix['variant']):
        raise ValueError('上下文参数改变')
    restored['baseline']['code'] = reference['baseline']['code']
    restored['baseline']['tmpdir'] = reference['baseline']['tmpdir']
    if restored != reference:
        raise ValueError('除声明的上下文辅助及运行身份外改变了旧配置')
    if cfg['seed'] != 123 or cfg['loader']['batch_size'] != 2 or cfg['model']['adv_method'] != ['None']:
        raise ValueError('必须配对seed123/batch2/无对抗原Prototype')
    if cfg['baseline']['policy'] != protocol.POLICY:
        raise ValueError('不能改checkpoint选模或早停')


def prepare(output, identity):
    output.mkdir(parents=True, exist_ok=False)
    references, tables = {}, {}
    for shift in SHIFTS:
        entry = REFERENCE/shift/'prototype'
        references[shift] = audit_reference(entry)
        cfg = read(entry/'config.json')
        table_path = output/'tables'/(shift+'.json')
        table = prepare_table(cfg, INVENTORY, table_path)
        tables[shift] = dict(path=str(table_path), sha256=sha(table_path), summary=table['summary'])
        print('SOURCE_TABLE', shift, json.dumps(table['summary']), flush=True)
    # 冻结特征只全量复核一次；后续供体LRU首次加载仍核其SHA。
    inventory = read(INVENTORY)
    def verify(row):
        if sha(row['path']) != row['sha256']:
            raise ValueError('冻结特征SHA改变: '+row['path'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(verify, inventory))
    save(output/'prepared.json', dict(status='passed', schema=SCHEMA, code=identity,
         references=references, tables=tables, inventory_path=str(INVENTORY), inventory_sha256=sha(INVENTORY)))


def validate_probe(report, cfg, expected_initial):
    if report['initial_detector_sha'] != expected_initial or report['rng_roundtrip']['status'] != 'passed':
        raise ValueError('冷初始化或完整RNG恢复验收不通过')
    for phase in report['phases'].values():
        if phase['target_gt_losses'] != [0., 0., 0.] or not phase['branch_parameters_updated']:
            raise ValueError('目标GT隔离或原Prototype更新验收失败')
        mix = phase['context_mix']
        for key in ('mix_events','mix_cls_raw','mix_reg_raw','mix_positive_points','mix_fpn_gradient_norm'):
            if not math.isfinite(mix[key]) or mix[key] <= 0:
                raise ValueError('真实混合验收缺有效监督: '+key)
        if cfg['context_mix']['variant'] == 'target_context_pair' and not (
                mix['mix_pair_raw'] > 0 and mix['mix_pair_points'] > 0):
            raise ValueError('同位配对没有真实正点和非零损失')
        assert_alignment_weight(phase['auxiliary'])
        if phase['auxiliary']['rq3_target_gradient_norm'] <= 0:
            raise ValueError('原目标Prototype没有非零梯度')


def audit_ledger(entry, table, saved_state):
    import torch
    events = [json.loads(x) for x in (entry/'mix_events.jsonl').read_text().splitlines()]
    cfg = read(entry/'config.json')
    n = cfg['baseline']['epoch_budget']['iterations_per_epoch']
    terminal = read(entry/'termination.json')['decision']['epoch']
    digest = '0'*64
    if saved_state['step'] != n*terminal or saved_state['events'] != len(events) or not events:
        raise ValueError('混合步数/完整记录不匹配')
    source = {v['video_id']: v for v in table['videos']}
    for e in events:
        if e['step'] % 4 or not 1 <= e['step'] <= n*terminal:
            raise ValueError('辅助频率改变')
        if e['context'] != cfg['context_mix']['specification']['context'] or e['target_video'] not in table['target_train']:
            raise ValueError('上下文身份或训练成员不符')
        donor = source[e['source_video']]
        lo, hi = e['source_patch']; u, v = e['inserted']
        if hi-lo != v-u or not 0 <= u < v <= e['context_length']:
            raise ValueError('贴片长度/位置不符')
        selected = [(s, c) for s, c in zip(donor['segments'], donor['labels']) if s[1] > lo and s[0] < hi]
        if any(s[0] < lo or s[1] > hi for s, _ in selected):
            raise ValueError('截断了源GT')
        expected_segments = (torch.tensor([s for s, _ in selected], dtype=torch.float32)-lo+u).tolist()
        if expected_segments != e['segments'] or [c for _, c in selected] != e['labels']:
            raise ValueError('连续源GT平移或完整类别不符')
        background = source[e['source_background_video']]
        a, b = e['source_background']
        if any(end > a and start < b for start, end in background['segments']):
            raise ValueError('背景供体包含任务GT')
        payload = {k: value for k, value in e.items() if k not in ('context','digest')}
        import hashlib
        digest = hashlib.sha256((digest+json.dumps(payload,sort_keys=True,separators=(',',':'))).encode()).hexdigest()
        if digest != e['digest']:
            raise ValueError('辅助事件SHA链不符')
    if digest != saved_state['event_digest']:
        raise ValueError('checkpoint与辅助事件链不符')


def audit(entry):
    import torch
    entry = Path(entry); record = read(entry/'complete.json'); cfg = read(entry/'config.json')
    reference = REFERENCE/record['shift']/'prototype'
    if sha(reference/'complete.json') != record['reference_complete_sha256']:
        raise ValueError('参照完成凭证改变')
    assert_pair(cfg, read(reference/'config.json'))
    mix = cfg['context_mix']
    if sha(mix['table']) != mix['table_sha256']:
        raise ValueError('源表改变')
    result = protocol.audit_entry(entry, expected_schema=SCHEMA)
    cap = cfg['prototype_capacity']
    if sha(cap['prepared_cache']) != cap['prepared_cache_sha256']:
        raise ValueError('旧Qwen类别缓存改变')
    if read(entry/'detector_initialization.json') != read(reference/'detector_initialization.json'):
        raise ValueError('共享检测器或128维分支初始化改变')
    rows = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    for row in rows:
        assert_alignment_weight(row['auxiliary'])
    for name in ('mix_events','mix_reg_raw','mix_fpn_gradient_norm'):
        if not any(r['context_mix'][name] > 0 for r in rows):
            raise ValueError('正式训练混合监督无效: '+name)
    if mix['variant'] == 'target_context_pair' and not any(r['context_mix']['mix_pair_points'] > 0 for r in rows):
        raise ValueError('正式训练没有有效配对正点')
    latest = torch.load(entry/'latest.pt', map_location='cpu')
    if latest['optimizers'][2]['state']:
        raise ValueError('不应更新域判别器')
    audit_ledger(entry, read(mix['table']), latest['model_controls']['context_mix_state'])
    del latest
    ref_map = read(reference/'evaluation/target_val.json')['map_0.1_to_0.5']
    original = read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/record['shift']/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(result, variant=mix['variant'], no_adversary_map=ref_map, original_prototype_map=original,
                delta_no_adversary_pp=100*(result['map']-ref_map), delta_original_pp=100*(result['map']-original))


def complete(entry, root, identity):
    cfg = read(entry/'config.json')
    paths = ('config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
             'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
             'mix_events.jsonl','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
    reference = REFERENCE/cfg['baseline']['shift']/'prototype'
    save(entry/'complete.json', dict(schema=SCHEMA, shift=cfg['baseline']['shift'], regime='uda',
         variant=cfg['context_mix']['variant'], code=identity, acceptance_sha256=sha(root/'acceptance.json'),
         reference_complete_sha256=sha(reference/'complete.json'), target_gt_used=False,
         synthetic_gt_origin='source_train_only', artifacts={str(entry/p):sha(entry/p) for p in paths}))
    return audit(entry)


def state(root, phase, completed, **extra):
    save(root/'state.json', dict(phase=phase,pid=os.getpid(),completed=completed,
         active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()},**extra))


def stop(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise SystemExit(128+signum)


def main(args):
    output, data = args.output.resolve(), args.data_root.resolve()
    identity = protocol.code_identity()
    if args.prepare_only:
        prepare(output, identity)
        return
    prepared = read(args.prepared/'prepared.json')
    if prepared['status'] != 'passed' or prepared['code'] != identity:
        raise ValueError('CPU审计与当前clean代码不匹配')
    if sha(prepared['inventory_path']) != prepared['inventory_sha256']:
        raise ValueError('冻结特征清单改变')
    indices = [int(i) for i in args.gpus.split(',')]
    if len(indices) != len(set(indices)) or not set(indices) <= {0,1,2,3}:
        raise ValueError('GPU列表非法')
    output.mkdir(parents=True, exist_ok=False)
    (output/'tmp').mkdir()
    configs, finished, reports = {}, [], {}
    save(output/'plan.json', dict(schema=SCHEMA, code=identity, variants=VARIANTS, shifts=SHIFTS,
         new_training_count=18, reference_retrained=False, physical_gpus=indices,
         seed=123,batch=2,policy=protocol.POLICY,prepared=str(args.prepared),prepared_sha256=sha(args.prepared/'prepared.json')))
    try:
        for shift in SHIFTS:
            if sha(REFERENCE/shift/'prototype/complete.json') != prepared['references'][shift]['complete_sha256']:
                raise ValueError('旧参照完成SHA改变')
            table = prepared['tables'][shift]
            if sha(table['path']) != table['sha256']:
                raise ValueError('源表SHA改变')
            old = read(REFERENCE/shift/'prototype/config.json')
            for variant in VARIANTS:
                cfg = changed_config(old, variant, table['path'], identity, output/'tmp')
                assert_pair(cfg, old)
                configs[(shift, variant)] = cfg
        save(output/'inputs.json', dict(prepared=prepared, reference_retrained=False))
        probes = list(VARIANTS)
        while probes or ACTIVE:
            for key, task in list(ACTIVE.items()):
                if task['process'].poll() is None:
                    continue
                if task['process'].returncode:
                    raise RuntimeError('真实GPU验收失败: '+key)
                task['context'].__exit__(None,None,None);del ACTIVE[key]
                report = read(task['entry']/'result.json')
                expected = read(REFERENCE/'S1/prototype/detector_initialization.json')['sha256']
                validate_probe(report,configs[('S1',key)],expected)
                reports[key] = dict(report=report,path=str(task['entry']/'result.json'),sha256=sha(task['entry']/'result.json'))
            for variant in list(probes):
                found = acquire(indices,data,output,16384,identity['commit'])
                if not found:
                    break
                context, lease = found
                entry = output/'probes'/variant
                try:
                    entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[('S1',variant)])
                    process = spawn('probe',entry/'config.json',entry,lease,data,entry/'result.json')
                except BaseException:
                    context.__exit__(None,None,None);raise
                ACTIVE[variant] = dict(context=context,lease=lease,entry=entry,process=process,stage='probe')
                probes.remove(variant)
            state(output,'real_data_acceptance',finished,pending_probes=probes);time.sleep(3)
        peak = max(p['peak_reserved_mib'] for r in reports.values() for p in r['report']['phases'].values())
        minimum = max(12288, math.ceil(peak+4096))
        save(output/'acceptance.json', dict(status='passed',code=identity,batch=2,seed=123,
             policy=protocol.POLICY,reports=reports,training_minimum_free_mib=minimum))
        pending = [(s,v) for s in SHIFTS for v in VARIANTS]
        evaluating = []
        while pending or evaluating or ACTIVE:
            for key, task in list(ACTIVE.items()):
                if task['process'].poll() is None:
                    continue
                if task['process'].returncode:
                    raise RuntimeError('阶段失败: '+key+'/'+task['stage'])
                task['context'].__exit__(None,None,None);del ACTIVE[key]
                if task['stage']=='train':
                    evaluating.append(task['job'])
                else:
                    finished.append(complete(task['entry'],output,identity))
                    print('COMPLETE',json.dumps(finished[-1]),flush=True)
            for stage, queue in [('evaluate', evaluating),('train',pending)]:
                for job in list(queue):
                    found = acquire(indices,data,output,minimum,identity['commit'])
                    if not found:
                        break
                    context,lease = found
                    shift,variant = job;key = shift+'/'+variant
                    entry = output/'formal'/shift/variant
                    try:
                        if stage=='train':
                            entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[job])
                        process = spawn(stage,entry/'config.json',entry,lease,data)
                    except BaseException:
                        context.__exit__(None,None,None);raise
                    ACTIVE[key] = dict(context=context,lease=lease,entry=entry,process=process,stage=stage,job=job)
                    queue.remove(job)
            state(output,'formal',finished,pending=pending,evaluation_pending=evaluating);time.sleep(3)
        expected = {(s,v) for s in SHIFTS for v in VARIANTS}
        if {(r['shift'],r['variant']) for r in finished} != expected or len(finished)!=18:
            raise ValueError('不是精确三组×六方向')
        summaries = {v:dict(mean_map=sum(r['map'] for r in finished if r['variant']==v)/6,
                            delta_no_adversary_pp=sum(r['delta_no_adversary_pp'] for r in finished if r['variant']==v)/6)
                     for v in VARIANTS}
        save(output/'final_audit.json',dict(status='passed',count=18,results=finished,summaries=summaries))
        state(output,'complete',finished)
    except BaseException as error:
        save(output/'error.json',dict(type=type(error).__name__,message=str(error)))
        state(output,'stopped' if isinstance(error,SystemExit) else 'failed',finished)
        raise
    finally:
        for task in ACTIVE.values():
            if task['process'].poll() is None:
                os.killpg(task['process'].pid,signal.SIGTERM)
        for task in ACTIVE.values():
            finish_process(task['process']);task['context'].__exit__(None,None,None)
        ACTIVE.clear()


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--prepared',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--gpus',default='0,1,2,3')
    args = p.parse_args()
    if not args.prepare_only and args.prepared is None:
        p.error('正式启动必须提供当前代码的CPU审计目录')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    main(args)
