"""同seed123/batch2基线上，直接比较原版Prototype与加宽加深；不跑中间组。"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools.run_sada_baseline import acquire,spawn
from libs.models.SADA.prototype_capacity import specification

SCHEMA='rq3-prototype-capacity-seed123-v1'
ACTIVE={}
read,save,sha=protocol.read,protocol.save,protocol.sha


def stop(signum,frame):
    for task in list(ACTIVE.values()):
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)


def paired_config(base,variant,prepared,identity,tmpdir):
    cfg=deepcopy(base)
    if cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or cfg['baseline']['regime']!='uda':
        raise ValueError('必须是当前seed123/batch2 UDA')
    cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmpdir)
    cfg['prototype_capacity']=dict(variant=variant,specification=specification(variant),
        prepared_cache=str(prepared),prepared_cache_sha256=sha(prepared))
    return cfg


def assert_pair(cfg,base):
    plain=deepcopy(cfg);cap=plain.pop('prototype_capacity')
    if cap['specification']!=specification(cap['variant']):raise ValueError('容量配置改变')
    plain['baseline']['code']=base['baseline']['code'];plain['baseline']['tmpdir']=base['baseline']['tmpdir']
    if plain!=base:raise ValueError('除新增辅助与运行身份外改变了基线配置')


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json')
    base_entry=Path(record['baseline_complete']).parent
    if sha(base_entry/'complete.json')!=record['baseline_complete_sha256']:raise ValueError('配对基线完成凭证改变')
    base=read(base_entry/'config.json');assert_pair(cfg,base)
    if sha(cfg['prototype_capacity']['prepared_cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:
        raise ValueError('教师准备工件改变')
    initial=read(entry/'detector_initialization.json')
    if initial['sha256']!=read(base_entry/'termination.json')['initial_state_sha']:
        raise ValueError('新增分支改变了共享检测器初始化')
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(x.get('auxiliary',{}).get('rq3_target_gradient_norm',0)>0 and
               x['auxiliary'].get('rq3_alignment_loss',0)>0 and x['auxiliary'].get('rq3_source_loss',0)>0 for x in rows):
        raise ValueError('没有真实非零目标原型梯度/源分类监督')
    result['variant']=cfg['prototype_capacity']['variant']
    return result


def probe(root,cfg,indices,data,minimum):
    variant=cfg['prototype_capacity']['variant'];entry=root/'probes'/variant
    entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',cfg)
    while not (found:=acquire(indices,data,root,minimum,cfg['baseline']['code']['commit'])):
        print('WAIT_GPU_FOR_PROBE',variant,flush=True);time.sleep(15)
    context,lease=found
    try:
        process=spawn('probe',entry/'config.json',entry,lease,data,entry/'result.json')
        ACTIVE['probe']=dict(process=process)
        if process.wait():raise RuntimeError('真实数据验收失败: '+str(entry/'probe.log'))
        report=read(entry/'result.json')
        for phase in report['phases'].values():
            if not phase['branch_parameters_updated']:raise ValueError('分支参数未真正优化')
        aux=report['phases']['5']['auxiliary']
        if not all(aux.get(k,0)>0 and math.isfinite(aux[k]) for k in
                ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_source_loss','rq3_selected')):
            raise ValueError('真实验收缺候选/源分类/非零原型梯度')
        if report['rng_roundtrip']['status']!='passed':raise ValueError('GPU完整状态恢复验收失败')
        return report
    finally:ACTIVE.pop('probe',None);context.__exit__(None,None,None)


def main(args):
    root=args.output_root.resolve();data=args.data_root.resolve();base_root=args.baseline_root.resolve()
    identity=protocol.code_identity();root.mkdir(parents=True,exist_ok=False)
    state=dict(phase='preparing',pid=os.getpid(),completed=[],active={});save(root/'state.json',state)
    prepared=read(args.prepared_root/'audit.json')
    if prepared['status']!='passed' or prepared['count']!=6:raise ValueError('教师证据未完成复审')
    args.tmpdir=args.tmpdir.resolve();args.tmpdir.mkdir(parents=True,exist_ok=True)
    inputs=dict(code=identity,baseline_root=str(base_root),baseline_input_sha256=sha(base_root/'inputs/identity.json'),
        feature_inventory_sha256=sha(base_root/'inputs/feature_inventory.json'),prepared_audit=str(args.prepared_root/'audit.json'),
        prepared_audit_sha256=sha(args.prepared_root/'audit.json'),teacher='qwen3_vl_8b',seed=123,batch=2,
        variants={v:specification(v) for v in ('control','wide_deep')})
    for shift,row in prepared['entries'].items():
        if sha(row['path'])!=row['sha256'] or row['selected_count']!=40:raise ValueError('教师名单或准备SHA改变')
    save(root/'inputs.json',inputs)
    indices=[int(v) for v in args.gpus.split(',')]
    first=base_root/'formal/S1/uda'
    while not (first/'complete.json').exists():time.sleep(15)
    protocol.audit_entry(first)
    reports={}
    for variant in ('control','wide_deep'):
        cfg=paired_config(read(first/'config.json'),variant,Path(prepared['entries']['S1']['path']),identity,args.tmpdir)
        reports[variant]=probe(root,cfg,indices,data,12288)
        if reports[variant]['initial_detector_sha']!=read(first/'termination.json')['initial_state_sha']:
            raise ValueError('验收共享检测器初始状态与基线不同')
    minimum=max(12288,math.ceil(max(r['peak_reserved_mib'] for r in reports.values())+4096))
    save(root/'acceptance.json',dict(status='passed',code=identity,inputs_sha256=sha(root/'inputs.json'),
        batch=2,seed=123,minimum_free_mib=minimum,reports=reports,
        report_sha256={str(root/'probes'/v/'result.json'):sha(root/'probes'/v/'result.json') for v in reports}))
    jobs=[(s,v) for s in ('S1','S2','S3','S4','S5','S6') for v in ('wide_deep','control')]
    save(root/'plan.json',dict(schema=SCHEMA,scope='two paired variants; skip width-only',jobs=jobs,inputs=inputs))
    pending=list(jobs);completed=[];audited_baselines=set()
    try:
        while pending or ACTIVE:
            for key,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError('阶段失败: '+key+'/'+task['stage'])
                e=task['entry'];cfg=read(e/'config.json')
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',e/'config.json',e,task['lease'],data)
                    continue
                paths=('config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
                    'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
                    'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
                baseline=base_root/'formal'/task['shift']/'uda'/'complete.json'
                save(e/'complete.json',dict(schema=SCHEMA,shift=task['shift'],regime='uda',variant=task['variant'],
                    code=identity,acceptance_sha256=sha(root/'acceptance.json'),
                    baseline_complete=str(baseline),baseline_complete_sha256=sha(baseline),
                    artifacts={str(e/n):sha(e/n) for n in paths}))
                completed.append(audit(e));task['context'].__exit__(None,None,None);del ACTIVE[key]
                print('COMPLETE',key,completed[-1]['map'],flush=True)
            for job in list(pending):
                shift,variant=job;baseline=base_root/'formal'/shift/'uda'
                if not (baseline/'complete.json').exists():continue
                if shift not in audited_baselines:protocol.audit_entry(baseline);audited_baselines.add(shift)
                found=acquire(indices,data,root,minimum,identity['commit'])
                if not found:break
                context,lease=found;pending.remove(job)
                e=root/'formal'/variant/shift;e.mkdir(parents=True,exist_ok=False)
                cfg=paired_config(read(baseline/'config.json'),variant,Path(prepared['entries'][shift]['path']),identity,args.tmpdir)
                assert_pair(cfg,read(baseline/'config.json'));save(e/'config.json',cfg)
                key=variant+'_'+shift
                ACTIVE[key]=dict(entry=e,shift=shift,variant=variant,stage='train',context=context,lease=lease,
                    process=spawn('train',e/'config.json',e,lease,data))
            save(root/'state.json',dict(phase='formal',pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=v['process'].pid,gpu=v['lease']['index'],stage=v['stage']) for k,v in ACTIVE.items()}))
            time.sleep(5)
    except BaseException:
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise
    if {(r['shift'],r['variant']) for r in completed}!=set(jobs):raise ValueError('非精确12项')
    save(root/'final_audit.json',dict(status='passed',count=12,schema=SCHEMA,results=completed))
    save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--gpus',default='3,0,1,2');p.add_argument('--tmpdir',type=Path,default=Path('/vepfs/vlms/tmp/prototype_capacity'))
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main(args)
    except BaseException as e:
        if args.output_root.exists():save(args.output_root/'error.json',dict(type=type(e).__name__,error=str(e)))
        raise
