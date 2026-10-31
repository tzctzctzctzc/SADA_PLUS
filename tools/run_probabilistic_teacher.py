"""新机器S1三组：概率源模型→源机制核验→EMA与VLM两个独立完整对照。"""
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

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools import probabilistic_engine as engine
from tools.run_sada_baseline import acquire
from tools.run_prototype_wide import finish_process
from libs.models.SADA.probabilistic_temporal import specification

SCHEMA=engine.SCHEMA
VARIANTS=engine.VARIANTS
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal/S1/uda')
CONTROL=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control/S1')
MANIFEST=Path('/vepfs/vlms/migration_pt_s1_1031/manifest.json')
ACTIVE={}
read,save,sha=protocol.read,protocol.save,protocol.sha


def configuration(old,variant,identity,tmp,scale=(1.,1.),audit_sha=None):
    if variant not in VARIANTS:raise ValueError('未授权的变体')
    cfg=deepcopy(old)
    cfg['model']['adv_method']=['None']
    cfg['baseline'].update(code=identity,tmpdir=str(tmp),regime='so' if variant=='source' else 'uda',
        initialization='matched-cold-student-and-copied-EMA')
    prepared=read(CONTROL/'config.json')['prototype_capacity']
    cfg['probabilistic_teacher']=dict(variant=variant,specification=specification(),
        prepared_cache=prepared['prepared_cache'],prepared_cache_sha256=prepared['prepared_cache_sha256'],
        variance_scale=list(scale),source_uncertainty_audit_sha256=audit_sha)
    return cfg


def assert_config(cfg,old):
    pt=cfg['probabilistic_teacher']
    expected=configuration(old,pt['variant'],cfg['baseline']['code'],cfg['baseline']['tmpdir'],
                           pt['variance_scale'],pt['source_uncertainty_audit_sha256'])
    if cfg!=expected or cfg['seed']!=123 or cfg['loader']['batch_size']!=2 or cfg['baseline']['shift']!='S1':
        raise ValueError('配置超出新概率结构/三组方法的声明范围')
    if 'prototype_capacity' in cfg:raise ValueError('首轮不叠加旧128维分支')


def spawn(mode,entry,lease,data,reference=None):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=lease['lease'],
        SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'])
    for k in ('RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE'):env.pop(k,None)
    command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',mode,'--entry',str(entry),'--data-root',str(data)]
    if reference:command+=['--reference',str(reference)]
    with (entry/(mode+'.log')).open('x') as stream:
        return subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def child(args):
    from tools.baseline_gpu import first_cuda_guard
    from tools import baseline_engine as baseline
    entry=args.entry.resolve();cfg=read(entry/'config.json')
    if protocol.code_identity()!=cfg['baseline']['code']:raise ValueError('不是冻结clean代码')
    if sha(cfg['dataset']['json_file'])!=cfg['baseline']['source_manifest_sha256']:raise ValueError('源manifest改变')
    if sha(cfg['baseline']['target_manifest'])!=cfg['baseline']['target_manifest_sha256']:raise ValueError('无标签目标manifest改变')
    os.chdir(args.data_root);proof=first_cuda_guard()
    if args.child=='probe':engine.probe(cfg,entry,args.reference)
    elif args.child=='train':engine.train(cfg,entry)
    elif args.child=='source_audit':engine.uncertainty_audit(cfg,entry,entry/'uncertainty.json')
    else:baseline.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(entry/'config.json')))


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json')
    assert_config(cfg,read(BASE/'config.json'))
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    initial=read(entry/'initialization.json')
    if initial['student_sha256']!=initial['teacher_sha256'] or initial['specification']!=specification():
        raise ValueError('教师未同权初始化或概率头规格改变')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(r['source_nll']>0 and r['source_positive_points']>0 for r in rows):raise ValueError('概率源监督未生效')
    if not any(r['source_uncertainty_gradient_norm']>0 for r in rows):raise ValueError('概率头没有实际源梯度')
    variant=cfg['probabilistic_teacher']['variant']
    if variant=='source':
        if any(r['accepted'] or r['target_semantic_loss'] or r['target_localization_loss'] for r in rows):
            raise ValueError('仅源域组用了目标辅助')
    else:
        root=entry.parents[2]
        if sha(root/'target_acceptance.json')!=record['target_acceptance_sha256']:raise ValueError('目标GPU验收改变')
        source=root/'formal/S1/source'
        if sha(source/'uncertainty.json')!=cfg['probabilistic_teacher']['source_uncertainty_audit_sha256']:
            raise ValueError('源机制凭证改变')
        mechanism=read(source/'uncertainty.json')
        if mechanism['status']!='passed' or cfg['probabilistic_teacher']['variance_scale']!=mechanism['variance_scale']:
            raise ValueError('源机制失败或未使用冻结源方差标定')
        if initial['initial_state_sha256']!=read(source/'initialization.json')['initial_state_sha256']:
            raise ValueError('三组冷初始化不相同')
        if not any(r['target_classifier_gradient_norm']>0 and r['target_regressor_gradient_norm']>0 for r in rows):
            raise ValueError('目标学习没有直接进入检测双头')
    return dict(result,variant=variant)


def complete(entry,root,identity):
    cfg=read(entry/'config.json');variant=cfg['probabilistic_teacher']['variant']
    names=['config.json','runtime_config.json','seed_proof.json','initialization.json','metrics.jsonl','termination.json',
        'best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
    if variant=='source':names+=['uncertainty.json','source_audit_gpu.json']
    save(entry/'complete.json',dict(schema=SCHEMA,shift='S1',regime=cfg['baseline']['regime'],variant=variant,code=identity,
        acceptance_sha256=sha(root/'acceptance.json'),
        target_acceptance_sha256=sha(root/'target_acceptance.json') if variant!='source' else None,
        artifacts={str(entry/n):sha(entry/n) for n in names}))
    return audit(entry)


def state(root,phase,completed,**extra):
    save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,
        active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()},**extra))


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def serial_stage(root,data,indices,identity,completed,key,mode,entry,minimum,reference=None):
    state(root,'waiting_'+mode,completed)
    while not (found:=acquire(indices,data,root,minimum,identity['commit'])):time.sleep(10)
    context,lease=found
    try:process=spawn(mode,entry,lease,data,reference)
    except BaseException:context.__exit__(None,None,None);raise
    ACTIVE[key]=dict(process=process,context=context,lease=lease,stage=mode,entry=entry)
    state(root,mode,completed)
    if process.wait():raise RuntimeError('阶段失败: '+key+'/'+mode)
    context.__exit__(None,None,None);del ACTIVE[key]


def main(args):
    root=args.output.resolve();data=args.data_root.resolve();identity=protocol.code_identity()
    indices=[int(x) for x in args.gpus.split(',')]
    if len(set(indices))!=len(indices) or not indices or not set(indices)<={0,1}:raise ValueError('只允许新机器物理GPU0/1')
    root.mkdir(parents=True,exist_ok=False);completed=[]
    tmp=Path('/root/vlm/tmp/pt_s1');tmp.mkdir(parents=True,exist_ok=True)
    save(root/'plan.json',dict(schema=SCHEMA,variants=VARIANTS,shift='S1',paper_scenario='S3',seed=123,batch=2,
        specification=specification(),code=identity,physical_gpus=indices,source_only_checkpoint_selection=True,
        source_gate_before_target_training=True,target_gt_used_for_training=False,
        checkpoint_resume='not-supported-refuse-existing-root'))
    try:
        state(root,'input_audit',completed)
        manifest=read(MANIFEST)
        for row in manifest['files']:
            if sha(row['path'])!=row['sha256']:raise ValueError('迁移工件SHA失败: '+row['path'])
        old=read(BASE/'config.json');configs={v:configuration(old,v,identity,tmp) for v in VARIANTS}
        for cfg in configs.values():assert_config(cfg,old)
        target=read(old['baseline']['target_manifest'])['database']
        if not target or any('annotations' in row for row in target.values()):raise ValueError('目标训练manifest有GT')
        cache=read(configs['vlm']['probabilistic_teacher']['prepared_cache'])
        rows=[r for video in cache['by_video'].values() for r in video]
        if (cache['candidate_count']!=200 or cache['selected_count']!=40 or len(rows)!=200 or
                sum(r.get('participation_selected',True) for r in rows)!=40):
            raise ValueError('不是原200候选池中的固定40名单')
        save(root/'inputs.json',dict(code=identity,migration_manifest_sha256=sha(MANIFEST),configurations=configs,
            prepared_cache_sha256=sha(configs['vlm']['probabilistic_teacher']['prepared_cache'])))
        source_probe=root/'probes/source';source_probe.mkdir(parents=True);save(source_probe/'config.json',configs['source'])
        serial_stage(root,data,indices,identity,completed,'source_probe','probe',source_probe,16384)
        report=read(source_probe/'result.json')
        if report['status']!='passed':raise ValueError('源概率头GPU验收失败')
        minimum=max(9216,math.ceil(report['peak_reserved_mib']+4096))
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,inputs_sha256=sha(root/'inputs.json'),
            report_sha256={str(source_probe/n):sha(source_probe/n) for n in ('result.json','probe_gpu.json')},minimum_free_mib=minimum))
        source=root/'formal/S1/source';source.mkdir(parents=True);save(source/'config.json',configs['source'])
        serial_stage(root,data,indices,identity,completed,'source','train',source,minimum)
        serial_stage(root,data,indices,identity,completed,'source','source_audit',source,16384)
        mechanism=read(source/'uncertainty.json')
        # 源机制规则先冻结并落盘，再打开目标GT做一次最终离线评估。
        serial_stage(root,data,indices,identity,completed,'source','evaluate',source,16384)
        completed.append(complete(source,root,identity))
        if mechanism['status']!='passed':
            save(root/'final_audit.json',dict(status='source-mechanism-failed',count=1,results=completed,
                mechanism=str(source/'uncertainty.json'),target_methods_not_started=True))
            state(root,'stopped_source_mechanism_failed',completed);return
        for variant in ('ema','vlm'):
            configs[variant]=configuration(old,variant,identity,tmp,mechanism['variance_scale'],sha(source/'uncertainty.json'))
            probe=root/'probes'/variant;probe.mkdir(parents=True);save(probe/'config.json',configs[variant])
            serial_stage(root,data,indices,identity,completed,variant,'probe',probe,16384,source/'best.pt')
        paths=[root/'probes'/v/n for v in ('ema','vlm') for n in ('result.json','probe_gpu.json')]
        save(root/'target_acceptance.json',dict(status='passed',source_checkpoint_sha256=sha(source/'best.pt'),
            uncertainty_audit_sha256=sha(source/'uncertainty.json'),configurations={v:configs[v] for v in ('ema','vlm')},
            reports={str(p):sha(p) for p in paths}))
        pending=['ema','vlm'];evaluations=[]
        while pending or evaluations or ACTIVE:
            for variant,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('正式阶段失败: '+variant+'/'+t['stage'])
                t['context'].__exit__(None,None,None);del ACTIVE[variant]
                if t['stage']=='train':evaluations.append(variant)
                else:completed.append(complete(t['entry'],root,identity))
            for mode,queue in [('evaluate',evaluations),('train',pending)]:
                for variant in list(queue):
                    found=acquire(indices,data,root,16384,identity['commit'])
                    if not found:break
                    context,lease=found;entry=root/'formal/S1'/variant
                    try:
                        if mode=='train':entry.mkdir(parents=True);save(entry/'config.json',configs[variant])
                        process=spawn(mode,entry,lease,data)
                    except BaseException:context.__exit__(None,None,None);raise
                    ACTIVE[variant]=dict(process=process,context=context,lease=lease,entry=entry,stage=mode);queue.remove(variant)
            state(root,'formal_target_methods',completed,pending=pending,evaluation_pending=evaluations);time.sleep(3)
        if len(completed)!=3 or {x['variant'] for x in completed}!=set(VARIANTS):raise ValueError('不是精确三组')
        maps={x['variant']:x['map'] for x in completed}
        save(root/'final_audit.json',dict(status='passed',count=3,results=completed,
            ema_gain_pp=100*(maps['ema']-maps['source']),vlm_gain_pp=100*(maps['vlm']-maps['ema'])))
        state(root,'complete',completed)
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        state(root,'stopped' if isinstance(error,SystemExit) else 'failed',completed);raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():finish_process(t['process']);t['context'].__exit__(None,None,None)
        ACTIVE.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='0,1')
    p.add_argument('--child',choices=['probe','train','source_audit','evaluate']);p.add_argument('--entry',type=Path)
    p.add_argument('--reference',type=Path);args=p.parse_args()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.child:child(args)
    else:main(args)
