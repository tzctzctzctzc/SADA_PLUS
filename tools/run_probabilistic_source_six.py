"""复用已完成S1，仅续跑概率源模型S2–S6；不启动目标学习或新增对照。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import run_probabilistic_teacher as single
from tools import baseline_protocol as protocol
from tools.run_sada_baseline import acquire
from tools.run_prototype_wide import finish_process
from libs.models.SADA.probabilistic_temporal import specification

read,save,sha=protocol.read,protocol.save,protocol.sha
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')
MANIFEST=Path('/vepfs/vlms/migration_pt_source6_1031/manifest.json')
SHIFTS=tuple('S%d'%i for i in range(1,7))
REMAINING=SHIFTS[1:]
ACTIVE={}


def configuration(old,identity,tmp):
    cfg=deepcopy(old)
    cfg['model']['adv_method']=['None']
    cfg['baseline'].update(code=identity,tmpdir=str(tmp),regime='so',
        initialization='matched-cold-student-and-copied-EMA')
    # source分支不读取教师缓存；不为源模型迁移额外VLM依赖。
    cfg['probabilistic_teacher']=dict(variant='source',specification=specification(),
        prepared_cache=None,prepared_cache_sha256=None,variance_scale=[1.,1.],
        source_uncertainty_audit_sha256=None)
    return cfg


def assert_config(cfg,old):
    if (cfg!=configuration(old,cfg['baseline']['code'],cfg['baseline']['tmpdir']) or
        cfg['baseline']['shift'] not in REMAINING or cfg['seed']!=123 or
        cfg['loader']['batch_size']!=2 or 'prototype_capacity' in cfg):
        raise ValueError('仅允许冻结源模型S2–S6，不能改变学习率、预算或新增目标学习')


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');shift=cfg['baseline']['shift']
    assert_config(cfg,read(BASE/shift/'uda/config.json'))
    result=protocol.audit_entry(entry,expected_schema=single.SCHEMA)
    initial=read(entry/'initialization.json')
    if initial['student_sha256']!=initial['teacher_sha256'] or initial['specification']!=specification():
        raise ValueError('EMA初始化或模型规格改变')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if any(r[k] for r in rows for k in ('accepted','target_semantic_loss','target_localization_loss',
                'target_classifier_gradient_norm','target_regressor_gradient_norm')):
        raise ValueError('源模型使用了目标学习')
    if not any(r['source_nll']>0 and r['source_uncertainty_gradient_norm']>0 for r in rows):
        raise ValueError('概率分支无真实源监督')
    return dict(result,variant='source')


def complete(entry,root,identity):
    cfg=read(entry/'config.json')
    names=('config.json','runtime_config.json','seed_proof.json','initialization.json','metrics.jsonl',
        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
    save(entry/'complete.json',dict(schema=single.SCHEMA,shift=cfg['baseline']['shift'],regime='so',
        variant='source',code=identity,acceptance_sha256=sha(root/'acceptance.json'),
        artifacts={str(entry/n):sha(entry/n) for n in names}))
    return audit(entry)


def validate_results(results):
    if len(results)!=6 or {x['shift'] for x in results}!=set(SHIFTS):
        raise ValueError('必须精确六个方向，S1不得重复训练')
    if any(x['regime']!='so' or x['variant']!='source' or not 0<=x['map']<=1 for x in results):
        raise ValueError('不是六方向概率源模型结果')
    return sum(x['map'] for x in results)/6


def state(root,phase,completed,**extra):
    save(root/'state.json',dict(pid=os.getpid(),phase=phase,completed=completed,
        active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['mode'])
                for k,t in ACTIVE.items()},**extra))


def execute(root,data,indices,identity,completed,jobs,on_done):
    pending=list(jobs)
    while pending or ACTIVE:
        for key,t in list(ACTIVE.items()):
            if t['process'].poll() is None:continue
            if t['process'].returncode:raise RuntimeError('阶段失败: '+key+'/'+t['mode'])
            t['context'].__exit__(None,None,None);del ACTIVE[key]
            following=on_done(key,t['mode'],t['entry'])
            if following:pending.insert(0,following)
        while pending:
            found=acquire(indices,data,root,16384,identity['commit'])
            if not found:break
            context,lease=found;key,mode,entry=pending.pop(0)
            try:process=single.spawn(mode,entry,lease,data)
            except BaseException:context.__exit__(None,None,None);raise
            ACTIVE[key]=dict(process=process,context=context,lease=lease,mode=mode,entry=entry)
        state(root,'executing',completed,pending=[dict(shift=k,stage=m) for k,m,e in pending])
        time.sleep(2)


def main(args):
    root=args.output.resolve();data=args.data_root.resolve();previous=args.previous_s1.resolve()
    identity=protocol.code_identity();indices=[int(x) for x in args.gpus.split(',')]
    if not indices or len(set(indices))!=len(indices) or not set(indices)<={0,1}:raise ValueError('只用新机GPU0/1')
    root.mkdir(parents=True,exist_ok=False);completed=[]
    try:
        state(root,'input_audit',completed)
        old=single.audit(previous)
        if old['shift']!='S1' or old['variant']!='source':raise ValueError('前序不是唯一S1源模型')
        # 复用同一科学实现，新增文件只负责五方向调度/审计，不修改模型或训练器。
        for name in ('libs/models/SADA/meta_archs.py','libs/models/SADA/probabilistic_temporal.py',
                     'tools/probabilistic_engine.py','tools/baseline_engine.py'):
            import subprocess
            prior=subprocess.check_output(['git','show',read(previous/'complete.json')['code']['commit']+':'+name],cwd=CODE)
            import hashlib
            if hashlib.sha256(prior).hexdigest()!=sha(CODE/name):raise ValueError('S1科学实现发生变化: '+name)
        completed.append(old);save(root/'predecessor.json',dict(result=old,training_repeated=False,
            uncertainty_failure_does_not_gate_source_training=True))
        manifest=read(MANIFEST)
        for row in manifest['files']:
            if sha(row['path'])!=row['sha256']:raise ValueError('迁移SHA失败: '+row['path'])
        configs={s:configuration(read(BASE/s/'uda/config.json'),identity,'/root/vlm/tmp/pt_'+s) for s in REMAINING}
        for s,cfg in configs.items():
            assert_config(cfg,read(BASE/s/'uda/config.json'));Path(cfg['baseline']['tmpdir']).mkdir(parents=True,exist_ok=True)
            target=read(cfg['baseline']['target_manifest'])['database']
            if not target or any('annotations' in row for row in target.values()):raise ValueError('目标manifest含GT')
        save(root/'plan.json',dict(schema='probabilistic-source-six-v1',code=identity,reused=['S1'],
            new_training=list(REMAINING),additional_controls=False,physical_gpus=indices,
            uncertainty_gate=False,source_only=True,specification=specification(),
            paper_scenarios=dict(zip(SHIFTS,('S3','S1','S4','S2','S5','S6')))))
        save(root/'inputs.json',dict(configurations=configs,migration_manifest_sha256=sha(MANIFEST)))
        jobs=[]
        for s,cfg in configs.items():
            entry=root/'probes'/s;entry.mkdir(parents=True);save(entry/'config.json',cfg);jobs.append((s,'probe',entry))
        reports={}
        def probe_done(key,mode,entry):
            report=read(entry/'result.json')
            if report['status']!='passed' or report['checkpoint_sha256'] is not None:raise ValueError('源模型真实GPU验收失败')
            reports[key]={str(entry/n):sha(entry/n) for n in ('result.json','probe_gpu.json')}
        execute(root,data,indices,identity,completed,jobs,probe_done)
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,
            inputs_sha256=sha(root/'inputs.json'),reports=reports))
        jobs=[]
        for s,cfg in configs.items():
            entry=root/'formal'/s/'source';entry.mkdir(parents=True);save(entry/'config.json',cfg);jobs.append((s,'train',entry))
        def formal_done(key,mode,entry):
            if mode=='train':return key,'evaluate',entry
            completed.append(complete(entry,root,identity))
        execute(root,data,indices,identity,completed,jobs,formal_done)
        mean=validate_results(completed)
        save(root/'final_audit.json',dict(status='passed',count=6,results=completed,mean_map=mean,
            comparison='historical-SADA-results-are-references-not-hardware-or-EMA-matched-controls'))
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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-s1',type=Path,required=True);p.add_argument('--gpus',default='0,1')
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));args=p.parse_args()
    signal.signal(signal.SIGTERM,single.stop);signal.signal(signal.SIGINT,single.stop);main(args)
