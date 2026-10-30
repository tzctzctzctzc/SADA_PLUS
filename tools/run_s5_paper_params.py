"""S5旧→新：只改六层权重和SADA伪标签阈值，配对训练UDA及原128维Prototype。"""
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
from tools import run_prototype_capacity as capacity
from tools.run_prototype_wide import assert_alignment_weight,finish_process,memory_policy
from tools.run_sada_baseline import acquire,spawn
from libs.models.SADA.prototype_capacity import specification

SCHEMA='sada-s5-paper-hyperparameters-pair-v1'
LEVELS=(1.,.4,.8,.7,.9,.6)
THRESHOLD=.5
REFERENCES=dict(uda=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal/S5/uda'),
                prototype=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control/S5'))
ACTIVE={}
read,save,sha=protocol.read,protocol.save,protocol.sha


def changed_config(old,identity,tmp):
    cfg=deepcopy(old)
    for i,w in enumerate(LEVELS):cfg['model']['weight_level_%d'%i]=w
    cfg['model']['pl_threshold']=THRESHOLD
    cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmp)
    return cfg


def assert_declared_changes(cfg,old,method):
    if method not in REFERENCES:raise ValueError('只允许S5 UDA/Prototype两项')
    if old['baseline']['shift']!='S5' or old['baseline']['regime']!='uda':raise ValueError('不是冻结S5方向')
    if old['seed']!=123 or old['loader']['batch_size']!=2:raise ValueError('不是seed123/batch2')
    expected=changed_config(old,cfg['baseline']['code'],cfg['baseline']['tmpdir'])
    if cfg!=expected:raise ValueError('除六层权重/阈值/运行身份外发生变化')
    if method=='prototype':
        cap=cfg['prototype_capacity']
        if cap['variant']!='control' or cap['specification']!=specification('control'):
            raise ValueError('必须是128维单层、单原型、目标权重0.1')
    elif 'prototype_capacity' in cfg:raise ValueError('UDA不允许带Prototype分支')


def validate_probe(report,method,initial):
    if report['status']!='passed' or report['batch']!=2 or report['initial_detector_sha']!=initial:
        raise ValueError('真实验收身份/共享初始化不配对')
    if report['rng_roundtrip']['status']!='passed' or report['checked_target_batches']<=0:
        raise ValueError('缺少完整RNG恢复/目标隔离证明')
    if report['cuda_seed_proof']['actual_cuda_seed']!=123:raise ValueError('GPU种子不是123')
    for p in report['phases'].values():
        if (not p['model_updated'] or p['target_gt_losses']!=[0.,0.,0.]
                or not math.isfinite(p['source_gradient_norm']) or p['source_gradient_norm']<=0):
            raise ValueError('缺少真实源监督更新或有目标GT')
        if method=='prototype' and not p['branch_parameters_updated']:raise ValueError('辅助分支没有真实更新')
    if method=='prototype':
        aux=report['phases']['5']['auxiliary']
        if not all(math.isfinite(aux[k]) and aux[k]>0 for k in
                   ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_source_loss','rq3_selected')):
            raise ValueError('缺少实际候选/源分类/目标原型梯度')
        for p in report['phases'].values():assert_alignment_weight(p['auxiliary'])


def audit(entry):
    entry=Path(entry);record=read(entry/'complete.json');method=record['method'];cfg=read(entry/'config.json')
    reference=REFERENCES[method]
    if sha(reference/'complete.json')!=record['reference_complete_sha256']:raise ValueError('原对照完成凭证改变')
    assert_declared_changes(cfg,read(reference/'config.json'),method)
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    initial=read(REFERENCES['uda']/'termination.json')['initial_state_sha']
    if method=='prototype':
        cap=cfg['prototype_capacity']
        if sha(cap['prepared_cache'])!=cap['prepared_cache_sha256']:raise ValueError('原固定教师缓存改变')
        init=read(entry/'detector_initialization.json')
        if init['sha256']!=initial or init['specification']!=specification('control'):
            raise ValueError('实际128维分支/共享检测器初始化不符')
        rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
        for row in rows:assert_alignment_weight(row['auxiliary'])
        if not any(row['auxiliary']['rq3_target_gradient_norm']>0 for row in rows):
            raise ValueError('没有真实目标原型梯度')
    elif read(entry/'termination.json')['initial_state_sha']!=initial:raise ValueError('UDA冷初始化改变')
    old=read(reference/'evaluation/target_val.json')['map_0.1_to_0.5']
    return dict(result,method=method,old_map=old,delta_to_old_pp=100*(result['map']-old))


def complete(entry,method,root,identity):
    paths=['config.json','runtime_config.json','seed_proof.json','metrics.jsonl','termination.json',
           'best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
           'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
    if method=='prototype':paths.append('detector_initialization.json')
    save(entry/'complete.json',dict(schema=SCHEMA,shift='S5',regime='uda',method=method,code=identity,
        acceptance_sha256=sha(root/'acceptance.json'),reference_complete=str(REFERENCES[method]/'complete.json'),
        reference_complete_sha256=sha(REFERENCES[method]/'complete.json'),
        artifacts={str(entry/n):sha(entry/n) for n in paths}))
    return audit(entry)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def state(root,phase,completed,**extra):
    save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,
        active={k:dict(pid=t['process'].pid,stage=t['stage'],gpu=t['lease']['index']) for k,t in ACTIVE.items()},**extra))


def main(args):
    root=args.output.resolve();data=args.data_root.resolve();identity=protocol.code_identity()
    indices=[int(x) for x in args.gpus.split(',')]
    if len(set(indices))!=len(indices) or not set(indices)<=set(range(4)):raise ValueError('GPU列表不合法')
    root.mkdir(parents=True,exist_ok=False);tmp=root/'tmp';tmp.mkdir();completed=[];configs={};reports={}
    save(root/'plan.json',dict(schema=SCHEMA,code=identity,shift='S5',direction='old_to_new',methods=list(REFERENCES),
        level_weights=list(LEVELS),sada_pseudo_label_threshold=THRESHOLD,prototype_specification=specification('control'),
        physical_gpus=indices,only_declared_parameter_changes=True,source_selection_only=True,
        note='paired_parameter_experiment_not_an_exact_paper_reproduction'))
    try:
        state(root,'audit_references',completed)
        old_results={}
        for method,entry in REFERENCES.items():
            old_results[method]=capacity.audit(entry) if method=='prototype' else protocol.audit_entry(entry)
            cfg=changed_config(read(entry/'config.json'),identity,tmp);assert_declared_changes(cfg,read(entry/'config.json'),method)
            configs[method]=cfg
        capacity.assert_pair(configs['prototype'],configs['uda'])
        save(root/'inputs.json',dict(code=identity,references=old_results,configurations=configs))
        initial=read(REFERENCES['uda']/'termination.json')['initial_state_sha'];pending=list(REFERENCES)
        while pending or ACTIVE:
            for method,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('真实GPU验收失败: '+method)
                report=read(t['entry']/'result.json');validate_probe(report,method,initial);reports[method]=report
                t['context'].__exit__(None,None,None);del ACTIVE[method]
            for method in list(pending):
                found=acquire(indices,data,root,12288,identity['commit'])
                if not found:break
                context,lease=found;entry=root/'probes'/method
                try:
                    entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[method])
                    process=spawn('probe',entry/'config.json',entry,lease,data,entry/'result.json')
                except BaseException:context.__exit__(None,None,None);raise
                ACTIVE[method]=dict(process=process,stage='probe',context=context,lease=lease,entry=entry);pending.remove(method)
            state(root,'real_data_acceptance',completed,pending=pending);time.sleep(3)
        if set(reports)!=set(REFERENCES):raise ValueError('缺少两项真实验收')
        policies=[memory_policy(x) for x in reports.values()]
        policy={key:max(p[key] for p in policies) for key in ('training_minimum_free_mib','evaluation_minimum_free_mib')}
        report_paths=[root/'probes'/m/n for m in REFERENCES for n in ('result.json','probe_gpu.json')]
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,inputs_sha256=sha(root/'inputs.json'),
            initial_detector_sha256=initial,resource_policy=policy,report_sha256={str(p):sha(p) for p in report_paths}))
        pending=list(REFERENCES);eval_pending=[]
        while pending or eval_pending or ACTIVE:
            for method,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('正式阶段失败: '+method+'/'+t['stage'])
                if t['stage']=='train':eval_pending.append(method)
                else:
                    completed.append(complete(t['entry'],method,root,identity));print('COMPLETE',json.dumps(completed[-1]),flush=True)
                t['context'].__exit__(None,None,None);del ACTIVE[method]
            for stage,jobs,minimum in [('evaluate',eval_pending,policy['evaluation_minimum_free_mib']),
                                       ('train',pending,policy['training_minimum_free_mib'])]:
                for method in list(jobs):
                    found=acquire(indices,data,root,minimum,identity['commit'])
                    if not found:break
                    context,lease=found;entry=root/'formal/S5'/method
                    try:
                        if stage=='train':entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[method])
                        process=spawn(stage,entry/'config.json',entry,lease,data)
                    except BaseException:context.__exit__(None,None,None);raise
                    ACTIVE[method]=dict(process=process,stage=stage,context=context,lease=lease,entry=entry);jobs.remove(method)
            state(root,'formal',completed,pending=pending,evaluation_pending=eval_pending,resource_policy=policy);time.sleep(3)
        if len(completed)!=2 or {x['method'] for x in completed}!=set(REFERENCES):raise ValueError('不是精确两项')
        rows={m:audit(root/'formal/S5'/m) for m in REFERENCES}
        capacity.assert_pair(read(root/'formal/S5/prototype/config.json'),read(root/'formal/S5/uda/config.json'))
        if read(root/'formal/S5/uda/termination.json')['initial_state_sha']!=read(root/'formal/S5/prototype/detector_initialization.json')['sha256']:
            raise ValueError('两项共享检测器冷初始化不配对')
        save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=2,results=rows,
            prototype_gain_pp=100*(rows['prototype']['map']-rows['uda']['map']),
            old_prototype_gain_pp=100*(rows['prototype']['old_map']-rows['uda']['old_map'])))
        state(root,'complete',list(rows.values()))
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
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='2,1,3,0')
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
