"""原版Prototype六方向消融：只关闭SADA对抗对齐，保留真实目标视频流。"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools import run_prototype_capacity as capacity
from tools.run_sada_baseline import acquire,spawn
from tools.run_s5_paper_params import validate_probe
from tools.run_prototype_wide import assert_alignment_weight,finish_process,memory_policy
from libs.models.SADA.prototype_capacity import specification

SCHEMA='prototype-without-adversarial-alignment-v1'
CONTROL=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')
SHIFTS=tuple('S%d'%i for i in range(1,7))
ACTIVE={}
read,save,sha=protocol.read,protocol.save,protocol.sha


def changed_config(old,identity,tmp):
    cfg=deepcopy(old)
    cfg['model']['adv_method']=['None']
    cfg['baseline'].update(code=identity,tmpdir=str(tmp))
    return cfg


def assert_change(cfg,old):
    if old['baseline']['shift'] not in SHIFTS or old['baseline']['regime']!='uda':
        raise ValueError('必须保留真实目标域UDA数据流')
    if old['seed']!=123 or old['loader']['batch_size']!=2:
        raise ValueError('只能使用seed123/batch2原版对照')
    cap=old['prototype_capacity']
    if cap['variant']!='control' or cap['specification']!=specification('control'):
        raise ValueError('必须是128维单层单原型、目标权重0.1')
    if 'conditional_rev' not in old['model']['adv_method']:
        raise ValueError('原对照没有开启SADA对抗')
    if cfg!=changed_config(old,cfg['baseline']['code'],cfg['baseline']['tmpdir']):
        raise ValueError('除对抗开关/运行身份外发生变化')


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json')
    shift=record['shift'];reference=CONTROL/shift
    if sha(reference/'complete.json')!=record['control_complete_sha256']:
        raise ValueError('原版Prototype完成SHA改变')
    assert_change(cfg,read(reference/'config.json'))
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    cap=cfg['prototype_capacity']
    if sha(cap['prepared_cache'])!=cap['prepared_cache_sha256']:raise ValueError('教师缓存改变')
    initial=read(entry/'detector_initialization.json')
    if (initial['specification']!=specification('control') or
            initial['sha256']!=read(BASE/shift/'uda/termination.json')['initial_state_sha']):
        raise ValueError('实际辅助结构或共享检测器冷初始化改变')
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    for row in rows:assert_alignment_weight(row['auxiliary'])
    if not any(r['auxiliary']['rq3_target_gradient_norm']>0 and r['auxiliary']['rq3_selected']>0 for r in rows):
        raise ValueError('真实目标视频没有产生原型梯度')
    import torch
    # 原判别器保留以维持相同冷初始化；无梯度就不应产生Adam状态。
    for name in ('best','latest'):
        saved=torch.load(entry/(name+'.pt'),map_location='cpu')
        if saved['optimizers'][2]['state']:raise ValueError('关闭后仍更新了域判别器')
        del saved
    references={m:BASE/shift/m for m in ('so','uda')};references['prototype']=reference
    old_maps={m:read(p/'evaluation/target_val.json')['map_0.1_to_0.5'] for m,p in references.items()}
    return dict(result,method='prototype_no_adversary',reference_maps=old_maps,
                delta_pp={m:100*(result['map']-value) for m,value in old_maps.items()})


def complete(entry,root,identity):
    cfg=read(entry/'config.json');shift=cfg['baseline']['shift']
    paths=('config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
        'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
    save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',method='prototype_no_adversary',
        code=identity,acceptance_sha256=sha(root/'acceptance.json'),
        control_complete=str(CONTROL/shift/'complete.json'),control_complete_sha256=sha(CONTROL/shift/'complete.json'),
        target_gt_used=False,real_target_stream=True,adversarial_alignment=False,
        artifacts={str(entry/n):sha(entry/n) for n in paths}))
    return audit(entry)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def state(root,phase,completed,**extra):
    save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,
        active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()},**extra))


def main(args):
    root=args.output.resolve();data=args.data_root.resolve();identity=protocol.code_identity()
    indices=[int(x) for x in args.gpus.split(',')]
    if len(set(indices))!=len(indices) or not indices or not set(indices)<=set(range(4)):
        raise ValueError('GPU列表不合法')
    root.mkdir(parents=True,exist_ok=False);tmp=root/'tmp';tmp.mkdir();completed=[]
    save(root/'plan.json',dict(schema=SCHEMA,shifts=SHIFTS,new_training_count=6,variant='control',
        specification=specification('control'),adv_method=['None'],target_data_stream='real-unlabeled-target',
        seed=123,batch=2,code=identity,physical_gpus=indices,target_gt_used=False,
        only_scientific_change='disable-adversarial-alignment',no_automatic_new_experiments=True))
    try:
        state(root,'audit_references',completed);configs={};references={}
        for shift in SHIFTS:
            references[shift]=dict(prototype=capacity.audit(CONTROL/shift),
                so=protocol.audit_entry(BASE/shift/'so'),uda=protocol.audit_entry(BASE/shift/'uda'))
            old=read(CONTROL/shift/'config.json');cfg=changed_config(old,identity,tmp)
            assert_change(cfg,old);configs[shift]=cfg
        save(root/'inputs.json',dict(code=identity,references=references,configurations=configs))
        probe=root/'probes/control';probe.mkdir(parents=True,exist_ok=False);save(probe/'config.json',configs['S1'])
        state(root,'waiting_gpu_for_acceptance',completed)
        while not (found:=acquire(indices,data,root,12288,identity['commit'])):time.sleep(10)
        context,lease=found
        try:
            process=spawn('probe',probe/'config.json',probe,lease,data,probe/'result.json')
        except BaseException:context.__exit__(None,None,None);raise
        ACTIVE['acceptance']=dict(process=process,context=context,lease=lease,stage='probe')
        state(root,'real_data_acceptance',completed)
        if process.wait():raise RuntimeError('无对抗Prototype真实验收失败')
        report=read(probe/'result.json')
        validate_probe(report,'prototype',read(BASE/'S1/uda/termination.json')['initial_state_sha'])
        context.__exit__(None,None,None);del ACTIVE['acceptance']
        policy=memory_policy(report)
        save(root/'acceptance.json',dict(status='passed',batch=2,seed=123,code=identity,
            inputs_sha256=sha(root/'inputs.json'),resource_policy=policy,
            report_sha256={str(probe/n):sha(probe/n) for n in ('result.json','probe_gpu.json')}))
        pending=list(SHIFTS);evaluations=[]
        while pending or evaluations or ACTIVE:
            for shift,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('条目阶段失败: '+shift+'/'+t['stage'])
                t['context'].__exit__(None,None,None);del ACTIVE[shift]
                if t['stage']=='train':evaluations.append(shift)
                else:
                    completed.append(complete(t['entry'],root,identity));print('COMPLETE',json.dumps(completed[-1]),flush=True)
            for stage,queue,minimum in [('evaluate',evaluations,policy['evaluation_minimum_free_mib']),
                                        ('train',pending,policy['training_minimum_free_mib'])]:
                for shift in list(queue):
                    found=acquire(indices,data,root,minimum,identity['commit'])
                    if not found:break
                    context,lease=found;entry=root/'formal'/shift/'prototype'
                    try:
                        if stage=='train':entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[shift])
                        process=spawn(stage,entry/'config.json',entry,lease,data)
                    except BaseException:context.__exit__(None,None,None);raise
                    ACTIVE[shift]=dict(process=process,context=context,lease=lease,entry=entry,stage=stage);queue.remove(shift)
            state(root,'formal',completed,pending=pending,evaluation_pending=evaluations);time.sleep(3)
        if len(completed)!=6 or {r['shift'] for r in completed}!=set(SHIFTS):raise ValueError('不是精确六方向')
        save(root/'final_audit.json',dict(status='passed',count=6,schema=SCHEMA,results=completed,
            mean_map=sum(r['map'] for r in completed)/6,
            mean_delta_pp={m:sum(r['delta_pp'][m] for r in completed)/6 for m in ('so','uda','prototype')}))
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
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='0,2,3,1')
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
