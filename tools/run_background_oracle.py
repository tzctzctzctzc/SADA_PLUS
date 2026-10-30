"""只跑O六方向；每项真实验收后fresh训练，保留原Prototype作为已完成对照。"""
import argparse
from pathlib import Path
import os
import signal
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools.background_oracle_protocol import prepare,audit
from libs.models.SADA.background_oracle import SCHEMA,POLICY
from tools.run_sada_baseline import acquire,spawn

ACTIVE={}


def stop(signum,frame):
    for task in list(ACTIVE.values()):
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)


def main(args):
    root=args.output_root.resolve();data=args.data_root.resolve();control=args.control.resolve()
    root.mkdir(exist_ok=False);identity=p.code_identity();(root/'tmp').mkdir()
    p.save(root/'state.json',dict(phase='cpu_preflight',pid=os.getpid(),active={},completed=[]))
    os.chdir(data);configs=prepare(root,control,identity);os.chdir(CODE)
    p.save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,shifts=list(configs),
        variant='O-only',new_formal_runs=6,random_control=False,reference_retrained=False,
        target_gt_used=True,target_gt_scope=POLICY['scope'],seed=123,batch=2,
        control=str(control),gpus=args.gpus,minimum_free_mib=16384))
    pending=list(configs);completed=[]
    try:
        while pending or ACTIVE:
            for shift,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError(shift+' '+task['stage']+' failed; outputs preserved')
                entry=task['entry'];cfg=configs[shift]
                if task['stage']=='probe':
                    report=p.read(entry/'probe_result.json')
                    actual=report['phases']['5'].get('background_oracle',{})
                    if sum(v for k,v in actual.items() if k.startswith('oracle_bg_removed_'))<=0:raise ValueError('真实验收未移除GT正点')
                    if report['rng_roundtrip']['status']!='passed':raise ValueError('GPU状态恢复验收不通过')
                    if report['initial_detector_sha']!=p.read(control/'formal/control'/shift/'detector_initialization.json')['sha256']:raise ValueError('冷初始化不配对')
                    for phase in report['phases'].values():
                        if not phase['branch_parameters_updated']:raise ValueError('原Prototype分支没有实际更新')
                    acceptance=dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
                        target_gt_used=True,oracle_policy=POLICY,config_sha256=p.sha(entry/'config.json'),
                        report=str(entry/'probe_result.json'),report_sha256=p.sha(entry/'probe_result.json'))
                    p.save(entry/'acceptance.json',acceptance)
                    task['stage']='train';task['process']=spawn('train',entry/'config.json',entry,task['lease'],data)
                    print('FORMAL_STARTED',shift,task['process'].pid,task['lease']['index'],flush=True)
                    continue
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',entry/'config.json',entry,task['lease'],data);continue
                paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
                    'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','probe_gpu.json',
                    'probe_result.json','acceptance.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
                reference=control/'formal/control'/shift/'complete.json'
                p.save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant='O',code=identity,
                    target_gt_used=True,oracle_policy=POLICY,control_complete=str(reference),control_complete_sha256=p.sha(reference),
                    gpu_acceptance=str(entry/'acceptance.json'),acceptance_sha256=p.sha(entry/'acceptance.json'),
                    artifacts={str(entry/name):p.sha(entry/name) for name in paths}))
                completed.append(audit(entry));task['context'].__exit__(None,None,None);del ACTIVE[shift]
                print('ORACLE_COMPLETE',shift,completed[-1]['map'],flush=True)
            for shift in list(pending):
                found=acquire(args.gpus,data,root,16384,identity['commit'])
                if not found:break
                context,lease=found;pending.remove(shift);entry=root/'formal'/shift;entry.mkdir(parents=True,exist_ok=False)
                p.save(entry/'config.json',configs[shift])
                ACTIVE[shift]=dict(entry=entry,stage='probe',context=context,lease=lease,
                    process=spawn('probe',entry/'config.json',entry,lease,data,entry/'probe_result.json'))
            p.save(root/'state.json',dict(phase='oracle_only',pid=os.getpid(),completed=completed,pending=pending,
                active={s:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for s,t in ACTIVE.items()},target_gt_used=True))
            time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise
    if {x['shift'] for x in completed}!=set(configs):raise ValueError('不是精确六方向')
    p.save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,target_gt_used=True,results=completed))
    p.save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={},target_gt_used=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,3,1,2');args=parser.parse_args();args.gpus=[int(x) for x in args.gpus.split(',')]
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
