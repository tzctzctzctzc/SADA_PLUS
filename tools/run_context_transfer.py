"""源GT完整性检查→六向真实验收→四卡共享fresh正式六方向。"""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools import run_phase_alignment as owned
from tools.run_sada_baseline import acquire
from tools.context_transfer_protocol import prepare,verify_probe,audit
from libs.models.SADA.context_transfer import SCHEMA,POLICY


def observed(root,phase,completed,pending):
    p.save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,pending=pending,
        active={s:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for s,t in owned.ACTIVE.items()},
        target_gt_used=False))


def spawn(stage,entry,lease,data):
    cfg=p.read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ('RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE'):env.pop(name,None)
    with (entry/(stage+'.log')).open('x') as out:
        return subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),'--child',stage,
            '--entry',str(entry),'--data-root',str(data)],cwd=CODE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)


def launch(shift,stage,cfg,args,identity):
    entry=args.output_root/'formal'/shift
    minimum=args.minimum_free_mib if stage=='probe' else p.read(entry/'acceptance.json')['minimum_free_mib']
    found=acquire(args.gpus,args.data_root,args.output_root,minimum,identity['commit'])
    if not found:return False
    context,lease=found
    try:
        if stage=='probe':entry.mkdir(parents=True,exist_ok=False);p.save(entry/'config.json',cfg)
        else:
            if p.read(entry/'config.json')!=cfg:raise ValueError('正式前配置漂移')
            task=dict(entry=entry,lease=lease)
            owned.formal_admission(task,p.read(entry/'probe_result.json'),args.output_root,identity,args.minimum_free_mib)
            lease=task['lease']
        process=spawn(stage,entry,lease,args.data_root)
    except BaseException:
        context.__exit__(None,None,None);raise
    owned.ACTIVE[shift]=dict(process=process,stage=stage,entry=entry,context=context,lease=lease)
    print('CONTEXT_STARTED',shift,stage,process.pid,lease['index'],flush=True);return True


def child(args):
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import probe,train,evaluate
    cfg=p.read(args.entry/'config.json')
    if p.code_identity()!=cfg['baseline']['code']:raise ValueError('子进程代码身份漂移')
    for path,digest in [(cfg['dataset']['json_file'],cfg['baseline']['source_manifest_sha256']),
                        (cfg['baseline']['target_manifest'],cfg['baseline']['target_manifest_sha256'])]:
        if p.sha(path)!=digest:raise ValueError('源或无GT目标输入改变')
    os.chdir(args.data_root);guard=first_cuda_guard()
    if args.child=='probe':probe(cfg,args.entry/'probe_result.json')
    elif args.child=='train':train(cfg,args.entry)
    else:evaluate(cfg,args.entry,cfg['baseline']['annotation_for_offline_evaluation'])
    p.save(args.entry/(args.child+'_gpu.json'),dict(guard=guard,guard_sha256=p.sha(guard),config_sha256=p.sha(args.entry/'config.json')))


def main(args):
    root=args.output_root;identity=p.code_identity();root.mkdir(exist_ok=False);(root/'tmp').mkdir()
    observed(root,'input_audit',[],[]);os.chdir(args.data_root)
    configs=prepare(root,args.control,identity);os.chdir(CODE)
    p.save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,control=str(args.control),
        reference_final_audit_sha256=p.sha(args.control/'final_audit.json'),gpus=args.gpus,
        minimum_free_mib=args.minimum_free_mib,shifts=list(configs),new_formal_runs=6,
        extra_compute='one_mixed_input_shared_detector_forward_per_active_batch',retrain_control=False,
        exploratory=True,target_gt_used=False,scientific_ablations=False))
    if args.preflight_only:observed(root,'cpu_preflight_passed',[],[]);return
    pending=list(configs);accepted=[];completed=[]
    try:
        while pending or owned.ACTIVE:
            for shift,task in list(owned.ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError(shift+'真实验收技术失败，保留现场')
                entry=task['entry'];report=p.read(entry/'probe_result.json')
                initial=p.read(args.control/'formal/control'/shift/'detector_initialization.json')['sha256']
                verify_probe(report,initial)
                p.save(entry/'acceptance.json',dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
                    report_sha256=p.sha(entry/'probe_result.json'),source_gate_sha256=p.sha(root/'source_gate.json'),
                    minimum_free_mib=max(args.minimum_free_mib,math.ceil(report['peak_reserved_mib']+2048))))
                accepted.append(shift);task['context'].__exit__(None,None,None);del owned.ACTIVE[shift]
                print('CONTEXT_ACCEPTANCE_PASSED',shift,flush=True)
            for shift in list(pending):
                if not launch(shift,'probe',configs[shift],args,identity):break
                pending.remove(shift)
            observed(root,'real_acceptance',completed,pending);time.sleep(5)
        if set(accepted)!=set(configs):raise ValueError('未完成精确六向共同验收')
        p.save(root/'acceptance_all6.json',dict(status='passed',code=identity,
            acceptance={str(root/'formal'/s/'acceptance.json'):p.sha(root/'formal'/s/'acceptance.json') for s in accepted}))
        pending=list(configs)
        while pending or owned.ACTIVE:
            for shift,task in list(owned.ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError(shift+' '+task['stage']+'技术失败，保留现场')
                entry=task['entry']
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',entry,task['lease'],args.data_root);continue
                paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
                    'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','probe_gpu.json','probe_result.json',
                    'acceptance.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
                ref=args.control/'formal/control'/shift/'complete.json'
                p.save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant='source_context_transfer',code=identity,
                    target_gt_used=False,control_complete=str(ref),control_complete_sha256=p.sha(ref),
                    acceptance_sha256=p.sha(entry/'acceptance.json'),source_gate=str(root/'source_gate.json'),
                    source_gate_sha256=p.sha(root/'source_gate.json'),group_acceptance=str(root/'acceptance_all6.json'),
                    group_acceptance_sha256=p.sha(root/'acceptance_all6.json'),artifacts={str(entry/n):p.sha(entry/n) for n in paths}))
                os.chdir(args.data_root);completed.append(audit(entry));os.chdir(CODE)
                task['context'].__exit__(None,None,None);del owned.ACTIVE[shift]
                print('CONTEXT_COMPLETE',shift,completed[-1]['map'],flush=True)
            for shift in list(pending):
                if not launch(shift,'train',configs[shift],args,identity):break
                pending.remove(shift)
            observed(root,'formal',completed,pending);time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        observed(root,'failed',completed,pending);owned.terminate_owned();raise
    if {r['shift'] for r in completed}!=set(configs):raise ValueError('不是精确六向完整方法')
    rows=[]
    for result in sorted(completed,key=lambda x:x['shift']):
        ref=args.control/'formal/control'/result['shift'];done=p.read(ref/'complete.json')
        if p.sha(ref/'evaluation/target_val.json')!=done['artifacts'][str(ref/'evaluation/target_val.json')]:raise ValueError('对照评估改变')
        old=p.read(ref/'evaluation/target_val.json')['map_0.1_to_0.5']
        rows.append(dict(shift=result['shift'],control=old,new=result['map'],delta_pp=100*(result['map']-old)))
    p.save(root/'comparison.json',dict(results=rows,mean_delta_pp=sum(r['delta_pp'] for r in rows)/6,
        target_gt_used_for_training=False,exploratory=True))
    p.save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,results=completed))
    observed(root,'complete',completed,[])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--entry',type=Path)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,1,2,3');parser.add_argument('--minimum-free-mib',type=int,default=8192)
    parser.add_argument('--preflight-only',action='store_true');parser.add_argument('--child',choices=['probe','train','evaluate'])
    args=parser.parse_args();args.gpus=[int(x) for x in args.gpus.split(',')]
    signal.signal(signal.SIGTERM,owned.stop);signal.signal(signal.SIGINT,owned.stop)
    if args.child:child(args)
    else:
        if args.output_root is None:parser.error('--output-root required')
        main(args)
