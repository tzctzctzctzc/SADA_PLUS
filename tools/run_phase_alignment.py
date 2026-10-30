"""四卡共享的有限六方向实验：源坐标核验→真实验收→fresh训练→源选模全审。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as p
from tools.phase_alignment_protocol import prepare,audit,diagnose,source_decision,SOURCE_GATE
from tools.run_sada_baseline import acquire
from libs.models.SADA.phase_alignment import SCHEMA,POLICY

ACTIVE={}


def stop(signum,frame):
    for task in list(ACTIVE.values()):
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    raise SystemExit(128+signum)


def spawn(mode,entry,lease,data,reference):
    cfg=p.read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(name,None)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',mode,'--entry',str(entry),
             '--data-root',str(data),'--reference',str(reference)]
    with (entry/(mode+'.log')).open('x') as stream:
        return subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def child(args):
    from tools.baseline_gpu import first_cuda_guard
    from tools.baseline_engine import probe,train,evaluate
    cfg=p.read(args.entry/'config.json')
    if p.code_identity()!=cfg['baseline']['code']:raise ValueError('运行不是冻结clean代码')
    for path,digest in [(cfg['dataset']['json_file'],cfg['baseline']['source_manifest_sha256']),
                        (cfg['baseline']['target_manifest'],cfg['baseline']['target_manifest_sha256'])]:
        if p.sha(path)!=digest:raise ValueError('冻结source/target输入改变')
    os.chdir(args.data_root);proof=first_cuda_guard()
    if args.child=='sourcecheck':diagnose(cfg,args.reference,args.entry/'source_report.json')
    elif args.child=='probe':probe(cfg,args.entry/'probe_result.json')
    elif args.child=='train':train(cfg,args.entry)
    else:evaluate(cfg,args.entry,cfg['baseline']['annotation_for_offline_evaluation'])
    p.save(args.entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=p.sha(proof),config_sha256=p.sha(args.entry/'config.json')))


def observed(root,phase,completed,pending):
    p.save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,pending=pending,
        active={s:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for s,t in ACTIVE.items()},
        target_gt_used=False))


def launch(shift,stage,entry,cfg,args,identity):
    found=acquire(args.gpus,args.data_root,args.output_root,args.minimum_free_mib,identity['commit'])
    if not found:return False
    context,lease=found
    try:
        entry.mkdir(parents=True,exist_ok=False);p.save(entry/'config.json',cfg)
        process=spawn(stage,entry,lease,args.data_root,args.control/'formal/control'/shift)
    except BaseException:
        context.__exit__(None,None,None);raise
    ACTIVE[shift]=dict(entry=entry,stage=stage,context=context,lease=lease,process=process)
    return True


def main(args):
    root=args.output_root;identity=p.code_identity();root.mkdir(exist_ok=False);(root/'tmp').mkdir()
    observed(root,'input_audit',[],[]);os.chdir(args.data_root)
    configs=prepare(root,args.control,identity);os.chdir(CODE)
    p.save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,source_gate=SOURCE_GATE,code=identity,
        gpus=args.gpus,minimum_free_mib=args.minimum_free_mib,control=str(args.control),
        shifts=list(configs),new_formal_runs=6,retrain_control=False,target_gt_used=False,
        qualification='exploratory_cross_task_transfer_not_proven_novelty',
        reference_final_audit_sha256=p.sha(args.control/'final_audit.json')))
    pending=list(configs);reports=[];completed=[]
    try:
        while pending or ACTIVE:
            for shift,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError(shift+' sourcecheck failed; outputs preserved')
                reports.append(p.read(task['entry']/'source_report.json'))
                task['context'].__exit__(None,None,None);del ACTIVE[shift]
            for shift in list(pending):
                if not launch(shift,'sourcecheck',root/'sourcecheck'/shift,configs[shift],args,identity):break
                pending.remove(shift)
            observed(root,'source_mechanism',[],pending);time.sleep(3)
        gate=source_decision(reports)
        gate['reports']={str(root/'sourcecheck'/r['shift']/'source_report.json'):p.sha(root/'sourcecheck'/r['shift']/'source_report.json') for r in reports}
        p.save(root/'source_gate.json',gate)
        if gate['status']!='passed':
            observed(root,'stopped_source_gate',[],[]);return
        pending=list(configs)
        while pending or ACTIVE:
            for shift,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError(shift+' '+task['stage']+' failed; outputs preserved')
                entry=task['entry'];cfg=configs[shift]
                if task['stage']=='probe':
                    report=p.read(entry/'probe_result.json')
                    if report['rng_roundtrip']['status']!='passed':raise ValueError('GPU完整恢复核验失败')
                    ref=args.control/'formal/control'/shift
                    if report['initial_detector_sha']!=p.read(ref/'detector_initialization.json')['sha256']:
                        raise ValueError('共同检测器冷初始化不配对')
                    phase=report['phases']['5']
                    if not phase['branch_parameters_updated'] or phase['auxiliary']['target_gradient_norm']<=0:
                        raise ValueError('原Prototype没有实际有效更新')
                    if sum(phase['phase_weight_separation'])<=0:raise ValueError('角色D仍然恒等')
                    if sum(v for k,v in phase['phase_alignment'].items() if k.startswith('phase_common_roles_'))<=0:
                        raise ValueError('没有实际跨域角色对齐')
                    p.save(entry/'acceptance.json',dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
                        source_gate_sha256=p.sha(root/'source_gate.json'),report_sha256=p.sha(entry/'probe_result.json'),
                        target_gt_used=False))
                    task['stage']='train';task['process']=spawn('train',entry,task['lease'],args.data_root,ref)
                    print('FORMAL_STARTED',shift,task['process'].pid,task['lease']['index'],flush=True);continue
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',entry,task['lease'],args.data_root,args.control/'formal/control'/shift);continue
                paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
                    'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','probe_gpu.json','probe_result.json',
                    'acceptance.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
                reference=args.control/'formal/control'/shift/'complete.json'
                p.save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant='relative_phase',code=identity,
                    target_gt_used=False,control_complete=str(reference),control_complete_sha256=p.sha(reference),
                    acceptance_sha256=p.sha(entry/'acceptance.json'),source_gate=str(root/'source_gate.json'),
                    source_gate_sha256=p.sha(root/'source_gate.json'),artifacts={str(entry/n):p.sha(entry/n) for n in paths}))
                os.chdir(args.data_root);completed.append(audit(entry));os.chdir(CODE)
                task['context'].__exit__(None,None,None);del ACTIVE[shift]
                print('PHASE_COMPLETE',shift,completed[-1]['map'],flush=True)
            for shift in list(pending):
                if not launch(shift,'probe',root/'formal'/shift,configs[shift],args,identity):break
                pending.remove(shift)
            observed(root,'formal',completed,pending);time.sleep(5)
    except BaseException as error:
        p.save(root/'error.json',dict(error=repr(error),completed=completed,pending=pending))
        observed(root,'failed',completed,pending)
        for task in ACTIVE.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise
    if {r['shift'] for r in completed}!=set(configs):raise ValueError('不是精确六方向')
    comparison=[]
    for row in sorted(completed,key=lambda r:r['shift']):
        old=p.read(args.control/'formal/control'/row['shift']/'evaluation/target_val.json')['map_0.1_to_0.5']
        comparison.append(dict(shift=row['shift'],control=old,new=row['map'],delta_pp=100*(row['map']-old)))
    p.save(root/'comparison.json',dict(results=comparison,mean_delta_pp=sum(r['delta_pp'] for r in comparison)/6,
        target_gt_used_for_training=False,exploratory=True))
    p.save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,results=completed))
    observed(root,'complete',completed,[])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path);parser.add_argument('--entry',type=Path);parser.add_argument('--reference',type=Path)
    parser.add_argument('--control',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--gpus',default='0,1,2,3');parser.add_argument('--minimum-free-mib',type=int,default=8192)
    parser.add_argument('--child',choices=['sourcecheck','probe','train','evaluate'])
    args=parser.parse_args();args.gpus=[int(v) for v in args.gpus.split(',')]
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.child:child(args)
    else:
        if args.output_root is None:parser.error('--output-root required')
        main(args)
