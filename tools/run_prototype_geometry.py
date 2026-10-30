"""四卡共享：固定源留出机制核验→真实验收→几何加权Prototype六方向。"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools.prototype_geometry_protocol import (SCHEMA,POLICY,VARIANT,config_for,assert_pair,
                                              mechanism_gate,verify_acceptance,audit_entry)
from tools.run_sada_baseline import acquire
from tools.run_prototype_capacity import audit as audit_control

read,save,sha=protocol.read,protocol.save,protocol.sha
ACTIVE={}


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def cleanup():
    for task in ACTIVE.values():
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    for task in ACTIVE.values():
        try:task['process'].wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(task['process'].pid,signal.SIGKILL);task['process'].wait()
        task['context'].__exit__(None,None,None)
    ACTIVE.clear()


def spawn(mode,entry,lease,data,root,control,source_pool):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE'):env.pop(key,None)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',mode,'--entry',str(entry),
         '--output-root',str(root),'--control-root',str(control),'--source-pool',str(source_pool),'--data-root',str(data)]
    with (entry/(mode+'.log')).open('x') as log:
        return subprocess.Popen(cmd,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def child(args):
    from tools import baseline_engine as engine
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import calibrate
    entry=args.entry;cfg=read(entry/'config.json');identity=protocol.code_identity()
    if identity!=cfg['baseline']['code']:raise ValueError('不是冻结clean代码')
    if sha(cfg['dataset']['json_file'])!=cfg['baseline']['source_manifest_sha256']:
        raise ValueError('source manifest改变')
    if cfg['baseline']['target_gt_used']:raise ValueError('本实验禁止target GT')
    if args.child!='calibrate':
        if sha(cfg['baseline']['target_manifest'])!=cfg['baseline']['target_manifest_sha256']:
            raise ValueError('无标注target manifest改变')
        assert_pair(cfg,read(args.control_root/'formal/control'/cfg['baseline']['shift']/'config.json'))
        if args.child!='probe':verify_acceptance(args.output_root,cfg)
    os.chdir(args.data_root)
    proof=first_cuda_guard()
    if args.child=='calibrate':
        calibrate(args.control_root/'formal/control'/cfg['baseline']['shift'],args.source_pool,entry,identity)
    elif args.child=='probe':engine.probe(cfg,entry/'result.json')
    elif args.child=='train':
        save(entry/'supervision_audit.json',dict(schema=SCHEMA,target_gt_used=False,
            geometry_policy=POLICY,source_validation_only=True,prepared_cache_sha256=cfg['prototype_capacity']['prepared_cache_sha256']))
        engine.train(cfg,entry)
    else:engine.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(entry/'config.json')))


def finish(root,task,control,identity):
    e=task['entry'];reference=control/'formal/control'/task['shift']/'complete.json'
    paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','supervision_audit.json',
        'train.log','evaluate.log','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
    save(e/'complete.json',dict(schema=SCHEMA,shift=task['shift'],regime='uda',variant=VARIANT,code=identity,
        control_complete=str(reference),control_complete_sha256=sha(reference),
        acceptance_sha256=sha(root/'acceptance.json'),artifacts={str(e/p):sha(e/p) for p in paths}))
    return audit_entry(e)


def check_probe(report,reference):
    if report['status']!='passed' or report['rng_roundtrip']['status']!='passed':
        raise ValueError('真实验收/完整随机状态恢复未通过')
    if report['initial_detector_sha']!=read(reference/'detector_initialization.json')['sha256']:
        raise ValueError('真实验收与原版冷初始化不配对')
    effective=False
    for phase in report['phases'].values():
        aux=phase['auxiliary']
        if phase['target_gt_losses']!=[0.,0.,0.] or not phase['branch_parameters_updated']:
            raise ValueError('目标GT隔离或辅助分支优化失败')
        if not all(math.isfinite(aux.get(k,0)) and aux.get(k,0)>0 for k in
                   ('rq3_source_loss','rq3_alignment_loss','rq3_selected','rq3_target_gradient_norm')):
            raise ValueError('真实候选/源监督/目标梯度未生效')
        effective |= aux.get('rq3_geometry_weighted_candidates',0)>0
    if not effective:raise ValueError('几何权重全部恒等，没有真实作用')


def main(args):
    root=args.output_root.resolve();control=args.control_root.resolve();data=args.data_root.resolve()
    args.source_pool=args.source_pool.resolve();args.tmpdir=args.tmpdir.resolve()
    indices=[int(x) for x in args.gpus.split(',')]
    if set(indices)!={0,1,2,3} or len(indices)!=4:raise ValueError('本轮只使用明确授权的四张物理卡')
    identity=protocol.code_identity();root.mkdir(parents=True,exist_ok=False)
    args.tmpdir.mkdir(parents=True,exist_ok=True)
    shifts=['S'+str(i) for i in range(1,7)]
    save(root/'state.json',dict(phase='input_audit',pid=os.getpid(),active={},completed=[]))
    save(root/'plan.json',dict(schema=SCHEMA,variant=VARIANT,shifts=shifts,gpus=indices,
        sequence=['source_calibration_and_mechanism','real_gpu_acceptance','formal_six'],policy=POLICY,
        teacher='qwen3_vl_8b',seed=123,batch=2,source_only_gate=True,old_control_reused=True))
    # 在任何新GPU训练前独立重审六个原版；不读取新方法target结果选择方案。
    references={s:audit_control(control/'formal/control'/s) for s in shifts}
    from tools.calibrate_prototype_geometry import cpu_source_preflight
    preflight={s:cpu_source_preflight(control/'formal/control'/s) for s in shifts}
    pool=read(args.source_pool)
    if pool['target_data_used']:raise ValueError('机制核验不能读取target GT')
    save(root/'inputs.json',dict(code=identity,control_root=str(control),references=references,
        cpu_source_preflight=preflight,source_pool=str(args.source_pool),source_pool_sha256=sha(args.source_pool)))
    phase='calibrate';pending=list(shifts);reports={};completed=[];minimum=12288
    try:
        while True:
            for key,task in list(ACTIVE.items()):
                process=task['process']
                if process.poll() is None:continue
                if process.returncode:raise RuntimeError('阶段失败 '+key+': '+str(task['entry']/(task['stage']+'.log')))
                e=task['entry'];s=task['shift']
                if task['stage']=='calibrate':reports[s]=read(e/'calibration.json')
                elif task['stage']=='probe':
                    report=read(e/'result.json');check_probe(report,control/'formal/control/S1')
                    minimum=max(12288,math.ceil(report['peak_reserved_mib']+4096))
                    save(root/'acceptance.json',dict(status='passed',code=identity,batch=2,seed=123,
                        mechanism_gate_sha256=sha(root/'mechanism_gate.json'),
                        reports={str(e/'result.json'):sha(e/'result.json')},minimum_free_mib=minimum))
                elif task['stage']=='train':
                    task['stage']='evaluate'
                    task['process']=spawn('evaluate',e,task['lease'],data,root,control,args.source_pool)
                    continue
                else:
                    completed.append(finish(root,task,control,identity))
                    print('GEOMETRY_COMPLETE',s,completed[-1]['map'],flush=True)
                task['context'].__exit__(None,None,None);del ACTIVE[key]
            if not pending and not ACTIVE:
                if phase=='calibrate':
                    gate=mechanism_gate(reports);save(root/'mechanism_gate.json',gate)
                    if gate['status']!='passed':
                        save(root/'state.json',dict(phase='stopped_no_source_signal',pid=os.getpid(),active={},completed=[],gate=gate))
                        print('SOURCE_MECHANISM_REJECTED',gate,flush=True);return
                    phase='probe';pending=['S1']
                elif phase=='probe':phase='train';pending=list(shifts)
                else:break
            for s in list(pending):
                found=acquire(indices,data,root,minimum,identity['commit'])
                if not found:break
                context,lease=found
                try:
                    e=root/('calibration' if phase=='calibrate' else 'probes' if phase=='probe' else 'formal')/(VARIANT if phase=='train' else '')/s
                    e.mkdir(parents=True,exist_ok=False)
                    original=read(control/'formal/control'/s/'config.json')
                    if phase=='calibrate':
                        cfg=deepcopy(original);cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(args.tmpdir)
                    else:
                        cfg=config_for(original,root/'calibration'/s/'calibration.json',identity,args.tmpdir)
                        assert_pair(cfg,original)
                    save(e/'config.json',cfg)
                    process=spawn(phase,e,lease,data,root,control,args.source_pool)
                except BaseException:
                    context.__exit__(None,None,None);raise
                key=phase+'_'+s
                ACTIVE[key]=dict(stage=phase,entry=e,shift=s,context=context,lease=lease,process=process)
                pending.remove(s)
            save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=v['process'].pid,gpu=v['lease']['index'],stage=v['stage']) for k,v in ACTIVE.items()}))
            time.sleep(5)
        final=[audit_entry(root/'formal'/VARIANT/s) for s in shifts]
        save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,results=final,
            old_controls=references,mechanism_gate_sha256=sha(root/'mechanism_gate.json')))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),active={},completed=final))
    finally:cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output-root','control-root','source-pool','entry'):
        p.add_argument('--'+name,type=Path,required=name!='entry')
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--tmpdir',type=Path,default=Path('/vepfs/vlms/tmp/prototype_geometry_1030'))
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--child',choices=['calibrate','probe','train','evaluate'])
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if a.child:child(a)
        else:main(a)
    except BaseException as error:
        state_path=a.output_root/'state.json'
        if state_path.exists() and not a.child and read(state_path).get('pid')==os.getpid():
            save(a.output_root/'error.json',dict(type=type(error).__name__,error=str(error)))
            state=read(a.output_root/'state.json');state.update(phase='stopped' if isinstance(error,SystemExit) else 'failed',error=str(error),active={})
            save(a.output_root/'state.json',state)
        raise
