"""候选更新一次→重新抽八帧/8B评分→六方向fresh原版Prototype；保留静态对照。"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity,assert_no_target_gt
from tools.prototype_refresh import (SCHEMA,POLICY,SHIFTS,BASELINE,CONTROL,DIAGNOSTIC,
    prepare,freeze_scores,validate_cache,config_for,assert_pair,audit_entry)

ACTIVE={}


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def cleanup():
    for t in ACTIVE.values():
        if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
    for t in ACTIVE.values():
        try:t['process'].wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
        if t.get('context'):t['context'].__exit__(None,None,None)
    ACTIVE.clear()


def run_cpu(command,root,name,env):
    with (root/(name+'.log')).open('x') as f:
        p=subprocess.Popen(command,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ACTIVE[name]=dict(process=p)
    if p.wait():raise RuntimeError('CPU准备失败: '+name)
    del ACTIVE[name]


def launch_student(mode,entry,lease,data):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(name,None)
    cmd=[sys.executable,'-B',str(CODE/'tools/run_prototype_refresh.py'),'--child',mode,
        '--output-root',str(entry.parents[2]),'--entry',str(entry),'--data-root',str(data)]
    with (entry/(mode+'.log')).open('x') as f:
        return subprocess.Popen(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)


def verify_acceptance(root,cfg):
    a=read(root/'acceptance.json')
    if (a['status']!='passed' or a['code']!=cfg['baseline']['code']
            or a['inputs_sha256']!=sha(root/'inputs.json')
            or a['prepared_audit_sha256']!=sha(root/'prepared/audit.json')
            or a['report_sha256']!=sha(root/'probes/single_refresh/S1/result.json')):
        raise ValueError('验收不绑定当前代码、候选和真实报告')


def child(args):
    from tools import baseline_engine as engine
    from tools.baseline_gpu import first_cuda_guard
    entry=args.entry;cfg=read(entry/'config.json')
    if code_identity()!=cfg['baseline']['code']:raise ValueError('学生代码身份改变')
    original=read(CONTROL/'formal/control'/cfg['baseline']['shift']/'config.json');assert_pair(cfg,original)
    cache=validate_cache(cfg['prototype_capacity']['prepared_cache'])
    if sha(cfg['prototype_capacity']['prepared_cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:
        raise ValueError('学生缓存身份改变')
    for path,key in [(cfg['dataset']['json_file'],'source_manifest_sha256'),(cfg['baseline']['target_manifest'],'target_manifest_sha256')]:
        if sha(path)!=cfg['baseline'][key]:raise ValueError('学生数据身份改变')
    if args.child!='probe':verify_acceptance(args.output_root,cfg)
    os.chdir(args.data_root)
    proof=first_cuda_guard()
    if args.child=='probe':engine.probe(cfg,entry/'result.json')
    elif args.child=='train':
        save(entry/'supervision_audit.json',dict(schema=SCHEMA,target_gt_used=False,
            target_tensors_empty=True,source_only_selection=True,selected_count=cache['selected_count'],
            prepared_cache_sha256=cfg['prototype_capacity']['prepared_cache_sha256'],fresh_initialization=True))
        engine.train(cfg,entry)
    else:engine.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(entry/'config.json')))


def finish(root,task,identity):
    e=task['entry'];s=task['shift']
    paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
        'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
        'supervision_audit.json','train.log','evaluate.log','evaluation/source_val.json',
        'evaluation/target_val.json','evaluation/manifest.json']
    static=CONTROL/'formal/control'/s/'complete.json'
    save(e/'complete.json',dict(schema=SCHEMA,shift=s,regime='uda',variant='single_refresh',code=identity,
        target_gt_for_auxiliary=False,acceptance_sha256=sha(root/'acceptance.json'),
        control_complete=str(static),control_complete_sha256=sha(static),
        artifacts={str(e/p):sha(e/p) for p in paths}))
    return audit_entry(e)


def main(args):
    from tools.run_sada_baseline import acquire
    from tools.run_prototype_capacity import audit as audit_control
    from tools.qwen_shard_gpu import acquire_four
    root=args.output_root.resolve();data=args.data_root.resolve();identity=code_identity()
    root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir()
    save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,code=identity,jobs=list(SHIFTS),gpus=[0,1,2,3],
        control_root=str(CONTROL),new_student_trainings=6,source_only_model_selection=True))
    save(root/'state.json',dict(phase='audit_static_controls',pid=os.getpid()))
    refs={s:audit_control(CONTROL/'formal/control'/s) for s in SHIFTS}
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
        PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    save(root/'state.json',dict(phase='exact_frame_preparation',pid=os.getpid()))
    run_cpu([str(args.teacher_python),'-B',str(Path(__file__).resolve()),'--prepare-only',
        '--output-root',str(root),'--video-root',str(args.video_root),'--recovery-root',str(args.recovery_root)],root,'frames',env)
    if args.prepare_only:return
    save(root/'state.json',dict(phase='waiting_four_gpu_teacher',pid=os.getpid()))
    while not (found:=acquire_four([0,1,2,3],data,root,identity['commit'])):time.sleep(15)
    context,cards=found;teacher=root/'teacher';teacher.mkdir()
    save(teacher/'active_leases.json',cards)
    scoring_env=dict(env,CUDA_VISIBLE_DEVICES=','.join(c['uuid'] for c in cards),HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',QWEN_SHARD_LEASES=str(teacher/'active_leases.json'),
        QWEN_SHARD_LEASES_SHA256=sha(teacher/'active_leases.json'))
    with (teacher/'inference.log').open('x') as f:
        p=subprocess.Popen([str(args.teacher_python),'-B',str(CODE/'tools/score_prototype_refresh.py'),'--root',str(root)],
            cwd=CODE,env=scoring_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    ACTIVE['teacher']=dict(process=p,context=context)
    save(root/'state.json',dict(phase='teacher_rescoring',pid=os.getpid(),child_pid=p.pid))
    if p.wait():raise RuntimeError('新候选8B重评失败，保留已评分记录')
    context.__exit__(None,None,None);del ACTIVE['teacher']
    prepared=freeze_scores(root)
    # CPU真实样本只验证输入隔离，不读取目标标注，也不根据目标准确率筛名单。
    os.chdir(data)
    from tools.baseline_engine import dataset
    preflight={}
    for s in SHIFTS:
        cfg=read(CONTROL/'formal/control'/s/'config.json');payload=validate_cache(prepared[s]['path'])
        target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True)
        assert_no_target_gt([target[0]])
        preflight[s]=dict(target_videos=len(target),target_sample_gt_empty=True,selected=payload['selected_count'])
    save(root/'inputs.json',dict(code=identity,policy=POLICY,controls=refs,prepared=prepared,preflight=preflight,
        candidate_audit_sha256=sha(root/'candidate_audit.json'),teacher_complete_sha256=sha(teacher/'inference_complete.json')))
    probe=root/'probes/single_refresh/S1';probe.mkdir(parents=True)
    cfg=config_for(read(CONTROL/'formal/control/S1/config.json'),prepared['S1']['path'],identity,root/'tmp')
    assert_pair(cfg,read(CONTROL/'formal/control/S1/config.json'));save(probe/'config.json',cfg)
    save(root/'state.json',dict(phase='real_student_acceptance',pid=os.getpid()))
    while not (found:=acquire([0,1,2,3],data,root,12288,identity['commit'])):time.sleep(15)
    context,lease=found;p=launch_student('probe',probe,lease,data);ACTIVE['probe']=dict(process=p,context=context)
    if p.wait():raise RuntimeError('刷新候选真实学生验收失败')
    context.__exit__(None,None,None);del ACTIVE['probe']
    report=read(probe/'result.json')
    from tools.run_prototype_oracle import acceptance
    acceptance(report,read(CONTROL/'formal/control/S1/detector_initialization.json')['sha256'])
    minimum=max(12288,math.ceil(report['peak_reserved_mib']+4096))
    save(root/'acceptance.json',dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
        inputs_sha256=sha(root/'inputs.json'),prepared_audit_sha256=sha(root/'prepared/audit.json'),
        report_sha256=sha(probe/'result.json'),minimum_free_mib=minimum,report=report))
    pending=list(SHIFTS);completed=[]
    while pending or ACTIVE:
        for key,task in list(ACTIVE.items()):
            if task['process'].poll() is None:continue
            if task['process'].returncode:raise RuntimeError('刷新训练阶段失败: '+key+'/'+task['stage'])
            if task['stage']=='train':
                task['stage']='evaluate';task['process']=launch_student('evaluate',task['entry'],task['lease'],data);continue
            completed.append(finish(root,task,identity));task['context'].__exit__(None,None,None);del ACTIVE[key]
            print('REFRESH_COMPLETE',key,completed[-1]['map'],flush=True)
        for s in list(pending):
            used={t['lease']['index'] for t in ACTIVE.values()}
            found=acquire([g for g in range(4) if g not in used],data,root,minimum,identity['commit'])
            if not found:break
            context,lease=found;e=root/'formal/single_refresh'/s
            try:
                e.mkdir(parents=True,exist_ok=False)
                cfg=config_for(read(CONTROL/'formal/control'/s/'config.json'),prepared[s]['path'],identity,root/'tmp')
                assert_pair(cfg,read(CONTROL/'formal/control'/s/'config.json'));save(e/'config.json',cfg)
                process=launch_student('train',e,lease,data)
            except BaseException:context.__exit__(None,None,None);raise
            ACTIVE[s]=dict(entry=e,shift=s,stage='train',process=process,lease=lease,context=context);pending.remove(s)
        save(root/'state.json',dict(phase='formal',pid=os.getpid(),completed=completed,pending=pending,
            active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()}))
        time.sleep(5)
    final=[audit_entry(Path(x['complete']).parent) for x in completed]
    if len(final)!=6 or {x['shift'] for x in final}!=set(SHIFTS):raise ValueError('不是精确六方向')
    summary=[]
    for x in sorted(final,key=lambda x:x['shift']):
        s=x['shift'];old=read(CONTROL/'formal/control'/s/'evaluation/target_val.json')['map_0.1_to_0.5']
        uda=read(BASELINE/'formal'/s/'uda/evaluation/target_val.json')['map_0.1_to_0.5']
        summary.append(dict(shift=s,refreshed=x['map'],static=old,uda=uda,delta_vs_static=x['map']-old))
    save(root/'comparison.json',dict(results=summary,mean_delta_vs_static=sum(x['delta_vs_static'] for x in summary)/6,
        policy=POLICY,pipeline_has_additional_previous_student_and_VLM_cost=True))
    save(root/'final_audit.json',dict(status='passed',count=6,schema=SCHEMA,results=final,
        comparison_sha256=sha(root/'comparison.json')))
    save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=final,active={}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    p.add_argument('--video-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/videos'))
    p.add_argument('--recovery-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery_qwen_0924'))
    p.add_argument('--teacher-python',type=Path,default=Path('/vepfs/vlms/envs/rq1-cu121/bin/python'))
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--child',choices=['probe','train','evaluate'])
    p.add_argument('--entry',type=Path)
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        if args.child:child(args)
        elif args.prepare_only:prepare(args.output_root,args.video_root,args.recovery_root)
        else:main(args)
    except BaseException as error:
        if args.output_root.exists():save(args.output_root/('error_'+args.child+'.json' if args.child else 'error.json'),
            dict(type=type(error).__name__,message=str(error)))
        raise
    finally:cleanup()
