"""四卡共享调度24项GT类别oracle；复用基线，不继续时序/多原型或真实教师训练。"""
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
from tools.prototype_oracle import (SCHEMA,GROUPS,POLICY,prepare_shift,validate_cache,
    config_for,assert_pair,audit_entry,ExposureTracker,verify_acceptance)
from tools.run_sada_baseline import acquire
from tools.run_prototype_capacity import audit as audit_control

read,save,sha=protocol.read,protocol.save,protocol.sha
ACTIVE={}


def cleanup():
    for task in list(ACTIVE.values()):
        if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
    for task in list(ACTIVE.values()):
        try:task['process'].wait(timeout=30)
        except subprocess.TimeoutExpired:os.killpg(task['process'].pid,signal.SIGKILL);task['process'].wait()
        task['context'].__exit__(None,None,None)
    ACTIVE.clear()


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def spawn(mode,entry,lease,data,report=None):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(name,None)
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--child',mode,'--entry',str(entry),'--data-root',str(data)]
    if report:command+=['--report',str(report)]
    with (entry/(mode+'.log')).open('x') as stream:
        return subprocess.Popen(command,cwd=CODE,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def install_tracker(engine,entry):
    native_build=engine.build
    def build(cfg):
        ctx=native_build(cfg)
        capacity=ctx.model.rq3_capacity
        tracker=ExposureTracker(capacity.cache,capacity.strides,cfg['baseline']['epoch_budget']['iterations_per_epoch'],entry/'oracle_exposure.json')
        original=capacity.forward
        def forward(source_videos,target_videos,source_features,target_features,epoch,**kwargs):
            result=original(source_videos,target_videos,source_features,target_features,epoch,**kwargs)
            tracker.observe(target_videos,epoch,result['selected'])
            return result
        capacity.forward=forward
        return ctx
    engine.build=build


def child(args):
    from tools import baseline_engine as engine
    from tools.baseline_gpu import first_cuda_guard
    entry=args.entry;cfg=read(entry/'config.json')
    if cfg['baseline']['code']!=protocol.code_identity():raise ValueError('子进程不是冻结clean代码')
    if cfg.get('oracle_auxiliary',{}).get('schema')!=SCHEMA or cfg['baseline']['target_gt_used'] is not True:raise ValueError('缺少公开oracle身份')
    if sha(cfg['dataset']['json_file'])!=cfg['baseline']['source_manifest_sha256'] or sha(cfg['baseline']['target_manifest'])!=cfg['baseline']['target_manifest_sha256']:raise ValueError('训练源/无标注目标工件改变')
    if sha(cfg['prototype_capacity']['prepared_cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('oracle缓存改变')
    if args.child!='probe':verify_acceptance(entry.parents[2],cfg)
    native_capture=engine.capture
    def capture(*a,**kw):
        saved=native_capture(*a,**kw);saved['schema']=SCHEMA
        saved['oracle_auxiliary']=deepcopy(cfg['oracle_auxiliary'])
        return saved
    engine.capture=capture
    os.chdir(args.data_root)
    proof=first_cuda_guard()
    if args.child=='probe':engine.probe(cfg,args.report)
    elif args.child=='train':
        save(entry/'supervision_audit.json',dict(schema=SCHEMA,target_gt_used=True,
            target_gt_scope='oracle_auxiliary_cache_only',target_gt_tensor_losses=False,
            raw_target_annotations_loaded_by_trainer=False,source_validation_only_selection=True,
            oracle_cache_sha256=cfg['prototype_capacity']['prepared_cache_sha256']))
        install_tracker(engine,entry);engine.train(cfg,entry)
    else:engine.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    save(entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(entry/'config.json')))


def cpu_preflight(base,prepared):
    """不建CUDA模型：真实首个源/目标样本，冻结名单与GT标签独立复算。"""
    from tools.baseline_engine import dataset
    from libs.models.SADA.prototype_capacity import fully_visible_cache
    cfg=read(base/'config.json')
    source=dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
    target=dataset(cfg,cfg['baseline']['target_manifest'],cfg['tgt_train_split'],True,True)
    if source[0]['labels'].numel()==0:raise ValueError('源样本没有GT')
    sample=target[0]
    protocol.assert_no_target_gt([sample])
    for record in prepared.values():validate_cache(record['path'])
    return dict(source_videos=len(source),target_videos=len(target),target_sample_gt_empty=True,
        all_four_cache_truth_recomputed=True)


def acceptance(report,detector_sha):
    if report['status']!='passed' or report['rng_roundtrip']['status']!='passed' or report['initial_detector_sha']!=detector_sha:raise ValueError('GPU验收/共享初始化/随机流失败')
    for phase in report['phases'].values():
        aux=phase['auxiliary']
        if phase['target_gt_losses']!=[0.,0.,0.] or not phase['branch_parameters_updated']:raise ValueError('目标主检测GT隔离/辅助参数更新失败')
        if not all(math.isfinite(aux.get(k,0)) and aux.get(k,0)>0 for k in ['rq3_source_loss','rq3_alignment_loss','rq3_selected','rq3_target_gradient_norm']):raise ValueError('oracle缺少真实候选或有效辅助梯度')


def finish(root,task,baseline,control,identity):
    entry=task['entry'];group,shift=task['group'],task['shift']
    paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','supervision_audit.json',
        'oracle_exposure.json','train.log','evaluate.log','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
    base=baseline/'formal'/shift/'uda/complete.json';old=control/'formal/control'/shift/'complete.json'
    save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,group=group,regime='uda',code=identity,
        target_gt_for_auxiliary=True,not_a_deployable_uda_result=True,
        acceptance_sha256=sha(root/'acceptance.json'),baseline_complete=str(base),baseline_complete_sha256=sha(base),
        control_complete=str(old),control_complete_sha256=sha(old),artifacts={str(entry/p):sha(entry/p) for p in paths}))
    return audit_entry(entry)


def summary(root,completed,baseline,control):
    output={}
    for group in GROUPS:
        rows=[x for x in completed if x['group']==group]
        if len(rows)!=6:continue
        output[group]=dict(mean_map=sum(x['map'] for x in rows)/6,shifts={x['shift']:x['map'] for x in rows},
            selected_candidates={x['shift']:x['selected_candidates'] for x in rows},
            unique_routable_candidates={x['shift']:x['unique_routable_candidates'] for x in rows})
    output['uda']=dict(mean_map=sum(read(baseline/'formal'/s/'uda/evaluation/target_val.json')['map_0.1_to_0.5'] for s in ['S'+str(i) for i in range(1,7)])/6)
    output['original_8b20']=dict(mean_map=sum(read(control/'formal/control'/s/'evaluation/target_val.json')['map_0.1_to_0.5'] for s in ['S'+str(i) for i in range(1,7)])/6)
    save(root/'results_summary.json',dict(scope='conditional GT-class oracle, not a mathematical mAP upper bound',
        oracle_cannot_select_real_teacher_hyperparameters_on_target=True,results=output))


def main(args):
    import numpy as np
    import torch
    root=args.output_root.resolve();data=args.data_root.resolve();baseline=args.baseline_root.resolve();control=args.control_root.resolve()
    identity=protocol.code_identity();root.mkdir(parents=True,exist_ok=False);(root/'tmp').mkdir()
    indices=[int(x) for x in args.gpus.split(',')]
    if len(set(indices))!=len(indices) or not set(indices)<={0,1,2,3}:raise ValueError('物理GPU集合非法')
    save(root/'state.json',dict(phase='reference_and_cache_audit',pid=os.getpid(),completed=[],active={}))
    references={};prepared={};preflight={}
    for i in range(1,7):
        shift='S'+str(i);base=baseline/'formal'/shift/'uda'
        references[shift]=dict(uda=protocol.audit_entry(base),control=audit_control(control/'formal/control'/shift))
        original=args.prepared_root/(shift+'.json')
        if sha(original)!=read(args.prepared_root/'audit.json')['entries'][shift]['sha256']:raise ValueError('原200候选缓存改变')
        prepared[shift]=prepare_shift(root,shift,base,original)
        torch.manual_seed(123);np.random.seed(123)
        preflight[shift]=cpu_preflight(base,prepared[shift]);print('CPU_READY',shift,prepared[shift],flush=True)
    inputs=dict(code=identity,policy=POLICY,groups=GROUPS,baseline=str(baseline),control=str(control),
        original_references=references,prepared=prepared,cpu_preflight=preflight,target_gt_for_auxiliary=True,
        baseline_feature_inventory_sha256=sha(baseline/'inputs/feature_inventory.json'))
    save(root/'inputs.json',inputs)
    jobs=[(g,'S'+str(i)) for i in range(1,7) for g in GROUPS]
    save(root/'plan.json',dict(schema=SCHEMA,policy=POLICY,jobs=jobs,gpus=indices))
    probes_pending=list(GROUPS);reports={};minimum=12288
    try:
        while probes_pending or ACTIVE:
            for group,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError('真实GPU验收失败: '+group)
                result=read(task['entry']/'result.json');acceptance(result,read(baseline/'formal/S1/uda/termination.json')['initial_state_sha'])
                reports[group]=result;task['context'].__exit__(None,None,None);del ACTIVE[group]
                print('GPU_ACCEPTED',group,flush=True)
            for group in list(probes_pending):
                used={t['lease']['index'] for t in ACTIVE.values()}
                found=acquire([g for g in indices if g not in used],data,root,minimum,identity['commit'])
                if not found:break
                context,lease=found;entry=root/'probes'/group
                try:
                    entry.mkdir(parents=True,exist_ok=False)
                    cfg=config_for(read(baseline/'formal/S1/uda/config.json'),group,prepared['S1'][group]['path'],identity,root/'tmp')
                    assert_pair(cfg,read(baseline/'formal/S1/uda/config.json'));save(entry/'config.json',cfg)
                    process=spawn('probe',entry,lease,data,entry/'result.json')
                except BaseException:context.__exit__(None,None,None);raise
                ACTIVE[group]=dict(entry=entry,group=group,context=context,lease=lease,process=process,stage='probe')
                probes_pending.remove(group)
            save(root/'state.json',dict(phase='real_gpu_acceptance',pid=os.getpid(),completed=[],
                active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()}))
            time.sleep(3)
        minimum=max(12288,math.ceil(max(x['peak_reserved_mib'] for x in reports.values())+4096))
        save(root/'acceptance.json',dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
            target_gt_for_auxiliary=True,inputs_sha256=sha(root/'inputs.json'),minimum_free_mib=minimum,reports=reports,
            report_sha256={str(root/'probes'/g/'result.json'):sha(root/'probes'/g/'result.json') for g in GROUPS}))
        pending=list(jobs);completed=[];groups_done=set()
        while pending or ACTIVE:
            for key,task in list(ACTIVE.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError('oracle阶段失败: '+key+'/'+task['stage'])
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',task['entry'],task['lease'],data);continue
                result=finish(root,task,baseline,control,identity);completed.append(result)
                print('COMPLETE',key,result['map'],flush=True)
                task['context'].__exit__(None,None,None);del ACTIVE[key]
                for group in GROUPS:
                    records=[x for x in completed if x['group']==group]
                    if len(records)==6 and group not in groups_done:
                        save(root/group/'final_audit.json',dict(status='passed',count=6,schema=SCHEMA,results=records))
                        groups_done.add(group);summary(root,completed,baseline,control)
            for group,shift in list(pending):
                used={t['lease']['index'] for t in ACTIVE.values()}
                found=acquire([g for g in indices if g not in used],data,root,minimum,identity['commit'])
                if not found:break
                context,lease=found;entry=root/'formal'/group/shift
                try:
                    entry.mkdir(parents=True,exist_ok=False)
                    cfg=config_for(read(baseline/'formal'/shift/'uda/config.json'),group,prepared[shift][group]['path'],identity,root/'tmp')
                    assert_pair(cfg,read(baseline/'formal'/shift/'uda/config.json'));save(entry/'config.json',cfg)
                    process=spawn('train',entry,lease,data)
                except BaseException:context.__exit__(None,None,None);raise
                ACTIVE[group+'_'+shift]=dict(entry=entry,group=group,shift=shift,context=context,lease=lease,process=process,stage='train')
                pending.remove((group,shift))
            save(root/'state.json',dict(phase='formal_oracle',pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()}))
            time.sleep(5)
        if {(x['group'],x['shift']) for x in completed}!=set(jobs):raise ValueError('完成身份不是精确24项')
        final=[audit_entry(Path(x['complete']).parent) for x in completed]
        save(root/'final_audit.json',dict(status='passed',count=24,schema=SCHEMA,target_gt_for_auxiliary=True,results=final))
        summary(root,final,baseline,control)
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=final,active={}))
    finally:cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--child',choices=['probe','train','evaluate']);p.add_argument('--entry',type=Path);p.add_argument('--report',type=Path)
    p.add_argument('--output-root',type=Path);p.add_argument('--baseline-root',type=Path,default=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029'))
    p.add_argument('--control-root',type=Path,default=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029'))
    p.add_argument('--prepared-root',type=Path,default=Path('/vepfs/vlms/prototype_capacity_evidence_1bfd135_1029'))
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='0,1,2,3')
    args=p.parse_args()
    if args.child:child(args)
    else:
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:main(args)
        except BaseException as error:
            if args.output_root and args.output_root.exists():
                save(args.output_root/'error.json',dict(type=type(error).__name__,message=str(error)))
            raise
