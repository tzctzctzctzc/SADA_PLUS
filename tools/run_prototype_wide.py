"""只新增256维/单层Prototype六方向；复用旧128维对照，不恢复已停止权重任务。"""
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
from tools import run_prototype_capacity as capacity
from tools.run_sada_baseline import acquire,spawn
from libs.models.SADA.prototype_capacity import specification

CONTROL=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029/formal')
VARIANT='wide_only'
SHIFTS=tuple('S%d'%i for i in range(1,7))
ACTIVE=capacity.ACTIVE
read,save,sha=protocol.read,protocol.save,protocol.sha


def assert_control_pair(cfg,control):
    if cfg['prototype_capacity']['variant']!=VARIANT or cfg['prototype_capacity']['specification']!=specification(VARIANT):
        raise ValueError('不是冻结的256维单层配置')
    plain=deepcopy(cfg)
    # 唯一科学差异为辅助分支宽度；同一缓存、名单、数据、早停、学习率全部相同。
    cap=plain['prototype_capacity'];old=control['prototype_capacity']
    if any(cap[k]!=old[k] for k in ('prepared_cache','prepared_cache_sha256')):
        raise ValueError('加宽时改变教师工件或名单')
    plain['prototype_capacity']=deepcopy(old)
    for k in ('code','tmpdir'):plain['baseline'][k]=control['baseline'][k]
    if plain!=control:raise ValueError('除宽度/运行身份外改变原版配置')


def validate_probe(report,expected_initial):
    if report['status']!='passed' or report['batch']!=2 or report['initial_detector_sha']!=expected_initial:
        raise ValueError('真实验收身份或共享检测器初始化不配对')
    if report['rng_roundtrip']['status']!='passed' or report['checked_target_batches']<=0:
        raise ValueError('缺少真实目标隔离或完整RNG恢复验收')
    for phase in report['phases'].values():
        if (not phase['branch_parameters_updated'] or not phase['model_updated']
                or phase['target_gt_losses']!=[0.,0.,0.]
                or not math.isfinite(phase['source_gradient_norm']) or phase['source_gradient_norm']<=0):
            raise ValueError('真实验收缺少有效更新/源梯度或存在目标GT')
    aux=report['phases']['5']['auxiliary']
    if not all(math.isfinite(aux[k]) and aux[k]>0 for k in
               ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_source_loss','rq3_selected')):
        raise ValueError('适配阶段没有真实候选/源分类/目标原型梯度')
    if abs(aux['rq3_target_alignment_weight']-.1)>1e-6:raise ValueError('目标权重不是0.1')


def memory_policy(report):
    train_peak=max(p['peak_reserved_mib'] for p in report['phases'].values())
    return dict(training_minimum_free_mib=max(9216,math.ceil(train_peak+4096)),
                evaluation_minimum_free_mib=max(12288,math.ceil(report['peak_reserved_mib']+4096)),
                measured_train_peak_mib=train_peak,measured_probe_peak_mib=report['peak_reserved_mib'])


def finish_process(process):
    if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait()


def run_probe(root,cfg,indices,data):
    entry=root/'probes'/VARIANT;entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',cfg)
    while not (found:=acquire(indices,data,root,12288,cfg['baseline']['code']['commit'])):
        print('WAIT_GPU_FOR_PROBE',flush=True);time.sleep(10)
    context,lease=found;process=None
    try:
        process=spawn('probe',entry/'config.json',entry,lease,data,entry/'result.json')
        ACTIVE['probe']=dict(process=process)
        if process.wait():raise RuntimeError('真实数据验收失败，保留probe.log')
        return read(entry/'result.json')
    finally:
        if process is not None:finish_process(process)
        ACTIVE.pop('probe',None);context.__exit__(None,None,None)


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json')
    control=Path(record['control_complete']).parent
    if sha(control/'complete.json')!=record['control_complete_sha256']:raise ValueError('原版完成凭证改变')
    assert_control_pair(cfg,read(control/'config.json'))
    result=capacity.audit(entry)
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if any(abs(r['auxiliary']['rq3_target_alignment_weight']-.1)>1e-6 for r in rows):
        raise ValueError('正式训练目标权重改变')
    initial=read(entry/'detector_initialization.json')
    if initial['specification']!=specification(VARIANT):raise ValueError('实际模型分支配置不符')
    result['control_map']=read(control/'evaluation/target_val.json')['map_0.1_to_0.5']
    result['delta_pp']=100*(result['map']-result['control_map'])
    return result


def complete(entry,root,identity):
    cfg=read(entry/'config.json');s=cfg['baseline']['shift']
    paths=('config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
           'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
           'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
    baseline=BASE/s/'uda/complete.json';control=CONTROL/s/'complete.json'
    save(entry/'complete.json',dict(schema=capacity.SCHEMA,shift=s,regime='uda',variant=VARIANT,code=identity,
        acceptance_sha256=sha(root/'acceptance.json'),baseline_complete=str(baseline),baseline_complete_sha256=sha(baseline),
        control_complete=str(control),control_complete_sha256=sha(control),
        artifacts={str(entry/n):sha(entry/n) for n in paths}))
    return audit(entry)


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def main(args):
    root=args.output.resolve();data=args.data_root.resolve();identity=protocol.code_identity()
    root.mkdir(parents=True,exist_ok=False);tmp=root/'tmp';tmp.mkdir();completed=[];pending=list(SHIFTS);eval_pending=[]
    inputs={};configs={};indices=[int(x) for x in args.gpus.split(',')]
    if len(set(indices))!=len(indices) or not set(indices)<=set(range(4)):raise ValueError('GPU列表不合法')
    save(root/'plan.json',dict(scope='six_new_width_only_entries_reuse_control',variant=VARIANT,shifts=list(SHIFTS),
        specification=specification(VARIANT),code=identity,physical_gpus=indices,control_root=str(CONTROL),
        no_resume_of_stopped_weights=True,no_new_control_training=True))
    try:
        save(root/'state.json',dict(phase='audit_references',pid=os.getpid(),completed=[],active={}))
        for s in SHIFTS:
            control=CONTROL/s;baseline=BASE/s/'uda'
            print('AUDIT_REFERENCE',s,flush=True)
            old=capacity.audit(control);uda=protocol.audit_entry(baseline)
            control_cfg=read(control/'config.json')
            cfg=capacity.paired_config(read(baseline/'config.json'),VARIANT,
                Path(control_cfg['prototype_capacity']['prepared_cache']),identity,tmp)
            capacity.assert_pair(cfg,read(baseline/'config.json'));assert_control_pair(cfg,control_cfg)
            configs[s]=cfg;inputs[s]=dict(control=old,uda=uda,prepared_cache_sha256=cfg['prototype_capacity']['prepared_cache_sha256'])
        save(root/'inputs.json',dict(references=inputs,code=identity,configurations=configs))
        save(root/'state.json',dict(phase='real_data_acceptance',pid=os.getpid(),completed=[],active={}))
        # 复用现有真实GPU验收：独立初始化，原epoch1/5各完整数据遍历，不存正式checkpoint。
        report=run_probe(root,configs['S1'],indices,data)
        validate_probe(report,read(BASE/'S1/uda/termination.json')['initial_state_sha'])
        policy=memory_policy(report);report_path=root/'probes'/VARIANT/'result.json'
        receipt=root/'probes'/VARIANT/'probe_gpu.json'
        guard=read(receipt)['guard']
        if sha(guard)!=read(receipt)['guard_sha256']:raise ValueError('验收首CUDA凭证SHA改变')
        save(root/'acceptance.json',dict(status='passed',code=identity,batch=2,seed=123,
            inputs_sha256=sha(root/'inputs.json'),specification=specification(VARIANT),resource_policy=policy,
            report=str(report_path),report_sha256=sha(report_path),gpu_receipt=str(receipt),gpu_receipt_sha256=sha(receipt)))
        while pending or eval_pending or ACTIVE:
            for key,t in list(ACTIVE.items()):
                if t['process'].poll() is None:continue
                if t['process'].returncode:raise RuntimeError('阶段失败: '+key+'/'+t['stage'])
                if t['stage']=='train':eval_pending.append(t['entry'])
                else:
                    completed.append(complete(t['entry'],root,identity));print('COMPLETE',key,completed[-1]['map'],flush=True)
                t['context'].__exit__(None,None,None);del ACTIVE[key]
            # 整视频评估独立准入，可换卡；训练不因显存较小的卡而占锁等待评估。
            for stage,jobs,minimum in [('evaluate',eval_pending,policy['evaluation_minimum_free_mib']),
                                       ('train',pending,policy['training_minimum_free_mib'])]:
                for job in list(jobs):
                    found=acquire(indices,data,root,minimum,identity['commit'])
                    if not found:break
                    context,lease=found;entry=job if stage=='evaluate' else root/'formal'/VARIANT/job
                    try:
                        if stage=='train':entry.mkdir(parents=True,exist_ok=False);save(entry/'config.json',configs[job])
                        process=spawn(stage,entry/'config.json',entry,lease,data)
                    except BaseException:context.__exit__(None,None,None);raise
                    s=read(entry/'config.json')['baseline']['shift'];jobs.remove(job)
                    ACTIVE[s]=dict(process=process,stage=stage,context=context,lease=lease,entry=entry)
            save(root/'state.json',dict(phase='formal',pid=os.getpid(),completed=completed,pending=pending,
                evaluation_pending=[str(x) for x in eval_pending],resource_policy=policy,
                active={k:dict(pid=t['process'].pid,gpu=t['lease']['index'],stage=t['stage']) for k,t in ACTIVE.items()}))
            time.sleep(3)
        if len(completed)!=6 or {r['shift'] for r in completed}!=set(SHIFTS):raise ValueError('不是精确六方向')
        rows=[audit(Path(r['complete']).parent) for r in completed]
        save(root/'final_audit.json',dict(status='passed',count=6,variant=VARIANT,results=rows,
            mean_map=sum(x['map'] for x in rows)/6,control_mean_map=sum(x['control_map'] for x in rows)/6,
            mean_delta_pp=sum(x['delta_pp'] for x in rows)/6))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=rows,active={}))
    except BaseException as error:
        save(root/'error.json',dict(type=type(error).__name__,message=str(error)))
        save(root/'state.json',dict(phase='stopped' if isinstance(error,SystemExit) else 'failed',
            pid=os.getpid(),completed=completed,active={}))
        raise
    finally:
        for t in ACTIVE.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in ACTIVE.values():
            finish_process(t['process'])
            if 'context' in t:t['context'].__exit__(None,None,None)
        ACTIVE.clear()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--gpus',default='0,1,2,3')
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop);main(args)
