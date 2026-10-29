"""双卡分别运行时序汇聚/多原型；复用已完成原版六方向，不做累加或scaling。"""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import signal
import sys
import time

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools import baseline_protocol as protocol
from tools.run_sada_baseline import acquire,spawn
from tools.run_prototype_capacity import assert_pair,paired_config,audit as audit_control
from libs.models.SADA.prototype_methods import VARIANTS,method_specification

SCHEMA='rq3-prototype-independent-pool-multi-seed123-v1'
ACTIVE={}
read,save,sha=protocol.read,protocol.save,protocol.sha


def stop(signum,frame):
    signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
    raise SystemExit(128+signum)


def cleanup():
    for task in ACTIVE.values():
        process=task['process']
        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
    for task in ACTIVE.values():
        process=task['process']
        try:process.wait(timeout=30)
        except __import__('subprocess').TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        task['context'].__exit__(None,None,None)
    ACTIVE.clear()


def acceptance_check(report,variant,detector_sha):
    if report['status']!='passed' or report['rng_roundtrip']['status']!='passed':
        raise ValueError('真实GPU验收或完整状态恢复失败')
    if report['initial_detector_sha']!=detector_sha:raise ValueError('共享检测器初始化与配对基线不符')
    for phase in report['phases'].values():
        aux=phase['auxiliary']
        if phase['target_gt_losses']!=[0.,0.,0.] or not phase['branch_parameters_updated']:
            raise ValueError('目标GT或真实参数更新验收失败')
        if not all(aux.get(k,0)>0 and math.isfinite(aux[k]) for k in
                   ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_source_loss','rq3_selected')):
            raise ValueError('缺少原始名单真实候选/源分类/非零目标梯度')
        if variant=='temporal_pool':
            if set(phase['pool_submodules_updated'])!={'inside_attention','outside_attention','pool_fusion'}:
                raise ValueError('没有核验全部时序汇聚子模块')
            if not all(phase['pool_submodules_updated'].values()):raise ValueError('时序汇聚存在未更新的子模块')
            if not aux.get('rq3_target_temporal_context_valid',0)>0:raise ValueError('没有真实有效上下文')
        elif not (aux.get('rq3_prototype_classes_with_multiple',0)>0 and aux.get('rq3_prototype_source_updates',0)>0):
            raise ValueError('多原型只有名义K=3，没有真实源GT多中心')


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json')
    variant=cfg['prototype_capacity']['variant']
    if variant not in VARIANTS or cfg['prototype_capacity']['specification']!=method_specification(variant):
        raise ValueError('不是本轮独立方法')
    base=Path(record['baseline_complete']).parent
    if sha(base/'complete.json')!=record['baseline_complete_sha256']:raise ValueError('UDA基线凭证改变')
    assert_pair(cfg,read(base/'config.json'))
    control=Path(record['control_complete']).parent;control_cfg=deepcopy(read(control/'config.json'))
    original=control_cfg.pop('prototype_capacity')
    if original['variant']!='control' or sha(control/'complete.json')!=record['control_complete_sha256']:
        raise ValueError('原版Prototype对照身份改变')
    if sha(cfg['prototype_capacity']['prepared_cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:
        raise ValueError('固定教师缓存改变')
    if original['prepared_cache_sha256']!=cfg['prototype_capacity']['prepared_cache_sha256']:
        raise ValueError('新方法不是原版同一固定名单')
    if read(entry/'detector_initialization.json')['sha256']!=read(base/'termination.json')['initial_state_sha']:
        raise ValueError('共享检测器初始化改变')
    result=protocol.audit_entry(entry,expected_schema=SCHEMA)
    rows=[__import__('json').loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(x['auxiliary'].get('rq3_target_gradient_norm',0)>0 and
               x['auxiliary'].get('rq3_alignment_loss',0)>0 for x in rows):raise ValueError('目标原型未实际作用')
    result['variant']=variant
    return result


def main(args):
    root=args.output_root.resolve();data=args.data_root.resolve();baseline=args.baseline_root.resolve()
    control_root=args.control_root.resolve();prepared_root=args.prepared_root.resolve()
    gpu_indices=[int(x) for x in args.gpus.split(',')]
    if len(gpu_indices)!=2 or len(set(gpu_indices))!=2:raise ValueError('必须两张不同的物理卡')
    identity=protocol.code_identity();root.mkdir(parents=True,exist_ok=False)
    args.tmpdir=args.tmpdir.resolve();args.tmpdir.mkdir(parents=True,exist_ok=True)
    state=dict(phase='audit_original',pid=os.getpid(),active={},completed=[]);save(root/'state.json',state)
    while not (control_root/'final_audit.json').exists():
        if (control_root/'error.json').exists():raise ValueError('原版队列异常，不跳过前序审计')
        print('WAIT_ORIGINAL_FINAL',flush=True);time.sleep(15)
    final=read(control_root/'final_audit.json')
    if final['status']!='passed' or final['count']!=12:raise ValueError('前序原版/宽深版未完整终审')
    prepared=read(prepared_root/'audit.json')
    if prepared['status']!='passed' or set(prepared['entries'])!={'S'+str(i) for i in range(1,7)}:
        raise ValueError('教师证据不是精确六方向')
    references={}
    for shift in ['S'+str(i) for i in range(1,7)]:
        old=control_root/'formal/control'/shift
        references[shift]=audit_control(old)
        row=prepared['entries'][shift]
        if sha(row['path'])!=row['sha256'] or row['selected_count']!=40:raise ValueError('教师工件或固定40名单改变')
    inputs=dict(code=identity,seed=123,batch=2,teacher='qwen3_vl_8b',
        variants={v:method_specification(v) for v in VARIANTS},
        original_final=str(control_root/'final_audit.json'),original_final_sha256=sha(control_root/'final_audit.json'),
        original_control_audits=references,baseline_input_sha256=sha(baseline/'inputs/identity.json'),
        feature_inventory_sha256=sha(baseline/'inputs/feature_inventory.json'),
        prepared_audit=str(prepared_root/'audit.json'),prepared_audit_sha256=sha(prepared_root/'audit.json'))
    save(root/'inputs.json',inputs)
    mapping=dict(zip(VARIANTS,gpu_indices));minimum=12288
    jobs=[(v,'S'+str(i)) for i in range(1,7) for v in VARIANTS]
    save(root/'plan.json',dict(schema=SCHEMA,scope='independent; original128; unchanged7/FPNoneroute',jobs=jobs,gpus=mapping))
    reports={};completed=[];pending=list(jobs);phase='gpu_acceptance'
    try:
        while pending or ACTIVE or phase=='gpu_acceptance':
            for key,task in list(ACTIVE.items()):
                process=task['process']
                if process.poll() is None:continue
                if process.returncode:raise RuntimeError('阶段失败: '+key+'/'+task['stage'])
                e=task['entry'];variant=task['variant'];shift=task['shift']
                if task['stage']=='probe':
                    report=read(e/'result.json')
                    acceptance_check(report,variant,read(baseline/'formal/S1/uda/termination.json')['initial_state_sha'])
                    reports[variant]=report
                elif task['stage']=='train':
                    task['stage']='evaluate';task['process']=spawn('evaluate',e/'config.json',e,task['lease'],data)
                    continue
                else:
                    paths=('config.json','runtime_config.json','seed_proof.json','detector_initialization.json',
                        'metrics.jsonl','termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json',
                        'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
                    base=baseline/'formal'/shift/'uda/complete.json';old=control_root/'formal/control'/shift/'complete.json'
                    save(e/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant=variant,code=identity,
                        acceptance_sha256=sha(root/'acceptance.json'),baseline_complete=str(base),baseline_complete_sha256=sha(base),
                        control_complete=str(old),control_complete_sha256=sha(old),artifacts={str(e/n):sha(e/n) for n in paths}))
                    completed.append(audit(e));print('COMPLETE',key,completed[-1]['map'],flush=True)
                task['context'].__exit__(None,None,None);del ACTIVE[key]
            if phase=='gpu_acceptance' and set(reports)==set(VARIANTS):
                minimum=max(12288,math.ceil(max(x['peak_reserved_mib'] for x in reports.values())+4096))
                save(root/'acceptance.json',dict(status='passed',code=identity,inputs_sha256=sha(root/'inputs.json'),
                    batch=2,seed=123,minimum_free_mib=minimum,reports=reports,
                    report_sha256={str(root/'probes'/v/'result.json'):sha(root/'probes'/v/'result.json') for v in VARIANTS}))
                phase='formal'
            for variant in VARIANTS:
                if any(x['variant']==variant for x in ACTIVE.values()):continue
                if phase=='gpu_acceptance':
                    if variant in reports:continue
                    shift='S1';stage='probe';e=root/'probes'/variant
                else:
                    next_job=next((x for x in pending if x[0]==variant),None)
                    if next_job is None:continue
                    _,shift=next_job;stage='train';e=root/'formal'/variant/shift
                found=acquire([mapping[variant]],data,root,minimum,identity['commit'])
                if not found:continue
                context,lease=found
                try:
                    e.mkdir(parents=True,exist_ok=False)
                    base=read(baseline/'formal'/shift/'uda/config.json')
                    cfg=paired_config(base,variant,Path(prepared['entries'][shift]['path']),identity,args.tmpdir)
                    assert_pair(cfg,base);save(e/'config.json',cfg)
                    process=spawn(stage,e/'config.json',e,lease,data,e/'result.json' if stage=='probe' else None)
                except BaseException:
                    context.__exit__(None,None,None);raise
                key=variant+'_'+shift
                ACTIVE[key]=dict(entry=e,variant=variant,shift=shift,stage=stage,context=context,lease=lease,process=process)
                if stage=='train':pending.remove((variant,shift))
            save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=completed,pending=pending,
                active={k:dict(pid=v['process'].pid,gpu=v['lease']['index'],stage=v['stage']) for k,v in ACTIVE.items()}))
            time.sleep(5)
        if {(x['variant'],x['shift']) for x in completed}!=set(jobs):raise ValueError('不是精确两方法六方向')
        for variant in VARIANTS:
            rows=[audit(root/'formal'/variant/('S'+str(i))) for i in range(1,7)]
            save(root/variant/'final_audit.json',dict(status='passed',count=6,results=rows))
        save(root/'final_audit.json',dict(status='passed',count=12,schema=SCHEMA,results=completed))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))
    finally:cleanup()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output-root','baseline-root','control-root','prepared-root','tmpdir'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--gpus',default='0,1');p.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    args=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main(args)
    except BaseException as error:
        if args.output_root.exists():
            save(args.output_root/'error.json',dict(type=type(error).__name__,error=str(error)))
            old=read(args.output_root/'state.json');old.update(phase='failed',error=str(error),active={});save(args.output_root/'state.json',old)
        raise
