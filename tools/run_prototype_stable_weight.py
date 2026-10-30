"""新版多原型六项与独立权重十二项；双卡各一组，共享外部负载但每卡单自有CUDA。"""
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
from libs.models.SADA.prototype_methods import REVISION_VARIANTS,revision_specification

SCHEMA='rq3-prototype-stable-multi-and-weight-v1'
BASE=Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
CONTROL=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
GROUPS={'multi':[0,3],'weight':[1,2]}
read,save,sha=protocol.read,protocol.save,protocol.sha


def assert_pair(cfg,reference):
    x=deepcopy(cfg);v=x['prototype_capacity']['variant']
    if x['prototype_capacity']['specification']!=revision_specification(v):raise ValueError('新版参数偏离冻结方案')
    x['prototype_capacity']['variant']='control';x['prototype_capacity']['specification']=reference['prototype_capacity']['specification']
    for k in ['code','tmpdir']:x['baseline'][k]=reference['baseline'][k]
    if x!=reference or cfg['seed']!=123 or cfg['loader']['batch_size']!=2:raise ValueError('不与原版严格配对')


def make_config(reference,variant,identity,tmpdir):
    cfg=deepcopy(reference);cfg['baseline']['code']=identity;cfg['baseline']['tmpdir']=str(tmpdir)
    cfg['prototype_capacity'].update(variant=variant,specification=revision_specification(variant))
    assert_pair(cfg,reference);return cfg


def acceptance(report,variant,detector_sha,source_bank=None):
    if report['status']!='passed' or report['initial_detector_sha']!=detector_sha or report['rng_roundtrip']['status']!='passed':
        raise ValueError('真实GPU验收/检测器初始化/完整随机状态失败')
    weight=revision_specification(variant)['target_alignment_weight']
    for phase in report['phases'].values():
        aux=phase['auxiliary']
        if phase['target_gt_losses']!=[0.,0.,0.] or not phase['branch_parameters_updated']:raise ValueError('GT隔离或真实参数更新失败')
        for k in ['rq3_source_loss','rq3_alignment_loss','rq3_target_gradient_norm','rq3_selected']:
            if not math.isfinite(aux.get(k,0)) or aux.get(k,0)<=0:raise ValueError('缺少真实监督/候选/目标辅助梯度: '+k)
        if abs(aux['rq3_target_alignment_weight']-weight)>1e-6:raise ValueError('实际辅助权重错误')
    if variant=='multi_stable':
        if not source_bank or not source_bank['rng_preserved'] or source_bank['target_data_loaded'] or source_bank['source_validation_loaded']:
            raise ValueError('源多原型初始化违反隔离或随机流协议')
        if report['phases']['5']['auxiliary']['rq3_source_bank_initialized']!=1:raise ValueError('新版未实际完成源簇初始化')
        if not any(x['active_centers']>1 for x in source_bank['classes'].values()):raise ValueError('真实源数据未产生有支持的多中心，不放宽门槛强训')


def child(args):
    from tools import baseline_engine as engine
    from tools.baseline_gpu import first_cuda_guard
    cfg=read(args.config);variant=cfg['prototype_capacity']['variant'];entry=args.entry
    if protocol.code_identity()!=cfg['baseline']['code']:raise ValueError('子进程代码身份不符')
    for path,key in [(cfg['dataset']['json_file'],'source_manifest_sha256'),(cfg['baseline']['target_manifest'],'target_manifest_sha256')]:
        if sha(path)!=cfg['baseline'][key]:raise ValueError('数据成员/SHA改变')
    assert_pair(cfg,read(CONTROL/cfg['baseline']['shift']/'config.json'))
    os.chdir(args.data_root);proof=first_cuda_guard()
    last_ctx={}
    if variant=='multi_stable':
        from tools.prototype_stable_bank import initialize_source_bank,prototype_snapshot
        original=engine.epoch_setup
        def setup(ctx,epoch):
            last_ctx['ctx']=ctx
            if epoch>1:prototype_snapshot(ctx,entry,epoch-1)
            if epoch>=5 and not bool(ctx.model.rq3_capacity.branch.source_bank_initialized):initialize_source_bank(ctx,entry)
            original(ctx,epoch)
        engine.epoch_setup=setup
    if args.child=='probe':engine.probe(cfg,entry/'result.json')
    elif args.child=='train':engine.train(cfg,entry)
    else:engine.evaluate(cfg,entry,cfg['baseline']['annotation_for_offline_evaluation'])
    if variant=='multi_stable' and last_ctx:
        epoch=5 if args.child=='probe' else read(entry/'termination.json')['decision']['epoch']
        prototype_snapshot(last_ctx['ctx'],entry,epoch)
    save(entry/(args.child+'_gpu.json'),dict(guard=proof,guard_sha256=sha(proof),config_sha256=sha(args.config)))


def spawn(mode,entry,card,data):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        CUDA_VISIBLE_DEVICES=card['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED=str(cfg['seed']),
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8',SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
    for key in ['QWEN_INFORMATION_LEASE_SHA','RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(key,None)
    with (entry/(mode+'.log')).open('x') as log:
        return subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'--child',mode,'--config',str(entry/'config.json'),
            '--entry',str(entry),'--data-root',str(data)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)


def audit(entry):
    entry=Path(entry);cfg=read(entry/'config.json');record=read(entry/'complete.json');variant=cfg['prototype_capacity']['variant']
    reference=CONTROL/cfg['baseline']['shift'];assert_pair(cfg,read(reference/'config.json'))
    if sha(reference/'complete.json')!=record['control_complete_sha256']:raise ValueError('原版对照身份改变')
    if sha(cfg['prototype_capacity']['prepared_cache'])!=cfg['prototype_capacity']['prepared_cache_sha256']:raise ValueError('教师缓存改变')
    result=protocol.audit_entry(entry,expected_schema=SCHEMA);result['variant']=variant
    rows=[json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(r['auxiliary']['rq3_target_gradient_norm']>0 for r in rows):raise ValueError('没有真实辅助梯度')
    weight=revision_specification(variant)['target_alignment_weight']
    if any(abs(r['auxiliary']['rq3_target_alignment_weight']-weight)>1e-6 for r in rows):raise ValueError('训练权重非固定值')
    if variant=='multi_stable':
        bank=read(entry/'source_bank.json')
        if bank['specification']!=revision_specification(variant) or not bank['rng_preserved'] or bank['target_data_loaded']:
            raise ValueError('源建簇不符合冻结协议')
        if any(r['auxiliary']['rq3_source_bank_initialized']!=(1. if r['epoch']>=5 else 0.) for r in rows):
            raise ValueError('源簇初始化阶段不正确')
        final=read(entry/'prototype_diagnostics'/('epoch_%03d.json'%rows[-1]['epoch']))
        if not final['initialized']:raise ValueError('缺最终逐中心使用诊断')
    return result


def complete(entry,root,identity):
    cfg=read(entry/'config.json');shift=cfg['baseline']['shift'];variant=cfg['prototype_capacity']['variant']
    paths=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl','termination.json',
        'best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
    if variant=='multi_stable':
        paths+=['source_bank.json']
        paths += [str(p.relative_to(entry)) for p in (entry/'prototype_diagnostics').glob('*.json')]
    save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant=variant,code=identity,
        acceptance_sha256=sha(root/'acceptance.json'),control_complete=str(CONTROL/shift/'complete.json'),
        control_complete_sha256=sha(CONTROL/shift/'complete.json'),artifacts={str(entry/n):sha(entry/n) for n in paths}))
    return audit(entry)


def supervisor(args):
    from tools.run_sada_baseline import acquire
    from tools.run_prototype_capacity import audit as audit_control
    root=args.output_root.resolve();data=args.data_root.resolve();identity=protocol.code_identity();root.mkdir(parents=True,exist_ok=False)
    tmpdir=root/'tmp';tmpdir.mkdir();active={};reports={};finished=[];minimum={v:12288 for v in REVISION_VARIANTS}
    jobs=[(v,f'S{i}') for i in range(1,7) for v in REVISION_VARIANTS];pending=list(jobs);references={}
    save(root/'plan.json',dict(schema=SCHEMA,code=identity,jobs=jobs,physical_gpus=GROUPS,
        scope='six stabilized-multi at weight0.1; twelve original-single-center at fixed weight0.2/0.4',
        specifications={v:revision_specification(v) for v in REVISION_VARIANTS},
        selection='source_loss_only',weight_results='exploratory_target_comparison_no_adaptive_parameter_changes'))
    def stop(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        save(root/'state.json',dict(phase='audit_references',pid=os.getpid()))
        for shift in [f'S{i}' for i in range(1,7)]:
            protocol.audit_entry(BASE/'formal'/shift/'uda');audit_control(CONTROL/shift)
            references[shift]=dict(control_complete_sha256=sha(CONTROL/shift/'complete.json'),
                baseline_complete_sha256=sha(BASE/'formal'/shift/'uda/complete.json'),
                config=read(CONTROL/shift/'config.json'))
        save(root/'inputs.json',dict(references=references,code=identity,control_root=str(CONTROL),baseline_root=str(BASE)))
        probes=list(REVISION_VARIANTS);phase='gpu_acceptance'
        while probes or pending or active:
            for key,t in list(active.items()):
                if t['p'].poll() is None:continue
                if t['p'].returncode:raise RuntimeError('阶段失败: '+key+'/'+t['stage'])
                e=t['entry'];v=t['variant']
                if t['stage']=='probe':
                    report=read(e/'result.json');bank=read(e/'source_bank.json') if v=='multi_stable' else None
                    acceptance(report,v,read(BASE/'formal/S1/uda/termination.json')['initial_state_sha'],bank)
                    reports[v]=report
                    peak=max(report['peak_reserved_mib'],bank['peak_reserved_mib'] if bank else 0.)
                    minimum[v]=max(12288,math.ceil(peak+4096))
                elif t['stage']=='train':
                    t['stage']='evaluate';t['p']=spawn('evaluate',e,t['card'],data);continue
                else:
                    result=complete(e,root,identity);finished.append(result);print('COMPLETE',v,t['shift'],result['map'],flush=True)
                t['ctx'].__exit__(None,None,None);del active[key]
            if phase=='gpu_acceptance' and len(reports)==3:
                save(root/'acceptance.json',dict(status='passed',code=identity,batch=2,seed=123,
                    inputs_sha256=sha(root/'inputs.json'),report_sha256={str(root/'probes'/v/'result.json'):sha(root/'probes'/v/'result.json') for v in reports},
                    source_bank_sha256=sha(root/'probes/multi_stable/source_bank.json'),minimum_free_mib=minimum))
                phase='formal'
            tasks=[(v,'S1','probe') for v in probes] if phase=='gpu_acceptance' else [(v,s,'train') for v,s in pending]
            for v,s,stage in tasks:
                group='multi' if v=='multi_stable' else 'weight';used={t['card']['index'] for t in active.values()}
                found=acquire([i for i in GROUPS[group] if i not in used],data,root,minimum[v],identity['commit'])
                if not found:continue
                ctx,card=found;e=root/'probes'/v if stage=='probe' else root/'formal'/v/s
                try:
                    e.mkdir(parents=True,exist_ok=False)
                    cfg=make_config(references[s]['config'],v,identity,tmpdir);save(e/'config.json',cfg)
                    p=spawn(stage,e,card,data)
                except BaseException:ctx.__exit__(None,None,None);raise
                active[v+'_'+s]=dict(p=p,stage=stage,entry=e,variant=v,shift=s,ctx=ctx,card=card)
                if stage=='probe':probes.remove(v)
                else:pending.remove((v,s))
            save(root/'state.json',dict(phase=phase,pid=os.getpid(),completed=finished,pending=pending,
                active={k:dict(pid=t['p'].pid,gpu=t['card']['index'],stage=t['stage']) for k,t in active.items()},probes_pending=probes))
            time.sleep(5)
        if {(x['variant'],x['shift']) for x in finished}!=set(jobs) or len(finished)!=18:raise ValueError('不是精确18项')
        for v in REVISION_VARIANTS:
            rows=[audit(root/'formal'/v/f'S{i}') for i in range(1,7)]
            controls=[read(CONTROL/f'S{i}'/'evaluation/target_val.json')['map_0.1_to_0.5'] for i in range(1,7)]
            save(root/v/'final_audit.json',dict(status='passed',count=6,results=rows,
                mean_map=sum(x['map'] for x in rows)/6,control_mean=sum(controls)/6,
                delta_pp=100*(sum(x['map'] for x in rows)-sum(controls))/6))
        save(root/'final_audit.json',dict(status='passed',count=18,results=finished))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=finished,active={}))
    except BaseException as e:
        save(root/'error.json',dict(type=type(e).__name__,message=str(e)))
        save(root/'state.json',dict(phase='stopped' if isinstance(e,SystemExit) else 'failed',pid=os.getpid(),completed=finished))
        raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['p'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['p'].pid,signal.SIGKILL);t['p'].wait()
            t['ctx'].__exit__(None,None,None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-root',type=Path)
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    parser.add_argument('--child',choices=['probe','train','evaluate']);parser.add_argument('--entry',type=Path);parser.add_argument('--config',type=Path)
    args=parser.parse_args();child(args) if args.child else supervisor(args)
