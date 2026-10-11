"""单卡共识试跑：先S1-S3六项；合并均值下降即停止，不筛教师/shift。"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.rq4_consensus import prepare,freeze_acceptance,VARIANT,read,save
from tools.rq4_artifacts import reference
from tools.rq4_protocol import file_sha
from tools.run_rq2_matched_control import check_code

CODE=Path(__file__).resolve().parents[1]
TEACHERS=('timelens2_8b','timeple_8b')
PARAMETERS=dict(minimum_quality=.3,maximum_uncertainty=.5,anchor_fraction=.5)


def baseline(shift):
    gpu=2 if int(shift[1:])%2 else 3
    return Path(f'/vepfs/vlms/sada_unified_f9ac276_1009_gpu{gpu}/{shift}/uda_preflight.json')


def pilot_decision(rows):
    expected={(t,'S'+str(i)) for i in range(1,4) for t in TEACHERS}
    if len(rows)!=6 or {(r['teacher'],r['shift']) for r in rows}!=expected:
        raise ValueError('pilot must contain the exact six teacher/shift identities')
    if any(not math.isfinite(r[k]) or not 0<=r[k]<=1 for r in rows for k in ('old','new')):
        raise ValueError('pilot mAP must be finite and on the 0-1 scale')
    old=sum(r['old'] for r in rows)/6;new=sum(r['new'] for r in rows)/6
    return dict(old_mean=old,new_mean=new,delta_pp=100*(new-old),
        continue_remaining=new>=old,rule='combined-six-mean-decrease-stops-all',
        exploratory_target_screening=True,rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('output','source-root','caches','old-relation-root','data-root'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--recovered-entry',type=Path,action='append',default=[])
    p.add_argument('--gpu',choices=('0',),default='0')
    args=p.parse_args();check_code(args.expected_head)
    for key in ('output','source_root','caches','old_relation_root','data_root'):
        setattr(args,key,getattr(args,key).resolve())
    lock=open(args.data_root/'.rq4_consensus_pilot.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    args.output.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
    active={};completed=[]
    state=dict(pid=os.getpid(),phase='source_prepare',variant=VARIANT,
        code_commit=args.expected_head,gpu=args.gpu,completed=completed)
    def status():
        state['active']={k:dict(pid=v[0].pid) for k,v in active.items()}
        tmp=args.output/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(args.output/'state.json')
    def stop(*_): raise RuntimeError('explicit stop; preserve all outputs')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def launch(key,command,log,cpu=False):
        log.parent.mkdir(parents=True,exist_ok=True);stream=log.open('x')
        local=dict(env)
        if cpu: local.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        child=subprocess.Popen(command,cwd=CODE,env=local,stdout=stream,
                               stderr=subprocess.STDOUT,start_new_session=True)
        active[key]=(child,stream);status();print('CONSENSUS_STARTED',key,child.pid,flush=True)
    def wait_one(key):
        child,stream=active[key]
        rc=child.wait();stream.close();del active[key];status()
        if rc: raise RuntimeError(f'{key} failed exit={rc}; see preserved log')
    def run(key,command,cpu=False):
        launch(key,command,args.output/'logs'/(key+'.log'),cpu);wait_one(key)
    try:
        save(args.output/'plan.json',dict(variant=VARIANT,code_commit=args.expected_head,
            gpu=args.gpu,pilot_shifts=['S1','S2','S3'],teachers=TEACHERS,
            source_fit=dict(epochs=500,learning_rate=.001,quantile=.5,per_video=288),
            parameters=PARAMETERS,weight=.1,participation=.2,seed=6,batch_size=2,
            rule='combined-six-mean-decrease-stops-all',same_gpu_cuda_concurrency=1,
            cpu_entry_pipeline=2,target_metrics_used_for_hyperparameters=False))
        original=read(args.old_relation_root/'final_audit.json')
        if original['status']!='passed' or original['count']!=12:
            raise ValueError('old relational matrix must already be complete')
        save(args.output/'old_reference.json',reference(args.old_relation_root/'final_audit.json'))
        status()
        for recovered in args.recovered_entry:
            recovered=recovered.resolve();record=read(recovered/'complete.json')
            pair=(record['teacher'],record['shift'])
            if (pair not in {(t,s) for t in TEACHERS for s in ('S1','S2','S3')}
                    or pair in {(x['teacher'],x['shift']) for x in completed}):
                raise ValueError('unexpected or duplicate recovered pilot identity')
            recovery=read(recovered/'recovery.json')
            if record['artifacts'].get(str(recovered/'recovery.json'))!=file_sha(recovered/'recovery.json'):
                raise ValueError('recovered code provenance is not bound')
            spec=read(recovered/'preflight.json')['rq4_boundary']
            if (any(spec[k]!=v for k,v in PARAMETERS.items()) or spec['weight']!=.1
                    or read(spec['quality_path'])['calibration']['quantile']!=.5):
                raise ValueError('recovered method parameters differ')
            script=('import sys,os;sys.path.insert(0,'+repr(recovery['training_code_root'])+');'
                    'os.chdir('+repr(str(args.data_root))+');'
                    'from tools.rq4_consensus_audit import audit_complete;audit_complete('+repr(str(recovered))+')')
            run('audit_recovered_'+pair[0]+'_'+pair[1],[sys.executable,'-B','-c',script],True)
            completed.append(dict(teacher=pair[0],shift=pair[1],root=str(recovered)))
        status()
        for shifts in (('S1','S2','S3'),('S4','S5','S6')):
            state['phase']='source_prepare';state['shifts']=shifts;status()
            prepared={};reports=[]
            for shift in shifts:
                out=args.output/'quality'/shift;out.mkdir(parents=True)
                table=args.source_root/shift/'table.json'
                if not table.exists():
                    record=read(baseline(shift).with_name('complete.json'))
                    ckpts=list(Path(record['uda']).parent.glob('saved_ckpt/best_ckpt/*.pt'))
                    if len(ckpts)!=1: raise ValueError('ambiguous baseline EMA checkpoint')
                    table=out/'table.json'
                    run('collect_'+shift,[sys.executable,'-B','-m','tools.fit_rq4_localization_quality','collect',
                        '--baseline-config',str(baseline(shift)),'--checkpoint',str(ckpts[0]),
                        '--data-root',str(args.data_root),'--output',str(table),'--per-video','288'],True)
                quality=out/'quality.json'
                run('fit_'+shift,[sys.executable,'-B','-m','tools.fit_rq4_localization_quality','fit',
                    '--table-path',str(table),'--output',str(quality),'--epochs','500',
                    '--learning-rate','.001','--quantile','.5'],True)
                for teacher in TEACHERS:
                    folder=args.output/'prepared'/teacher/shift
                    prepare(args.caches/teacher/shift/'artifacts/direct.json',quality,folder,**PARAMETERS)
                    prepared[(teacher,shift)]=folder/'spec.json'
            pairs=[(t,s) for s in shifts for t in TEACHERS
                   if (t,s) not in {(x['teacher'],x['shift']) for x in completed}]
            for phase in ('acceptance','formal'):
                state['phase']=phase;status();pending=list(pairs);running={}
                while pending or running:
                    # 最多两个entry仅流水重叠CPU准备/审计；内部gpu_slot保证单卡单CUDA。
                    while pending and len(running)<2:
                        teacher,shift=pending.pop(0);key=f'{phase}_{teacher}_{shift}'
                        output=args.output/phase/teacher/shift
                        command=[sys.executable,'-B','-m','tools.run_rq4_consensus_entry',
                            '--baseline-config',str(baseline(shift)),'--spec',str(prepared[(teacher,shift)]),
                            '--output',str(output),'--expected-head',args.expected_head,
                            '--data-root',str(args.data_root),'--gpu',args.gpu]
                        if phase=='acceptance': command.append('--smoke-only')
                        launch(key,command,args.output/'logs'/(key+'.log'))
                        running[key]=(teacher,shift,output)
                    for key,(teacher,shift,output) in list(running.items()):
                        if active[key][0].poll() is None: continue
                        wait_one(key);del running[key]
                        if phase=='acceptance': reports.append(output/'smoke.json')
                        else:
                            # 单项入口已完整终审，矩阵另起CPU进程独立复核。
                            script=('import sys,os;sys.path.insert(0,'+repr(str(CODE))+');os.chdir('+repr(str(args.data_root))+');'
                                    'from tools.rq4_consensus_audit import audit_complete;audit_complete('+repr(str(output))+')')
                            run('audit_'+teacher+'_'+shift,[sys.executable,'-B','-c',script],True)
                            completed.append(dict(teacher=teacher,shift=shift,root=str(output)))
                        status();print('CONSENSUS_DONE',key,flush=True)
                    if running: time.sleep(5)
                if phase=='acceptance':
                    acceptance=args.output/('acceptance_'+shifts[0]+'.json')
                    freeze_acceptance(acceptance,reports,args.expected_head)
                    for pair,path in list(prepared.items()):
                        if pair not in pairs: continue
                        spec=read(path);spec.update(acceptance_path=str(acceptance),acceptance_file_sha256=file_sha(acceptance))
                        accepted=path.with_name('accepted.json');save(accepted,spec);prepared[pair]=accepted
            if shifts[0]=='S1':
                rows=[]
                for item in completed:
                    teacher,shift=item['teacher'],item['shift']
                    new=Path(item['root'])/'evaluation/target_val.json'
                    oldroot=args.old_relation_root/'formal'/teacher/shift/'relational'
                    old=oldroot/'evaluation/target_val.json';record=read(oldroot/'complete.json')
                    if (record['teacher']!=teacher or record['shift']!=shift
                            or record['artifacts'].get(str(old))!=file_sha(old)):
                        raise ValueError('old paired target evaluation identity/SHA differs')
                    rows.append(dict(teacher=teacher,shift=shift,new=read(new)['map_0.1_to_0.5'],
                        old=read(old)['map_0.1_to_0.5'],new_evaluation=reference(new),old_evaluation=reference(old)))
                decision=pilot_decision(rows);save(args.output/'pilot_decision.json',decision)
                if not decision['continue_remaining']:
                    state['phase']='stopped_pilot_underperformed';status()
                    save(args.output/'final_audit.json',dict(status='passed',count=6,variant=VARIANT,
                        scope='pilot-only-stopped-by-user-rule',entries=completed,decision=decision))
                    return
        state['phase']='complete';status()
        save(args.output/'final_audit.json',dict(status='passed',count=12,variant=VARIANT,entries=completed))
    except BaseException as exc:
        state['error']=repr(exc)
        for child,stream in active.values():
            if child.poll() is None:
                try: os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError: pass
        for child,stream in active.values():
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            stream.close()
        active.clear();status();raise


if __name__=='__main__':main()
