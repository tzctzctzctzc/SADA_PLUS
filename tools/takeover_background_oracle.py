"""只接管O的调度；已有CUDA训练原进程继续，训练代码/配置不变。"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
        return dict(state=fields[0],parent=int(fields[1]),start=fields[19])
    except FileNotFoundError:
        return None


def memory_requirement(reports):
    peak=max(float(x['peak_reserved_mib']) for x in reports)
    if not math.isfinite(peak) or peak<=0:raise ValueError('缺少真实显存峰值')
    return max(8192,math.ceil(peak+3072))


def validate_snapshot(state,root,parent):
    if state['pid']!=parent or state['completed'] or set(state['active'])!={'S1','S2','S3'}:
        raise ValueError('只允许接管当前三路未完成的精确快照')
    if state['pending']!=['S4','S5','S6']:raise ValueError('剩余集合不符')
    for shift,task in state['active'].items():
        identity=process_identity(task['pid'])
        if task['stage']!='train' or identity is None or identity['parent']!=parent or identity['state']=='Z':
            raise ValueError('待接管项不是真实存活训练')
        command=(Path('/proc')/str(task['pid'])/'cmdline').read_bytes().split(b'\0')
        if str(root/'formal'/shift).encode() not in command or b'train' not in command:
            raise ValueError('训练进程身份不符')
        rows=[json.loads(x) for x in (root/'formal'/shift/'metrics.jsonl').read_text().splitlines()]
        if not rows or [x['epoch'] for x in rows]!=list(range(1,len(rows)+1)):
            raise ValueError('没有连续真实epoch')


def main(args):
    training=args.training_code.resolve();root=args.root.resolve();data=args.data_root.resolve()
    # 所有CUDA子进程仍用原clean版本；本文件只改变资源准入与监督。
    sys.path.insert(0,str(training))
    from tools import baseline_protocol as p
    from tools.run_sada_baseline import acquire,spawn
    from tools.background_oracle_protocol import audit
    from libs.models.SADA.background_oracle import SCHEMA,POLICY
    scheduler=Path(__file__).resolve().parents[1]
    scheduler_commit=subprocess.check_output(['git','-C',str(scheduler),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(scheduler),'status','--porcelain'],text=True).strip():
        raise ValueError('调度代码必须clean')
    identity=p.code_identity();plan=p.read(root/'plan.json');original_state=p.read(root/'state.json')
    if identity!=plan['code']:raise ValueError('原训练代码改变')
    validate_snapshot(original_state,root,args.parent_pid)
    reports=[p.read(root/'formal'/s/'probe_result.json') for s in original_state['active']]
    minimum=memory_requirement(reports)
    if minimum!=8192:raise ValueError('实际峰值不支持本次8GiB调度')
    configs={s:p.read(root/'configs'/(s+'.json')) for s in plan['shifts']}
    active={};completed=[];pending=list(original_state['pending']);old_stopped=False;replaced=False
    ledger=root/'monitor/resource_takeover.json'
    if ledger.exists():raise ValueError('不能重复接管')
    proof=dict(status='preparing',old_pid=args.parent_pid,new_pid=os.getpid(),scheduler_commit=scheduler_commit,
        training_code=identity,original_state=original_state,minimum_free_mib=minimum,
        preserve_live_training=True,training_parameters_changed=False)
    def stop(signum,frame):
        for task in active.values():
            if process_identity(task['pid']) is not None:
                try:os.killpg(task['pid'],signal.SIGTERM)
                except ProcessLookupError:pass
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        # 冻结的只是CPU调度父进程；CUDA子进程不会收到信号。
        os.kill(args.parent_pid,signal.SIGSTOP);old_stopped=True
        deadline=time.monotonic()+5
        while process_identity(args.parent_pid)['state'] not in ('T','t'):
            if time.monotonic()>deadline:raise RuntimeError('旧调度未停稳')
            time.sleep(.01)
        frozen=p.read(root/'state.json');validate_snapshot(frozen,root,args.parent_pid)
        if frozen!=original_state:raise ValueError('接管期间原调度已切换阶段，恢复原调度')
        # 先拿第四张卡的真实锁和显存准入，拿不到就恢复旧父，不干预训练。
        fourth=acquire([3],data,root,minimum,identity['commit'])
        if not fourth:raise BlockingIOError('GPU3当前未达到8GiB准入或锁仍被占用')
        fourth_context,fourth_lease=fourth
        adopted={s:dict(task,identity=process_identity(task['pid'])) for s,task in frozen['active'].items()}
        proof.update(status='ready',adopted=adopted,fourth_lease=fourth_lease);p.save(ledger,proof)
        # 精确退出旧CPU父，不执行其会终止children的SIGTERM处理器。
        os.kill(args.parent_pid,signal.SIGKILL);replaced=True
        for shift,task in adopted.items():
            now=process_identity(task['pid'])
            if now is None or now['start']!=task['identity']['start']:
                raise ValueError('接管时训练PID发生变化')
            active[shift]=dict(pid=task['pid'],stage='train',gpu=task['gpu'],entry=root/'formal'/shift,
                adopted=True,start=now['start'],process=None,context=None,lease=None)
        # 重新取得原三张卡锁；已有train的成功凭证继续绑定原lease，后续评估用新lease。
        while any(t['context'] is None for t in active.values()):
            for task in active.values():
                if task['context'] is None:
                    found=acquire([task['gpu']],data,root,minimum,identity['commit'])
                    if found:task['context'],task['lease']=found
            if any(t['context'] is None for t in active.values()):time.sleep(1)
        proof.update(status='passed',adoption_verified={s:process_identity(t['pid'])['start']==t['start'] for s,t in active.items()})
        p.save(ledger,proof)
        def launch(shift,context,lease):
            entry=root/'formal'/shift;entry.mkdir(parents=True,exist_ok=False);p.save(entry/'config.json',configs[shift])
            process=spawn('probe',entry/'config.json',entry,lease,data,entry/'probe_result.json')
            active[shift]=dict(pid=process.pid,process=process,stage='probe',gpu=lease['index'],entry=entry,
                adopted=False,context=context,lease=lease)
        pending.remove('S4');launch('S4',fourth_context,fourth_lease)
        print('FOURTH_GPU_STARTED','S4',active['S4']['pid'],flush=True)
        while pending or active:
            for shift,task in list(active.items()):
                entry=task['entry'];stage=task['stage'];process=task['process']
                if process is None:
                    now=process_identity(task['pid'])
                    if now is not None and now['start']==task['start'] and now['state']!='Z':continue
                    # 非父进程无法waitpid；child成功返回才写*_gpu.json，是强成功凭证。
                    if not (entry/'train_gpu.json').exists() or not (entry/'termination.json').exists():
                        raise RuntimeError(shift+' adopted train没有合法成功证明；保留现场')
                else:
                    if process.poll() is None:continue
                    if process.returncode:raise RuntimeError(shift+' '+stage+' failed')
                if stage=='probe':
                    report=p.read(entry/'probe_result.json');actual=report['phases']['5'].get('background_oracle',{})
                    if report['status']!='passed' or sum(v for k,v in actual.items() if k.startswith('oracle_bg_removed_'))<=0:
                        raise ValueError('真实oracle验收不通过')
                    if report['rng_roundtrip']['status']!='passed' or not all(x['branch_parameters_updated'] for x in report['phases'].values()):
                        raise ValueError('真实恢复/分支更新验收不通过')
                    if report['initial_detector_sha']!=p.read(Path(plan['control'])/'formal/control'/shift/'detector_initialization.json')['sha256']:
                        raise ValueError('冷初始化不配对')
                    p.save(entry/'acceptance.json',dict(status='passed',schema=SCHEMA,code=identity,batch=2,seed=123,
                        target_gt_used=True,oracle_policy=POLICY,config_sha256=p.sha(entry/'config.json'),
                        report=str(entry/'probe_result.json'),report_sha256=p.sha(entry/'probe_result.json')))
                    next_stage='train'
                elif stage=='train':next_stage='evaluate'
                else:
                    names=['config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl',
                        'termination.json','best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','probe_gpu.json',
                        'probe_result.json','acceptance.json','evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json']
                    reference=Path(plan['control'])/'formal/control'/shift/'complete.json'
                    p.save(entry/'complete.json',dict(schema=SCHEMA,shift=shift,regime='uda',variant='O',code=identity,
                        target_gt_used=True,oracle_policy=POLICY,control_complete=str(reference),control_complete_sha256=p.sha(reference),
                        gpu_acceptance=str(entry/'acceptance.json'),acceptance_sha256=p.sha(entry/'acceptance.json'),
                        artifacts={str(entry/n):p.sha(entry/n) for n in names}))
                    completed.append(audit(entry));task['context'].__exit__(None,None,None);del active[shift]
                    print('ORACLE_COMPLETE',shift,completed[-1]['map'],flush=True);continue
                task['stage']=next_stage;task['adopted']=False
                task['process']=spawn(next_stage,entry/'config.json',entry,task['lease'],data);task['pid']=task['process'].pid
                print('STARTED',shift,next_stage,task['pid'],task['gpu'],flush=True)
            for shift in list(pending):
                found=acquire(plan['gpus'],data,root,minimum,identity['commit'])
                if not found:break
                pending.remove(shift);launch(shift,*found)
            p.save(root/'state.json',dict(phase='oracle_only',pid=os.getpid(),completed=completed,pending=pending,
                active={s:dict(pid=t['pid'],gpu=t['gpu'],stage=t['stage']) for s,t in active.items()},
                target_gt_used=True,minimum_free_mib=minimum,scheduler_commit=scheduler_commit))
            time.sleep(5)
        if {x['shift'] for x in completed}!=set(configs):raise ValueError('不是精确六方向')
        p.save(root/'final_audit.json',dict(status='passed',schema=SCHEMA,count=6,target_gt_used=True,results=completed))
        p.save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={},target_gt_used=True))
    except BaseException as error:
        if old_stopped and not replaced:os.kill(args.parent_pid,signal.SIGCONT)
        if replaced:
            p.save(root/'resource_takeover_error.json',dict(error=repr(error),active={s:t['pid'] for s,t in active.items()}))
            stop(signal.SIGTERM,None)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--training-code',type=Path,required=True)
    parser.add_argument('--parent-pid',type=int,required=True)
    parser.add_argument('--data-root',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'))
    main(parser.parse_args())
