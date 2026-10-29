"""四卡接管：原监督只暂停派单，现有子任务自然结束；训练/评估仍执行原e569239代码。"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SCRIPT_CODE=Path(__file__).resolve().parents[1]
JOBS={(v,'S'+str(i)) for v in ('temporal_pool','multi_prototype') for i in range(1,7)}


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True));tmp.replace(path)


def process_state(pid):
    path=Path('/proc')/str(pid)/'stat'
    try:fields=path.read_text().rsplit(')',1)[1].split()
    except FileNotFoundError:return dict(alive=False,exit_status=None)
    return dict(alive=fields[0]!='Z',state=fields[0],exit_status=int(fields[49]) if fields[0]=='Z' else None)


def stop_parent_after_children(parent):
    if not process_state(parent)['alive']:return
    os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    while process_state(parent)['alive']:time.sleep(.2)


def wait_natural_children(tasks,status_path):
    # 原父进程被SIGSTOP，退出的直接子进程保持zombie，可读取真实wait status。
    while True:
        status={k:process_state(t['pid']) for k,t in tasks.items()}
        save(status_path,status)
        if all(not x['alive'] for x in status.values()):return status
        time.sleep(5)


def freeze_snapshot(previous,training_code):
    previous=Path(previous)
    for _ in range(30):
        state=read(previous/'state.json');parent=state['pid']
        if state['phase']=='complete':return dict(parent=parent,tasks={},already_complete=True)
        if state['phase']!='formal' or not state['active']:time.sleep(1);continue
        proc=Path('/proc')/str(parent)
        command=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()
        if str(training_code/'tools/run_prototype_methods.py') not in command or str(previous) not in command:
            raise ValueError('旧监督进程身份不符')
        os.kill(parent,signal.SIGSTOP)
        try:
            state=read(previous/'state.json');actual=set(map(int,(proc/'task'/str(parent)/'children').read_text().split()))
            expected={int(x['pid']) for x in state['active'].values()}
            if actual!=expected:raise ValueError('快照处于派单边界，重试')
            tasks={}
            for key,task in state['active'].items():
                if task['stage'] not in ('train','evaluate') or not process_state(task['pid'])['alive']:
                    raise ValueError('子进程正在收尾，重试')
                v,shift=key.rsplit('_',1);entry=previous/'formal'/v/shift
                cmd=(Path('/proc')/str(task['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if str(entry) not in cmd or '--child '+task['stage'] not in cmd:raise ValueError('阶段与真实命令不符')
                tasks[key]=dict(task,entry=str(entry),variant=v,shift=shift)
            return dict(parent=parent,tasks=tasks,already_complete=False)
        except ValueError:
            os.kill(parent,signal.SIGCONT);time.sleep(1)
        except BaseException:
            os.kill(parent,signal.SIGCONT);raise
    raise RuntimeError('未能在一致状态冻结旧监督，未接管')


def bridge(args):
    snap=freeze_snapshot(args.previous_root,args.training_code)
    snap.update(bridge_pid=os.getpid(),previous_root=str(args.previous_root),training_code=str(args.training_code))
    def cancel(signum,frame):
        # 用户停止时才让旧handler停止其自有子组；不触碰NVML外部PID。
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        stop_parent_after_children(snap['parent']);raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    save(args.output_root/'handoff_ready.json',snap)
    status=wait_natural_children(snap['tasks'],args.output_root/'handoff_child_status.json')
    # 两路自然退出后才终止旧派单监督，其handler不会打断任何训练。
    stop_parent_after_children(snap['parent'])
    success=all(x['exit_status']==0 for x in status.values())
    save(args.output_root/'handoff_done.json',dict(status='passed' if success else 'failed',
        snapshot=snap,children=status,old_parent_exited=True))


def pending_jobs(completed,inflight):
    if completed & inflight or not (completed|inflight)<=JOBS:raise ValueError('前序身份重复或越界')
    return sorted(JOBS-completed-inflight,key=lambda x:(int(x[1][1:]),x[0]))


def exact_spawn(mode,entry,lease,data,training_code):
    cfg=read(entry/'config.json')
    env=dict(os.environ,PYTHONPATH=str(training_code),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
        PYTHONHASHSEED=str(cfg['seed']),CUDA_VISIBLE_DEVICES=lease['uuid'],OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',
        SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'],
        TMPDIR=cfg['baseline']['tmpdir'],TMP=cfg['baseline']['tmpdir'],TEMP=cfg['baseline']['tmpdir'],CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ['RQ3_EXTERNAL_GPU_LEASE','RQ3_EXTERNAL_GPU_LEASE_SHA256','RQ3_COHORT_LEASE']:env.pop(key,None)
    command=[sys.executable,'-B',str(training_code/'tools/run_sada_baseline.py'),'--child',mode,
        '--config',str(entry/'config.json'),'--entry',str(entry),'--data-root',str(data)]
    with (entry/(mode+'.log')).open('x') as stream:
        return subprocess.Popen(command,cwd=training_code,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)


def controller(args):
    sys.path.insert(0,str(SCRIPT_CODE))
    from tools import baseline_protocol as protocol
    from tools.run_sada_baseline import acquire
    from tools.run_prototype_capacity import paired_config,assert_pair
    from tools.run_prototype_methods import audit,audit_control
    root=args.output_root;old=args.previous_root;training=args.training_code;data=args.data_root
    scheduler=protocol.code_identity();identity=read(old/'inputs.json')['code']
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=training,text=True).strip()!=identity['commit']:
        raise ValueError('训练代码版本改变')
    if subprocess.check_output(['git','status','--porcelain'],cwd=training,text=True).strip():raise ValueError('训练源码非clean')
    changed=subprocess.check_output(['git','diff',identity['commit'],scheduler['commit'],'--','libs','tools/baseline_engine.py','tools/baseline_gpu.py','tools/baseline_protocol.py','tools/run_sada_baseline.py','tools/run_prototype_methods.py'],cwd=SCRIPT_CODE,text=True)
    if changed:raise ValueError('接管不能改变任何模型、数据、优化、审计或训练代码')
    snap=read(root/'handoff_ready.json');inflight={(t['variant'],t['shift']) for t in snap['tasks'].values()}
    completed=[]
    for path in sorted(old.glob('formal/*/*/complete.json')):completed.append(audit(path.parent))
    done={(x['variant'],x['shift']) for x in completed};pending=pending_jobs(done,inflight)
    for name in ('inputs.json','acceptance.json'):shutil.copyfile(old/name,root/name)
    acceptance=read(root/'acceptance.json')
    if acceptance['status']!='passed' or protocol.sha(root/'inputs.json')!=acceptance['inputs_sha256']:
        raise ValueError('原GPU验收与输入不匹配')
    if not all(protocol.sha(p)==v for p,v in acceptance['report_sha256'].items()):raise ValueError('原GPU报告SHA改变')
    for shift in ['S'+str(i) for i in range(1,7)]:audit_control(args.control_root/'formal/control'/shift)
    save(root/'orchestration.json',dict(status='passed',scheduler_code=scheduler,training_code=identity,
        acceptance_reused_from=str(old/'acceptance.json'),acceptance_sha256=protocol.sha(root/'acceptance.json'),
        reused_completed=completed,inflight=snap['tasks'],scientific_code_unchanged=True))
    save(root/'plan.json',dict(schema='rq3-four-card-exact-handoff-v1',jobs=sorted(JOBS),fresh_jobs=pending,
        original_inflight=list(inflight),gpus={'temporal_pool':[0,2],'multi_prototype':[1,3]}))
    prepared=read(args.prepared_root/'audit.json');minimum=acceptance['minimum_free_mib'];active={};inherited=False;eval_pending=[]
    def finish(entry,variant,shift):
        paths=('config.json','runtime_config.json','seed_proof.json','detector_initialization.json','metrics.jsonl','termination.json',
            'best.pt','latest.pt','train_gpu.json','evaluate_gpu.json','train.log','evaluate.log',
            'evaluation/source_val.json','evaluation/target_val.json','evaluation/manifest.json')
        base=args.baseline_root/'formal'/shift/'uda/complete.json';control=args.control_root/'formal/control'/shift/'complete.json'
        save(entry/'complete.json',dict(schema='rq3-prototype-independent-pool-multi-seed123-v1',shift=shift,regime='uda',variant=variant,
            code=identity,orchestration_code=scheduler,acceptance_sha256=protocol.sha(root/'acceptance.json'),
            baseline_complete=str(base),baseline_complete_sha256=protocol.sha(base),control_complete=str(control),
            control_complete_sha256=protocol.sha(control),artifacts={str(entry/p):protocol.sha(entry/p) for p in paths}))
        result=audit(entry);completed.append(result);print('COMPLETE',variant,shift,result['map'],flush=True)
    requested_stop=False
    def stop(signum,frame):
        nonlocal requested_stop
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        requested_stop=True;raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or active or not inherited or eval_pending:
            if not inherited and (root/'handoff_done.json').exists():
                receipt=read(root/'handoff_done.json')
                if receipt['status']!='passed':raise ValueError('旧任务非正常退出，保留训练现场，不重训')
                for task in snap['tasks'].values():
                    entry=Path(task['entry'])
                    if task['stage']=='train':
                        protocol.termination(read(entry/'termination.json')['history'])
                        if not (entry/'train_gpu.json').exists():raise ValueError('旧训练没有完整GPU结束凭证')
                        eval_pending.append((task['variant'],task['shift'],entry))
                    else:finish(entry,task['variant'],task['shift'])
                inherited=True
            for gpu,task in list(active.items()):
                if task['process'].poll() is None:continue
                if task['process'].returncode:raise RuntimeError('阶段失败: '+str(task['entry'])+'/'+task['stage'])
                if task['stage']=='train':
                    task['stage']='evaluate';task['process']=exact_spawn('evaluate',task['entry'],task['lease'],data,training);continue
                finish(task['entry'],task['variant'],task['shift']);task['context'].__exit__(None,None,None);del active[gpu]
            for gpu,variant in [(0,'temporal_pool'),(1,'multi_prototype'),(2,'temporal_pool'),(3,'multi_prototype')]:
                if gpu in active:continue
                evaluation=next((x for x in eval_pending if x[0]==variant),None)
                fresh=next((x for x in pending if x[0]==variant),None)
                if evaluation is None and fresh is None:continue
                found=acquire([gpu],data,root,minimum,identity['commit'])
                if not found:continue
                context,lease=found
                try:
                    if evaluation:
                        _,shift,entry=evaluation;stage='evaluate';eval_pending.remove(evaluation)
                    else:
                        _,shift=fresh;entry=root/'formal'/variant/shift;entry.mkdir(parents=True,exist_ok=False)
                        cfg=paired_config(read(args.baseline_root/'formal'/shift/'uda/config.json'),variant,
                            Path(prepared['entries'][shift]['path']),identity,root/'tmp')
                        assert_pair(cfg,read(args.baseline_root/'formal'/shift/'uda/config.json'))
                        (root/'tmp').mkdir(exist_ok=True);save(entry/'config.json',cfg);stage='train';pending.remove(fresh)
                    process=exact_spawn(stage,entry,lease,data,training)
                except BaseException:context.__exit__(None,None,None);raise
                active[gpu]=dict(entry=entry,variant=variant,shift=shift,stage=stage,context=context,lease=lease,process=process)
            save(root/'state.json',dict(phase='formal',pid=os.getpid(),completed=completed,pending=pending,
                legacy_inflight=[] if inherited else snap['tasks'],eval_pending=[(v,s,str(e)) for v,s,e in eval_pending],
                active={v['variant']+'_'+v['shift']:dict(gpu=g,pid=v['process'].pid,stage=v['stage']) for g,v in active.items()}))
            time.sleep(5)
        if {(x['variant'],x['shift']) for x in completed}!=JOBS:raise ValueError('完成身份不是精确12项')
        for variant in ('temporal_pool','multi_prototype'):
            rows=[audit(Path(x['complete']).parent) for x in completed if x['variant']==variant]
            save(root/variant/'final_audit.json',dict(status='passed',count=6,results=rows))
        save(root/'final_audit.json',dict(status='passed',count=12,results=completed,orchestration_sha256=protocol.sha(root/'orchestration.json')))
        save(root/'state.json',dict(phase='complete',pid=os.getpid(),completed=completed,active={}))
    finally:
        for task in active.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        for task in active.values():
            try:task['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(task['process'].pid,signal.SIGKILL);task['process'].wait()
            task['context'].__exit__(None,None,None)
        if requested_stop and not inherited and process_state(args.bridge_pid)['alive']:os.kill(args.bridge_pid,signal.SIGTERM)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bridge',action='store_true');p.add_argument('--bridge-pid',type=int)
    for key in ['output-root','previous-root','training-code','data-root','baseline-root','control-root','prepared-root']:
        p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    try:
        if args.bridge:bridge(args)
        else:controller(args)
    except BaseException as error:
        save(args.output_root/('bridge_error.json' if args.bridge else 'error.json'),dict(type=type(error).__name__,error=str(error)))
        raise
