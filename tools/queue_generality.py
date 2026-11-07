"""四卡独立队列；不启动旧队列，不改变正在运行的源码。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.baseline_gpu import slot


def ticks(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()[19]


def run(root,tasks_json=None):
    import fcntl
    identity=code_identity();commit=identity['commit']
    own=(root/'controller.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'state.json').exists(),'existing state: inspect, never silently resubmit'
    pending=[f'{d}/{c}/S{s}' for s in (1,3,2,4,5,6)
             for d in ('actionformer','tridet') for c in ('so','sada','sada_prototype')]
    if tasks_json:
        selected=read(tasks_json)
        assert len(selected)==len(set(selected)) and set(selected)<=set(pending)
        pending=selected
    stages=['probe','train','evaluate','audit'];active={};complete=[];failed={};formal=0
    cancelled=False
    def state():
        save(root/'state.json',dict(stage='running' if active or pending else 'complete' if not failed else 'requires_review',
            owner_pid=os.getpid(),owner_start_ticks=ticks(os.getpid()),commit=commit,
            pending=pending,completed=complete,failed=failed,formal_launches=formal,
            active={k:{q:v[q] for q in ('stage','pid','start_ticks','gpu','started')} for k,v in active.items()},
            updated=time.time()))
    def spawn(task,stage,gpu,lease):
        nonlocal formal
        entry=root/task
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if stage=='audit' else lease['uuid'],
            PYTHONPATH=str(CODE),PYTHONHASHSEED='123',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',
            SADA_BASELINE_LEASE=lease['lease'],SADA_BASELINE_LEASE_SHA256=lease['lease_sha256'])
        with (entry/(stage+'.log')).open('x') as output:
            p=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/run_generality.py'),stage,'--entry',str(entry)],
                env=env,cwd='/vepfs/vlms/SADA_PLUS',stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(stage=stage,pid=p.pid,start_ticks=ticks(p.pid),gpu=gpu,started=time.time())
        save(entry/(stage+'_launch.json'),dict(record,commit=commit))
        if stage=='train':formal+=1
        return dict(record,p=p,lease=lease)
    def stop(sig,frame):
        save(root/'user_stop.json',dict(signal=sig,automatic_resume=False))
        for row in active.values():
            if row['p'].poll() is None:os.killpg(row['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    state()
    while active or pending:
        for task,row in list(active.items()):
            code=row['p'].poll()
            if code is None:continue
            stage=row['stage'];lease=row['lease'];ctx=row['lock']
            if code:
                failed[task]=dict(stage=stage,exitcode=code,log=str(root/task/(stage+'.log')))
                cancelled=True;ctx.__exit__(None,None,None);del active[task]
            elif stage=='audit':
                complete.append(task);ctx.__exit__(None,None,None);del active[task]
            elif cancelled and stage=='probe':
                failed[task]=dict(stage='held_after_other_probe_failure',formal_training_started=False)
                ctx.__exit__(None,None,None);del active[task]
            else:
                row=spawn(task,stages[stages.index(stage)+1],row['gpu'],lease)
                row['lock']=ctx;active[task]=row
            state()
        if cancelled and pending:
            save(root/'unstarted_cancelled.json',dict(tasks=pending,reason='implementation failure requires review'))
            pending=[];state()
        used={v['gpu'] for v in active.values()}
        for gpu in range(4):
            if not pending or gpu in used:continue
            ctx=slot(gpu,'/vepfs/vlms/SADA_PLUS',root,16384,commit)
            try:lease=ctx.__enter__()
            except BlockingIOError:continue
            task=pending.pop(0)
            try:
                row=spawn(task,'probe',gpu,lease);row['lock']=ctx;active[task]=row
            except BaseException:
                ctx.__exit__(*sys.exc_info());raise
            state()
        if active or pending:time.sleep(10)
    summaries={}
    for detector in ('actionformer','tridet'):
        for condition in ('so','sada','sada_prototype'):
            rows={}
            for s in range(1,7):
                path=root/detector/condition/f'S{s}'/'audit.json'
                if path.exists():rows[f'S{s}']=read(path)['results']['target']
            summaries[detector+'/'+condition]=dict(count=len(rows),rows=rows,
                mean_percent=sum(v['map_0.1_to_0.5'] for v in rows.values())/6*100 if len(rows)==6 else None)
    save(root/'summary.json',dict(all_36_complete=len(complete)==36,formal_launches=formal,
        summaries=summaries,failed=failed,seed=123,original_prototype=True,
        target_validation_hyperparameter_selection=False))
    state()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--tasks-json');a=p.parse_args();run(Path(a.root),a.tasks_json)
