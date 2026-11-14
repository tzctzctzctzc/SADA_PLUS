"""六向一次性队列；私有互斥、最多四路，不占其他聊天的GPU锁。"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

CODE=Path(__file__).resolve().parents[1]


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)


def gpu_rows():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
    return [dict(index=int(a),uuid=b.strip(),free_mib=int(c)) for a,b,c in (line.split(',') for line in lines)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);args=parser.parse_args()
    root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/'queue_state.json').exists():raise RuntimeError('已有队列记录；先核对原进程与claim，禁止重新提交')
    mutex=threading.RLock();cancel=threading.Event();evaluation=threading.Semaphore(2)
    auth=json.loads((root/'formal_authorization.json').read_text())
    acceptance=json.loads((root/'acceptance.json').read_text())
    assert auth['budget']==12 and acceptance['summary']['status']=='passed'
    for path,digest in acceptance['unchanged_model_sha256_across_harness_repairs'].items():
        assert hashlib.sha256((CODE/path).read_text().encode()).hexdigest()==digest,'已验收的训练源码发生变化：'+path
    for i in range(1,7):
        for route in ('time_frequency','mda'):
            report=acceptance['summary']['results'][route+'/S%d'%i]
            assert report['updates']==40
            if route=='mda':assert report['natural_aggregations']>=1
            else:assert report['target_batch_nonempty_fraction']==1
    state=dict(stage='running',budget=12,seed=123,max_gpu_parallel=4,strict_bitwise_required=False,
               commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
               jobs={f'{route}/S{i}':dict(stage='pending') for i in range(1,7) for route in ('time_frequency','mda')},failures=[])
    reservations={};using={};minimum=8192

    def continuing_gpu_jobs():
        # 同一聊天旧TF队列仍在收尾，合计最多四路；不查看外部作业。
        legacy=json.loads((Path(auth['continuing_tf_root'])/'queue_state.json').read_text())
        return sum(bool(j.get('gpu')) and j.get('stage') in ('gpu_acquired','probe','train','evaluate')
                   for key,j in legacy['jobs'].items() if key.startswith('time_frequency/'))

    def update(shift,**values):
        with mutex:
            state['jobs'][shift].update(values);state['checked_at_unix']=time.time()
            save(root/'queue_state.json',state)

    @contextmanager
    def acquire_gpu(shift):
        nonlocal minimum
        while True:
            if cancel.is_set():raise RuntimeError('其他条目失败，尚未启动的训练停止派发')
            with mutex:
                now=time.time();rows=gpu_rows()
                for row in rows:
                    row['available_mib']=row['free_mib']-sum(size for uid,size,when in reservations.values() if uid==row['uuid'] and now-when<90)
                legacy_count=continuing_gpu_jobs()
                available=[r for r in rows if r['available_mib']>=minimum] if sum(using.values())+legacy_count<4 else []
                if available:
                    device=max(available,key=lambda r:(-using.get(r['uuid'],0),r['available_mib']))
                    reservations[shift]=(device['uuid'],minimum,now)
                    using[device['uuid']]=using.get(device['uuid'],0)+1
                    break
            update(shift,stage='waiting_gpu',last_resource_check_unix=time.time(),continuing_legacy_gpu_jobs=legacy_count)
            cancel.wait(15)
        update(shift,stage='gpu_acquired',gpu=device)
        try:yield device
        finally:
            with mutex:
                using[device['uuid']]-=1;reservations.pop(shift,None)

    def run(shift,mode,entry,device=None):
        env=dict(os.environ,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES=device['uuid'] if device else '')
        cfg=root/shift/'config.json';entry.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-B','tools/run_uda_two_routes.py',mode,'--config',str(cfg),'--entry',str(entry)]
        stdout=(entry/(mode+'.log')).open('a')
        process=subprocess.Popen(command,cwd=CODE,env=env,stdout=stdout,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=int(Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19])
        receipt=dict(pid=process.pid,start_ticks=ticks,mode=mode,command=command,started_unix=time.time(),gpu=device)
        save(entry/(mode+'_launch.json'),receipt);update(shift,stage=mode,active=receipt)
        while True:
            try:
                code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                metric=root/shift/'metrics.jsonl';lines=metric.read_text().splitlines() if metric.exists() else []
                last=json.loads(lines[-1]) if lines else None
                log=entry/(mode+'.log')
                update(shift,last_process_check_unix=time.time(),last_metric=last,
                       latest_log_mtime=log.stat().st_mtime if log.exists() else None)
        stdout.close()
        save(entry/(mode+'_complete.json'),dict(returncode=code,completed_unix=time.time()))
        update(shift,active=None,last_completed=mode)
        if code:raise RuntimeError(f'{shift} {mode}退出码{code}，见{entry}/{mode}.log')

    def job(shift):
        nonlocal minimum
        entry=root/shift
        try:
            with acquire_gpu(shift) as device:
                # 六向短验收已完成，正式模型全新冷启动，不重复探针。
                if cancel.is_set():raise RuntimeError('其他条目失败，本项正式训练未派发')
                run(shift,'train',entry,device)
                with evaluation:
                    run(shift,'evaluate',entry,device)
            update(shift,stage='waiting_cpu_audit')
            with cpu_slots:run(shift,'audit',entry)
            update(shift,stage='complete',completed_unix=time.time())
        except Exception as error:
            cancel.set()
            with mutex:
                state['failures'].append(dict(shift=shift,error=str(error),traceback=traceback.format_exc()))
            update(shift,stage='failed',error=str(error))

    save(root/'queue_state.json',state);cpu_slots=threading.Semaphore(2)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(job,key) for key in state['jobs']]
        for future in as_completed(futures):future.result()
    if not state['failures']:
        records={key:json.loads((root/key/'state_audit.json').read_text()) for key in state['jobs']}
        summaries={}
        for route in ('time_frequency','mda'):
            values={key.split('/')[-1]:row['independent_AP']['target']['mean_map']*100 for key,row in records.items() if key.startswith(route+'/')}
            assert len(values)==6
            mean=sum(values.values())/6
            summaries[route]=dict(per_shift_percent=values,mean_percent=mean,delta_vs_prototype_pp=mean-54.73196872805816,target_plus1_achieved=mean>=55.73196872805816)
        save(root/'summary.json',dict(status='passed',groups=summaries,baseline_percent=54.73196872805816,formal_training_runs=12,seed=123,target_validation_development=True))
        state['stage']='complete'
    else:state['stage']='incomplete_requires_review'
    save(root/'queue_state.json',state)


if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
