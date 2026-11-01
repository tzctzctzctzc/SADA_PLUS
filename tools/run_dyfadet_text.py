"""DyFADet简单UDA四卡共享队列，含等源更新次数对照。"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CONTROLLER_CODE=Path(__file__).resolve().parents[1]
CODE=Path(os.environ.get('DYFA_WORKER_CODE',str(CONTROLLER_CODE)))
sys.path.insert(0,str(CODE))
from tools.dyfadet_text_protocol import SPEC,SCHEMA,read,save,sha,identity,prepare,audit,guard_audit
from tools.dyfadet_uda_gpu import admit,first_cuda_guard

def run(args):
    code=identity();root=args.output
    if args.resume_existing:
        assert read(root/'plan.json')['code']==code,'resume requires same committed code'
        save(root/('resume_launch_%d.json'%int(time.time())),dict(pid=os.getpid(),code=code,controller_code=str(CONTROLLER_CODE),max_gpu_workers=args.max_gpu_workers))
    else:prepare(root,code)
    shifts=['S'+str(i) for i in range(1,7)]
    active={};done={};probe_pending=[];pending=[];evals=[]
    for s in shifts:
        e=root/'formal'/s
        if (e/'complete.json').exists():done[s]=audit(e)
        elif (e/'gpu_complete.json').exists():evals.append(s)
        else:pending.append(s)
        if not (root/'probes'/s/'probe.json').exists():probe_pending.append(s)
    accepted=(root/'acceptance.json').exists()
    minimum=read(root/'acceptance.json')['minimum_free_mib'] if accepted else 12288
    def launch(s,stage,card=None):
        e=root/('probes' if stage=='probe' else 'formal')/s;e.mkdir(parents=True,exist_ok=True)
        if stage=='probe':save(e/'config.json',read(root/'formal'/s/'config.json'))
        env=dict(os.environ,PYTHONPATH=str(CODE)+':'+args.extension,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONHASHSEED='123',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        if card:env.update(CUDA_VISIBLE_DEVICES=card['uuid'],SADA_BASELINE_LEASE=card['lease'],SADA_BASELINE_LEASE_SHA256=card['lease_sha256'])
        log=(e/(stage+'.log')).open('a')
        p=subprocess.Popen([sys.executable,'-u','-B',str(CODE/'tools/run_dyfadet_text.py'),'--output',str(root),'--entry',str(e),'--stage',stage,'--extension',args.extension],
            cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
        return dict(process=p,stage=stage,shift=s,gpu=card['index'] if card else None)
    def acquire():
        busy={t['gpu'] for t in active.values() if t['gpu'] is not None}
        if len(busy)>=args.max_gpu_workers:return None
        for index in ([0,2,1,3] if SPEC['text_variant']=='tifad' else [1,3,0,2]):
            if index in busy:continue
            try:card=admit(index,root,minimum,code['commit'])
            except BlockingIOError:continue
            return card
    try:
        while probe_pending or pending or evals or active:
            for key,t in list(active.items()):
                rc=t['process'].poll()
                if rc is None:continue
                if t.get('context'):t['context'].__exit__(None,None,None)
                del active[key]
                if rc:raise RuntimeError('%s failed exit=%d'%(key,rc))
                if t['stage']=='probe':
                    report=read(root/'probes'/t['shift']/'probe.json');assert report['status']=='passed'
                    guard_audit(report['guard'],report['guard_sha256'],code['commit'])
                elif t['stage']=='gpu':evals.append(t['shift'])
                else:done[t['shift']]=audit(root/'formal'/t['shift'])
            if not accepted and all((root/'probes'/s/'probe.json').exists() for s in shifts) and not any(t['stage']=='probe' for t in active.values()):
                reports={s:read(root/'probes'/s/'probe.json') for s in shifts}
                assert all(x['status']=='passed' for x in reports.values())
                minimum=max(12288,int(max(x['peak_reserved_mib'] for x in reports.values()))+4096)
                save(root/'acceptance.json',dict(status='passed',code=code,spec=SPEC,minimum_free_mib=minimum,
                    reports={s:dict(path=str(root/'probes'/s/'probe.json'),sha256=sha(root/'probes'/s/'probe.json')) for s in shifts}))
                accepted=True
            queue=pending if accepted else probe_pending
            for s in list(queue):
                available=acquire()
                if not available:break
                card=available;t=launch(s,'gpu' if accepted else 'probe',card)
                active[s+'/'+t['stage']]=t;queue.remove(s)
            while evals and sum(t['stage']=='evaluate' for t in active.values())<1:
                s=evals.pop(0);active[s+'/evaluate']=launch(s,'evaluate')
            save(root/'state.json',dict(pid=os.getpid(),phase='formal' if accepted else 'gpu_acceptance',code=code,pending=pending,
                probe_pending=probe_pending,completed=list(done.values()),active={k:dict(pid=t['process'].pid,stage=t['stage'],shift=t['shift'],gpu=t['gpu']) for k,t in active.items()}))
            time.sleep(3)
        assert set(done)==set(shifts)
        results=[done[s] for s in shifts];mean=sum(x['map'] for x in results)/6
        save(root/'final_audit.json',dict(status='passed',count=6,results=results,mean_map=mean,
            reference_so_mean=sum(x['reference_so_map'] for x in results)/6,mean_delta_pp=sum(x['delta_pp'] for x in results)/6,source_control_mean=sum(x['control_map'] for x in results)/6,mean_delta_control_pp=sum(x['delta_control_pp'] for x in results)/6,spec=SPEC))
        save(root/'state.json',dict(phase='complete',active={},completed=results))
    except BaseException as err:
        save(root/'error.json',dict(error=repr(err),pid=os.getpid()));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        for t in active.values():
            try:t['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(t['process'].pid,signal.SIGKILL);t['process'].wait()
            if t.get('context'):t['context'].__exit__(None,None,None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--entry',type=Path)
    p.add_argument('--data',type=Path,default=Path('/vepfs/vlms/SADA_PLUS'));p.add_argument('--extension',required=True)
    p.add_argument('--stage',choices=['probe','gpu','evaluate']);p.add_argument('--resume-existing',action='store_true');p.add_argument('--max-gpu-workers',type=int,choices=[2,4],default=4);args=p.parse_args()
    sys.path.insert(0,args.extension)
    def stop(sig,frame):raise SystemExit(128+sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if args.stage:
        guard=first_cuda_guard() if args.stage in ('probe','gpu') else None
        from tools import dyfadet_text_engine as engine
        os.chdir(args.data)
        if args.stage=='evaluate':engine.evaluate(args.entry)
        else:
            cfg=read(args.entry/'config.json')
            if args.stage=='probe':engine.probe(cfg,args.entry,guard)
            else:engine.train_and_infer(cfg,args.entry,guard)
    else:run(args)
