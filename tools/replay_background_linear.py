"""不可变头的原CUDA精度逐值重放；另用禁TF32前向诊断CPU差异，不训练。"""
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


def replay(entry,source):
    import torch
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.baseline_engine import tensor_digest
    start=time.time();gpu=read(source/'gpu_complete.json')
    assert sha(source/'epoch020.pt')==gpu['saved']['20']['checkpoint_sha256'] and sha(source/'descriptors.pt')==gpu['descriptors_sha256']
    assert sha(source/'predictions020.pt')==gpu['saved']['20']['predictions_sha256']
    state=torch.load(source/'epoch020.pt',map_location='cpu');prepared=torch.load(source/'descriptors.pt',map_location='cpu');pred=torch.load(source/'predictions020.pt',map_location='cpu')
    guard=first_cuda_guard(entry);weights={m:{k:v.cuda() for k,v in s.items()} for m,s in state['heads'].items()}
    before=tensor_digest(weights);cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state(0)
    native_flag=torch.backends.cuda.matmul.allow_tf32;rows=[];max_native_cpu=0.;max_full_cpu=0.;changed=0
    with torch.no_grad():
        for role,modes in pred.items():
            items={v['video']:v for v in prepared[role]}
            for mode,vs in modes.items():
                w=weights[mode]['classifier.weight'];b=weights[mode]['classifier.bias']
                cw=state['heads'][mode]['classifier.weight'];cb=state['heads'][mode]['classifier.bias']
                for row in vs:
                    item=items[row['video']];cpu=item['descriptors'][mode];x=cpu.cuda();expected=row['pbg'][item['hard']]
                    native=torch.nn.functional.linear(x,w,b)[:,0].sigmoid().cpu()
                    assert torch.equal(native,expected), (role,mode,row['video'],'native CUDA replay differs')
                    try:
                        torch.backends.cuda.matmul.allow_tf32=False
                        full=torch.nn.functional.linear(x,w,b)[:,0].sigmoid().cpu()
                    finally:torch.backends.cuda.matmul.allow_tf32=native_flag
                    cpu_pred=torch.nn.functional.linear(cpu,cw,cb)[:,0].sigmoid()
                    ne=float((native-cpu_pred).abs().max()) if len(native) else 0.
                    fe=float((full-cpu_pred).abs().max()) if len(full) else 0.
                    assert torch.allclose(full,cpu_pred,atol=3e-6,rtol=1e-6), ('full precision CUDA/CPU mismatch',fe)
                    max_native_cpu=max(max_native_cpu,ne);max_full_cpu=max(max_full_cpu,fe);changed+=int(not torch.equal(native,full))
                    rows.append(dict(role=role,mode=mode,video=row['video'],candidates=len(native),native_bitwise_equal=True,native_cpu_max_error=ne,disabled_TF32_cpu_max_error=fe))
    assert before==tensor_digest(weights) and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state(0)) and native_flag==torch.backends.cuda.matmul.allow_tf32
    assert sha(source/'epoch020.pt')==gpu['saved']['20']['checkpoint_sha256']
    save(entry/'complete.json',dict(passed=True,source=str(source),source_artifacts={str(source/n):sha(source/n) for n in ('gpu_complete.json','epoch020.pt','descriptors.pt','predictions020.pt')},native_allow_tf32=native_flag,all_native_predictions_bitwise_equal=True,all_disabled_TF32_CPU_checks_within_3e6=True,parameters_and_CPUCUDA_RNG_unchanged=True,optimizer_steps=0,guard=guard,guard_sha256=sha(guard),max_native_CPU_error=max_native_cpu,max_disabled_TF32_CPU_error=max_full_cpu,native_vs_disabled_changed_video_mode_count=changed,rows=rows,seconds=time.time()-start,peak_mib=torch.cuda.max_memory_reserved()/2**20))
    print('REPLAYED',entry.name,len(rows),max_native_cpu,max_full_cpu,flush=True)


def run(root,source):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(source=str(source),code=code_identity(),optimizer_steps=0,GT_used=False));pending=[f'S{i}' for i in range(1,7)];active={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        while pending or active:
            for s,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[s]
                if t['p'].returncode:raise RuntimeError(s+' native replay failed')
            for g in (2,0,3,1):
                if not pending or g in [t['gpu'] for t in active.values()]:continue
                obs=snapshot(g)
                if obs['free_mib']<12288:continue
                s=pending.pop(0);e=root/s;e.mkdir();lease=e/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=obs['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=obs['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                log=(e/'replay.log').open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--source',str(source/s)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[s]=dict(p=p,gpu=g,log=log)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,active={s:dict(pid=t['p'].pid,gpu=t['gpu']) for s,t in active.items()}));time.sleep(2)
        save(root/'final.json',dict(passed=True,source=str(source),reports={f'S{i}':sha(root/f'S{i}'/'complete.json') for i in range(1,7)}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.entry:replay(a.entry,a.source)
    else:run(a.root,a.source)
