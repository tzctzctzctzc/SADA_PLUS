"""独立进程只读EMA20，完整重放全部原生/三尺度检测输出；不含优化器。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_detector_iacs import TASKS,READOUTS,SOURCE


def replay(entry,origin):
    import torch
    from tools import baseline_engine as base
    from tools.detector_iacs_math import QualityDetector
    from tools.detector_scale_math import rescale_raw,fuse_heads
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_dense_revisit import raw_load
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);old=origin/entry.name;g=read(old/'gpu_complete.json');s=entry.name.split('_')[0];p=old/'epoch020.pt';assert sha(p)==g['checkpoint_sha256'];state=torch.load(p,map_location='cpu');detector=load_detector(SOURCE/s);model=QualityDetector(detector).cuda();model.load_state_dict(state['ema']);model.eval();model.body_mode(False);model.requires_grad_(False)
    data=torch.load(origin/'inputs'/(s+'_unlabeled.pt'),map_location='cpu')['evaluation'];expected=torch.load(old/'predictions.pt',map_location='cpu');assert sha(old/'predictions.pt')==g['predictions_sha256'];digest=base.tensor_digest(model.state_dict());assert digest==g['final_EMA_digest'];cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state(0);cases=[];started=time.monotonic()
    with torch.no_grad():
        for role,rows in data.items():
            for i,r in enumerate(rows):
                raw=raw_load(r).cuda();f,v,c,z,_=model.forward_heads(raw,r['video_metadata']);points=model.point_generator([a[None] for a in f]);video=[dict(feats=raw.T,segments=raw.new_empty(0,2),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=r['video_metadata'])]
                def decode(cc,zz):return model.inference(video,points,[a[None] for a in v],[a[None] for a in cc],[a[None] for a in zz],None,None)[0]
                views=[]
                for scale in (.75,1.25):
                    x,ratio=rescale_raw(raw,scale);_,sv,sc,sr,_=model.forward_heads(x,r['video_metadata']);views.append((sv,sc,sr,ratio))
                fc=[];fr=[]
                for l in range(6):
                    a,b=fuse_heads(c[l].T,z[l].T,[(vc[l].T,vr[l].T,vm[l],ratio) for vm,vc,vr,ratio in views]);fc.append(a.T);fr.append(b.T)
                for name,out in [('native',decode(c,z)),('scale_score',decode(fc,z)),('scale_boundary',decode(fc,fr))]:
                    ref=expected[role][name][i];assert ref['video']==r['video']
                    for key in ('segments','scores','labels'):assert torch.equal(out[key].cpu(),ref[key]),(role,r['video'],name,key)
                cases.append(dict(role=role,video=r['video'],all_three_methods_and_arrays_bitwise_equal=True))
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(0))
    save(entry/'complete.json',dict(passed=True,origin_GPU_sha256=sha(old/'gpu_complete.json'),checkpoint_sha256=sha(p),predictions_sha256=sha(old/'predictions.pt'),guard=str(guard),guard_sha256=sha(guard),EMA_digest_unchanged=digest,CPUCUDA_RNG_unchanged=True,optimizer_steps=0,cases=cases,seconds=time.monotonic()-started))


def run(root,origin):
    from tools.baseline_gpu import snapshot
    root.mkdir(exist_ok=False);save(root/'plan.json',dict(code=code_identity(),origin=str(origin),scope='full4savedEMA replay, originalinference reused, nooptimizer or GT'));pending=list(TASKS);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:
        while pending or active:
            if (origin/'error.json').exists():raise RuntimeError('source producer failed; no replay')
            for t,a in list(active.items()):
                if a['p'].poll() is None:continue
                a['log'].close();del active[t];assert a['p'].returncode==0,t;done.append(t)
            # 所有正式GPU任务退出后才占卡重放，不与正式训练竞争。
            ready=all((origin/t/'gpu_complete.json').exists() for t in TASKS) and read(origin/'state.json')['stage']!='train'
            for g in range(4) if ready else []:
                if g in [a['gpu'] for a in active.values()] or not pending:continue
                v=snapshot(g)
                if v['free_mib']<12288:continue
                t=pending.pop(0);e=root/t;e.mkdir();lease=e/'lease_train.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env=dict(os.environ,CUDA_VISIBLE_DEVICES=v['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease));log=(e/'replay.log').open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--source',str(origin)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[t]=dict(p=p,gpu=g,log=log)
            save(root/'state.json',dict(owner_pid=os.getpid(),pending=pending,active={t:dict(pid=a['p'].pid,gpu=a['gpu']) for t,a in active.items()},completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,results_sha256={t:sha(root/t/'complete.json') for t in TASKS},exact_tasks=4,optimizer_steps=0))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex),automatic_restart_authorized=False))
        for a in active.values():
            if a['p'].poll() is None:os.killpg(a['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--source',type=Path,required=True);p.add_argument('--entry',type=Path);a=p.parse_args();replay(a.entry,a.source) if a.entry else run(a.root,a.source)
