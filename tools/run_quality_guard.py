"""固定质量保护读出：只读已终审native20，八次GPU读取而非再次训练。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools import run_native_joint as native
ORIGIN=Path('/vepfs/vlms/highscore_native_full20_706d076_1103')
BASEMODES=tuple(native.MODES);JOBS=tuple(native.TASKS)
MODES=BASEMODES+tuple(m+'guard' for m in BASEMODES)
SPEC=dict(schema='native-quality-protection-readout-v1',origin=str(ORIGIN),modes=MODES,
    risk='mean(sigmoid(BGlogits)) * (1 - mean(sigmoid(sameclassIoUqualitylogits)))',
    rationale='joint task quality output was trained but not consumed by BG-only deletion; fixed product tests foreground protection, not calibrated probability',
    epochs=0,optimizer_steps=0,all_four_existing_models_retained=True,rates=[.005,.01,.02,.05],
    precision='inherit originalnative TF32 environment; demand complete originalpBG bitwise replay',
    target_train_GT_read=False,target_train_used=False,original_score_class_endpoints_unchanged=True,
    selection='same original perclass quota and eligibility; high risk deleted; tie originalindex',
    acceptance='fixed5percent two-shift mean>=1pp, both positive and above samecountconfidence; report all methods/rates',
    scope='S1/S2 historical development; followup designed after native20 negative results, not independent validation',
    budget_seconds=300)


def protected_risk(pbg,quality):
    return pbg*(1-quality)


def replay(entry):
    import torch
    from tools import baseline_engine as base
    from tools.native_joint_math import NativeBackground
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector
    guard=first_cuda_guard(entry);s,mode=entry.name.split('_');began=time.monotonic()
    audited=read(ORIGIN/'independent_final_audit.json');assert audited['passed']
    assert sha(ORIGIN/'final.json')==audited['production_final_sha256']
    old=ORIGIN/entry.name;gpu=read(old/'gpu_complete.json');checkpoint=old/'epoch020.pt'
    assert sha(checkpoint)==gpu['checkpoint_sha256'] and sha(old/'predictions.pt')==gpu['predictions_sha256']
    roles=torch.load(native.DENSE/'inputs'/s/'data.pt',map_location='cpu')['roles']
    previous=torch.load(old/'predictions.pt',map_location='cpu')
    detector=load_detector(native.SOURCE/s);detector.requires_grad_(False);native.seed()
    model=NativeBackground(detector).cuda();model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model'])
    model.requires_grad_(False);model.eval();model.body_mode(False)
    digest=base.tensor_digest(model.state_dict());assert digest==gpu['final_digest']
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def output():
        result={};count=0;total=sum(len(roles[k]) for k in ('fit','heldout','validation','target_val'));pass_began=time.monotonic()
        with torch.no_grad():
            for role in ('fit','heldout','validation','target_val'):
                result[role]=[]
                for r,prior in zip(roles[role],previous[role]):
                    x=native.raw_load(r).cuda();h=r['hard'];seq,valid,length=model.sample(x,r['video_metadata'],r['grid_boxes'][h].cuda())
                    meta=torch.cat((torch.nn.functional.one_hot(r['labels'][h].long(),10).float(),torch.logit(r['scores'][h].clamp(1e-5,1-1e-5))[:,None]/5,torch.log1p((r['grid_boxes'][h,1]-r['grid_boxes'][h,0])/max(1,length))[:,None]),1)
                    z=model.logits(seq,valid,meta.cuda());p=torch.zeros(len(h));q=torch.zeros(len(h));p[h]=z[:,:,0].sigmoid().mean(0).cpu();q[h]=z[:,:,1].sigmoid().mean(0).cpu()
                    eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=(valid[:,:,4:20].sum(-1).min(0).values>=4).cpu()
                    assert torch.equal(p,prior['pbg']) and torch.equal(eligible,prior['eligible']) and r['video']==prior['video']
                    result[role].append(dict(video=r['video'],pbg=p,quality=q,eligible=eligible,risk=protected_risk(p,q)))
                    count+=1
                    if count==3 and not (entry/'throughput.json').exists():
                        estimate=(pass_began-began)+(time.monotonic()-pass_began)*total/3*2
                        save(entry/'throughput.json',dict(first_videos=3,total_videos=total,two_pass_endtoend_estimate_seconds=estimate,limit=300))
                        assert estimate<300,'read-only budget exceeded; no automatic restart'
        return result
    first=output();model.load_state_dict(torch.load(checkpoint,map_location='cpu')['model']);second=output()
    for role in first:
        for a,b in zip(first[role],second[role]):
            for k in ('pbg','quality','eligible','risk'):assert torch.equal(a[k],b[k])
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    assert sha(checkpoint)==gpu['checkpoint_sha256']
    base.checkpoint(entry/'readout.pt',first);os.link(old/'predictions.pt',entry/'predictions.pt')
    paired=entry.parent/(entry.name+'guard');paired.mkdir();outputs={}
    for role in first:
        outputs[role]=[dict(prior,pbg=r['risk']) for r,prior in zip(first[role],previous[role])]
    base.checkpoint(paired/'predictions.pt',outputs)
    report=dict(passed=True,source_checkpoint_sha256=gpu['checkpoint_sha256'],origin_GPU_sha256=sha(old/'gpu_complete.json'),origin_final_audit_sha256=sha(ORIGIN/'independent_final_audit.json'),
        input_sha256=sha(native.DENSE/'inputs'/s/'data.pt'),guard=guard,guard_sha256=sha(guard),readout_sha256=sha(entry/'readout.pt'),original_predictions_sha256=sha(entry/'predictions.pt'),
        guarded_predictions_sha256=sha(paired/'predictions.pt'),all_original_BG_and_eligibility_bitwise_replayed=True,all_BG_quality_readouts_disk_reconstruction_bitwise_exact=True,
        final_digest=digest,parameters_CPUCUDA_RNG_unchanged=True,optimizer_steps=0,seconds=time.monotonic()-began)
    save(entry/'gpu_complete.json',report);os.link(entry/'gpu_complete.json',paired/'gpu_complete.json')


def evaluate(root,s):
    native.MODES=MODES;native.TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
    native.SPEC=SPEC;native.evaluate(root,s)


def run(root):
    from tools.baseline_gpu import snapshot
    assert read(ORIGIN/'independent_final_audit.json')['passed'] and not (ORIGIN/'stop.json').exists()
    assert not Path('/proc',str(read(ORIGIN/'state.json')['owner_pid'])).exists()
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),origin_final_audit_sha256=sha(ORIGIN/'independent_final_audit.json')))
    active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage,tasks in [('replay',JOBS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);lease=entry/'lease_replay.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,GPU_replays=8,optimizer_steps=0,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in JOBS},evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex),automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['replay','evaluate']);a=p.parse_args()
    if a.worker=='replay':replay(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
