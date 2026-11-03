"""固定源视频留出学习曲线；四个预声明时点，不选最好epoch或阈值。"""
import argparse
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.background_fit_gap import summarize

ORIGIN=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
EPOCHS=(1,5,10,20)
RATES=(.005,.01,.02,.05)
MODES=('control','ap_cost')
SPEC=dict(schema='background-head-video-holdout-curve-v1',head_only=True,target_read=False,
    holdout_fraction=.2,split='ceil20percent lowest SHA256(head-holdout-v1|video)',epochs=20,
    checkpoints=EPOCHS,rates=RATES,head_seed=311104,order_seed=123,lr=3e-4,wd=.05,clip=1.,
    unchanged='HardBackgroundHead46274 and original two objectives',selection='none; all epochs and rates reported',
    caveat='detector saw all source train; only new head optimizer excludes heldout videos')


def split_videos(videos):
    ids=sorted(set(videos));assert len(ids)==len(videos) and len(ids)>=2
    ranked=sorted(ids,key=lambda v:(hashlib.sha256(('head-holdout-v1|'+v).encode()).hexdigest(),v))
    heldout=set(ranked[:min(len(ids)-1,max(1,int(np.ceil(.2*len(ids)))))] )
    return dict(fit=[v for v in ids if v not in heldout],heldout=sorted(heldout))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_background_ap_cost import gpu as old_kernel
    from tools.hard_background_modules import HardBackgroundHead
    from tools.background_ap_cost import step
    parent=ORIGIN/entry.name;start=time.time();cost=read(parent/'costs.json')
    gpu=read(parent/'gpu_complete.json');assert sha(parent/'costs.json')==gpu['artifacts'][str(parent/'costs.json')]
    split=split_videos([r['record']['video'] for r in cost['rows']]);save(entry/'split.json',dict(**split,origin_costs_sha256=sha(parent/'costs.json'),spec=SPEC))
    assert not set(split['fit']+split['heldout']) & {r['video'] for r in cost['validation']}
    probe=entry/'probe';probe.mkdir();probe_cost=deepcopy(cost);probe_cost['rows']=[r for r in cost['rows'] if r['record']['video'] in split['fit']];save(probe/'costs.json',probe_cost)
    old_kernel(probe,True)  # 新运行真实GPU禁用等价、梯度、磁盘恢复验收；probe头不转移。
    guard=first_cuda_guard(entry)
    torch.manual_seed(311104);torch.cuda.manual_seed_all(311104);random.seed(123);np.random.seed(123)
    initial=HardBackgroundHead().cuda();initial_digest=base.tensor_digest(initial.state_dict())
    assert initial_digest==gpu['initial_digest']
    heads={m:deepcopy(initial) for m in MODES};del initial
    opts={m:torch.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=.05) for m,h in heads.items()}
    cached=[]
    for r in cost['rows']:
        x=torch.load(r['record']['path'],map_location='cpu');ids=torch.tensor(r['hard_indices'],dtype=torch.long)
        item=dict(feature=x['feature'],valid=x['valid'],boxes=x['boxes'][ids],labels=x['labels'][ids],scores=x['scores'][ids],bg=torch.tensor(r['bg']),quality=torch.tensor(r['quality']),delta=torch.tensor(r['delta'],dtype=torch.float32))
        cached.append((r,x,ids,item))
    fit=[t for t in cached if t[0]['record']['video'] in split['fit'] and len(t[2])]
    generator=torch.Generator().manual_seed(123);history=[];saved={}
    def predict(items,grid):
        outputs={m:[] for m in MODES};before={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state(0)
        with torch.no_grad():
            for video,x in items:
                ids=x['scores']>=.3
                for m,h in heads.items():
                    h.eval();p=torch.zeros(len(ids))
                    if ids.any():p[ids]=h(x['feature'].cuda(),x['valid'].cuda(),x['boxes'][ids].cuda(),x['labels'][ids].cuda(),x['scores'][ids].cuda())[:,0].sigmoid().cpu()
                    outputs[m].append(dict(video=video,segments=x['boxes' if grid else 'segments'],scores=x['scores'],labels=x['labels'],pbg=p))
        assert before=={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
        assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state(0))
        for h in heads.values():h.train()
        return outputs
    for epoch in range(1,21):
        totals={m:0. for m in MODES};begin=time.time()
        for j in torch.randperm(len(fit),generator=generator).tolist():
            x={k:v.cuda() for k,v in fit[j][3].items()}
            for m,h in heads.items():
                opts[m].zero_grad(set_to_none=True);loss=step(h,x,m);assert torch.isfinite(loss);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.);assert torch.isfinite(norm);opts[m].step();totals[m]+=float(loss.detach())
        history.append(dict(epoch=epoch,loss={m:v/len(fit) for m,v in totals.items()},seconds=time.time()-begin))
        base.checkpoint(entry/'latest.pt',dict(heads={m:h.state_dict() for m,h in heads.items()},optimizers={m:o.state_dict() for m,o in opts.items()},history=history,order_rng=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),split_sha256=sha(entry/'split.json'),spec=SPEC))
        if epoch in EPOCHS:
            os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
            outputs={role:predict([(r['record']['video'],x) for r,x,ids,item in cached if r['record']['video'] in split[role]],True) for role in ('fit','heldout')}
            if epoch==20:
                outputs['validation']=predict([(r['video'],torch.load(r['path'],map_location='cpu')) for r in cost['validation']],False)
            base.checkpoint(entry/f'predictions{epoch:03d}.pt',outputs)
            saved[str(epoch)]=dict(checkpoint_sha256=sha(entry/f'epoch{epoch:03d}.pt'),predictions_sha256=sha(entry/f'predictions{epoch:03d}.pt'))
        save(entry/'history.json',history)
        print('EPOCH',entry.name,epoch,flush=True)
    save(entry/'gpu_complete.json',dict(passed=True,spec=SPEC,split_sha256=sha(entry/'split.json'),origin_costs_sha256=sha(parent/'costs.json'),initial_digest=initial_digest,fit_videos=len(split['fit']),heldout_videos=len(split['heldout']),fit_steps_per_epoch=len(fit),saved=saved,history_sha256=sha(entry/'history.json'),guard=guard,guard_sha256=sha(guard),probe_sha256=sha(probe/'kernel_complete.json'),readout_parameter_and_RNG_unchanged=True,seconds=time.time()-start,peak_mib=torch.cuda.max_memory_reserved()/2**20))


def evaluate(entry):
    import torch
    import pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.background_ap_cost import official_ap
    done=read(entry/'gpu_complete.json');split=read(entry/'split.json');source_truth=read(SOURCE/entry.name/'source_truth.json')
    cfg=read(SOURCE/entry.name/'config.json');val_gt=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    truths={role:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[role] for b,c in zip(source_truth[v]['segments'],source_truth[v]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')};truths['validation']=val_gt
    results={};official_checks=[]
    for epoch in EPOCHS:
        path=entry/f'predictions{epoch:03d}.pt';assert sha(path)==done['saved'][str(epoch)]['predictions_sha256'];outputs=torch.load(path,map_location='cpu');results[str(epoch)]={}
        for role,modes in outputs.items():
            frames={m:prediction_frame(rows) for m,rows in modes.items()};f=frames['control'];gt=truths[role];bg=background_mask(gt,f)
            assert f.drop(columns='pbg').equals(frames['ap_cost'].drop(columns='pbg'))
            engines={};positions={};baselines=[]
            for c in range(10):
                ids=np.flatnonzero(f.label.to_numpy()==c);positions[c]=ids;engines[c]=ClassAP(gt[gt.label==c],f.iloc[ids]);base=engines[c].evaluate(np.ones(len(ids),dtype=bool));baselines.append(base['map'])
                if epoch==1 and c==0 or epoch==20 and role=='validation' and c==0:
                    original=official_ap(gt[gt.label==c],f.iloc[ids]);assert np.max(np.abs(original-np.asarray(base['ap'])))<1e-7;official_checks.append(dict(epoch=epoch,role=role,label=c))
            records=[]
            for v,indices in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(indices);ids=ids[f.iloc[ids].score.to_numpy()>=.3]
                records.append(dict(video=v,labels=f.iloc[ids].label.tolist(),scores=f.iloc[ids].score.tolist(),bg=bg[ids].tolist(),pbg={m:frames[m].iloc[ids].pbg.tolist() for m in MODES}))
            metrics={}
            for m in ('control','ap_cost','confidence'):
                summary=summarize(records,m);budgets=[]
                for report in summary['all_fixed_budgets']:
                    rate=report['rate'];aps=[];selected_BG=selected_nonBG=0
                    for c in range(10):
                        ids=positions[c];pool=[int(i) for i in np.flatnonzero(f.iloc[ids].score.to_numpy()>=.3)]
                        pool.sort(key=lambda i:(float(f.score.iloc[ids[i]]) if m=='confidence' else -float(frames[m].pbg.iloc[ids[i]]),i));chosen=pool[:int(rate*len(pool))]
                        keep=np.ones(len(ids),dtype=bool);keep[chosen]=False;aps.append(engines[c].evaluate(keep)['map']);selected_BG+=int(bg[ids[chosen]].sum());selected_nonBG+=len(chosen)-int(bg[ids[chosen]].sum())
                    assert selected_BG==report['BG_removed'] and selected_nonBG==report['nonBG_removed']
                    budgets.append(dict(rate=rate,map=float(np.mean(aps)),delta_pp=float(100*(np.mean(aps)-np.mean(baselines))),BG_removed=selected_BG,nonBG_removed=selected_nonBG))
                metrics[m]=dict(video_macro_AUC=summary['video_macro_AUC'],AUC_valid_videos=summary['AUC_valid_videos'],total_videos=summary['total_videos'],baseline=float(np.mean(baselines)),all_fixed_budgets=budgets)
            results[str(epoch)][role]=metrics
            print('EVALUATED',entry.name,epoch,role,flush=True)
    save(entry/'complete.json',dict(passed=True,spec=SPEC,gpu_complete_sha256=sha(entry/'gpu_complete.json'),split_sha256=sha(entry/'split.json'),official_original_checks=official_checks,results=results))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),origin=str(ORIGIN)));active={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage in ('train','evaluate'):
            pending=[f'S{i}' for i in range(1,7)]
            while pending or active:
                for s,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[s]
                    if t['p'].returncode:raise RuntimeError(s+'/'+stage+' failed')
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in (2,0,3,1) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    s=pending.pop(0);e=root/s;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                    if g is not None:
                        obs=snapshot(g);lease=e/'train_lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=obs['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=obs['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(e/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(e),'--worker',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[s]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),pending=pending,active={s:dict(pid=t['p'].pid,gpu=t['gpu']) for s,t in active.items()}));time.sleep(3)
        results={f'S{i}':read(root/f'S{i}'/'complete.json') for i in range(1,7)};save(root/'final.json',dict(spec=SPEC,results=results,complete_sha256={s:sha(root/s/'complete.json') for s in results}))
        save(root/'state.json',dict(stage='complete',active={},owner_pid=os.getpid()))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['train','evaluate']);a=p.parse_args()
    if a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry)
    else:run(a.root)
