"""三种低容量源背景读出；同留出划分、同20轮，不读目标或选择阈值。"""
import argparse
from copy import deepcopy
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
from tools.background_holdout_curve import split_videos
from tools.background_linear_temporal import MODES,LinearBackground,descriptors,objective

ORIGIN=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
CURVE=Path('/vepfs/vlms/background_holdout_curve6_837d6fb_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
EPOCHS=(20,)
RATES=(.005,.01,.02,.05)
SPEC=dict(schema='background-linear-temporal-source-v1',modes=MODES,head_only=True,target_read=False,
    epochs=20,checkpoints=(1,5,10,20),prediction_epochs=EPOCHS,rates=RATES,head_seed=311104,order_seed=123,lr=3e-4,wd=.05,clip=1.,
    objective='unchanged control balanced_BG_BCE + .25 quality loss',init='all linear weights and biases zero',
    max_linear='raw frozen FPN original region max and inner-outer differences; 3104 parameters',
    temporal_abs='raw FPN regional means plus abs(late8mean-early8mean); 4130 parameters',
    temporal_signed='same as temporal_abs except signed late-minus-early; same4130 parameters',
    missing_half='zero temporal difference when either half has no valid samples; same pair-valid feature',
    feature_normalization='none; raw frozen FPN units',split='same head-holdout-v1 video split',
    efficiency='parameter-free descriptors computed once per video; real GPU cached/direct loss-gradient-update equality required',
    caveat='max vs old MLP changes projection and readout package; abs vs signed isolates direction readout, not proof of temporal semantics; original detector saw head-heldout videos',
    selection='all saved epochs/rates; sourceval only epoch20; no winner selection')


def update(head,opt,x,bg,quality):
    import torch
    opt.zero_grad(set_to_none=True);loss=objective(head(x),bg,quality);assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(norm)
    opt.step();return float(loss.detach()),float(norm)


def kernel(entry,x,bg,quality,guard):
    import torch
    from tools import baseline_engine as base
    began=time.time();reports={};repeats=3
    torch.cuda.synchronize();t=time.perf_counter()
    for _ in range(repeats):direct=descriptors(**x)
    torch.cuda.synchronize();descriptor_seconds=(time.perf_counter()-t)/repeats
    cache={m:v.cpu() for m,v in direct.items()}
    base.checkpoint(entry/'kernel_descriptors.pt',cache);loaded=torch.load(entry/'kernel_descriptors.pt',map_location='cpu')
    for m in MODES:
        a=LinearBackground(m).cuda();b=LinearBackground(m).cuda();oa=torch.optim.AdamW(a.parameters(),lr=3e-4,weight_decay=.05);ob=torch.optim.AdamW(b.parameters(),lr=3e-4,weight_decay=.05)
        original=base.tensor_digest(a.state_dict());cached=loaded[m].cuda()
        assert torch.equal(cached,direct[m])
        la=objective(a(descriptors(**x)[m]),bg,quality);lb=objective(b(cached),bg,quality)
        assert torch.equal(la,lb);la.backward();lb.backward()
        assert all(torch.equal(p.grad,q.grad) for p,q in zip(a.parameters(),b.parameters()))
        for h,o in ((a,oa),(b,ob)):torch.nn.utils.clip_grad_norm_(h.parameters(),1.);o.step()
        assert base.tensor_digest(a.state_dict())==base.tensor_digest(b.state_dict())!=original
        order=torch.Generator().manual_seed(123)
        path=entry/(m+'_kernel.pt');base.checkpoint(path,dict(head=a.state_dict(),optimizer=oa.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),python_rng=random.getstate(),numpy_rng=np.random.get_state(),order_rng=order.get_state()))
        state=torch.load(path,map_location='cpu');fresh=LinearBackground(m).cuda();fresh.load_state_dict(state['head']);fo=torch.optim.AdamW(fresh.parameters());fo.load_state_dict(state['optimizer'])
        def restore():
            torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'],0);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);order.set_state(state['order_rng'])
        restore();draw=(torch.rand(3),torch.rand(3,device='cuda'),random.random(),np.random.rand(),torch.randperm(11,generator=order))
        restore();replay=(torch.rand(3),torch.rand(3,device='cuda'),random.random(),np.random.rand(),torch.randperm(11,generator=order))
        assert all(torch.equal(draw[i],replay[i]) for i in (0,1,4)) and draw[2:4]==replay[2:4]
        restore();_,norm=update(a,oa,cached,bg,quality);update(fresh,fo,cached,bg,quality)
        assert norm>0 and base.tensor_digest(a.state_dict())==base.tensor_digest(fresh.state_dict()) and base.tensor_digest(oa.state_dict())==base.tensor_digest(fo.state_dict())
        # 相同head/opt完整状态重复一轮，计入额外验收成本，不转移正式头。
        durations=[];timed_digests=[]
        for use_cache in (False,True):
            model=LinearBackground(m).cuda();model.load_state_dict(state['head']);opt=torch.optim.AdamW(model.parameters());opt.load_state_dict(deepcopy(state['optimizer']))
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(repeats):update(model,opt,loaded[m].cuda() if use_cache else descriptors(**x)[m],bg,quality)
            torch.cuda.synchronize();durations.append((time.perf_counter()-begin)/repeats)
            timed_digests.append((base.tensor_digest(model.state_dict()),base.tensor_digest(opt.state_dict())))
        assert timed_digests[0]==timed_digests[1]
        reports[m]=dict(parameters=sum(p.numel() for p in a.parameters()),gradient=norm,cached_direct_loss_grad_update_exact=True,full_state_disk_restore_next_update_exact=True,RNG_draws_exact=True,checkpoint_sha256=sha(path),direct_step_seconds=durations[0],cached_step_seconds=durations[1])
    assert sum(v['cached_step_seconds'] for v in reports.values())<sum(v['direct_step_seconds'] for v in reports.values())
    save(entry/'kernel_complete.json',dict(passed=True,reports=reports,descriptor_seconds=descriptor_seconds,seconds=time.time()-began,guard=guard,guard_sha256=sha(guard),descriptor_sha256=sha(entry/'kernel_descriptors.pt'),cached_benchmark_includes_CPU_to_GPU=True,timed_updates_exact=True))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    began=time.time();cost=read(ORIGIN/entry.name/'costs.json');old=read(ORIGIN/entry.name/'gpu_complete.json')
    assert sha(ORIGIN/entry.name/'costs.json')==old['artifacts'][str(ORIGIN/entry.name/'costs.json')]
    audit=read(CURVE/'independent_final_audit.json');assert audit['passed']
    split=split_videos([r['record']['video'] for r in cost['rows']]);oldsplit=read(CURVE/entry.name/'split.json')
    assert all(split[k]==oldsplit[k] for k in split)
    assert not set(split['fit']+split['heldout']) & {r['video'] for r in cost['validation']}
    save(entry/'split.json',dict(**split,origin_costs_sha256=sha(ORIGIN/entry.name/'costs.json'),old_split_sha256=sha(CURVE/entry.name/'split.json'),spec=SPEC))
    guard=first_cuda_guard(entry)
    def seed():
        torch.manual_seed(311104);torch.cuda.manual_seed_all(311104);random.seed(123);np.random.seed(123)
    seed();probe=next(r for r in cost['rows'] if r['record']['video'] in split['fit'] and r['hard_indices'])
    raw=torch.load(probe['record']['path'],map_location='cpu');ids=torch.tensor(probe['hard_indices'],dtype=torch.long)
    x={k:(raw[k][ids] if k in ('boxes','labels','scores') else raw[k]).cuda() for k in ('feature','valid','boxes','labels','scores')}
    kernel(entry,x,torch.tensor(probe['bg'],device='cuda'),torch.tensor(probe['quality'],device='cuda'),guard)
    del raw,x;seed()
    heads={m:LinearBackground(m).cuda() for m in MODES};initial={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
    opts={m:torch.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=.05) for m,h in heads.items()}
    prepared={role:[] for role in ('fit','heldout','validation')};prep_begin=time.time()
    for role,records,grid in [('source',[r['record'] for r in cost['rows']],True),('validation',cost['validation'],False)]:
        label_rows={r['record']['video']:r for r in cost['rows']}
        for rec in records:
            cpu=torch.load(rec['path'],map_location='cpu');video=rec['video'];hard=cpu['scores']>=.3
            with torch.no_grad():d=descriptors(cpu['feature'].cuda(),cpu['valid'].cuda(),cpu['boxes'][hard].cuda(),cpu['labels'][hard].cuda(),cpu['scores'][hard].cuda())
            item=dict(video=video,segments=cpu['boxes' if grid else 'segments'],scores=cpu['scores'],labels=cpu['labels'],hard=hard,descriptors={m:t.cpu() for m,t in d.items()})
            dest=role if role=='validation' else ('fit' if video in split['fit'] else 'heldout')
            if role=='source':
                r=label_rows[video];assert torch.equal(torch.where(hard)[0],torch.tensor(r['hard_indices'],dtype=torch.long))
                item.update(bg=torch.tensor(r['bg']),quality=torch.tensor(r['quality']))
            prepared[dest].append(item)
    preparation_seconds=time.time()-prep_begin
    base.checkpoint(entry/'descriptors.pt',prepared)
    fit=[x for x in prepared['fit'] if x['hard'].any()];generator=torch.Generator().manual_seed(123);history=[];saved={};first={}
    def predict(items):
        before={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()};cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);outputs={m:[] for m in MODES}
        with torch.no_grad():
            for item in items:
                for m,h in heads.items():
                    h.eval();p=torch.zeros(len(item['scores']));p[item['hard']]=h(item['descriptors'][m].cuda())[:,0].sigmoid().cpu()
                    outputs[m].append(dict(video=item['video'],segments=item['segments'],scores=item['scores'],labels=item['labels'],pbg=p))
        assert before=={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()} and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
        for h in heads.values():h.train()
        return outputs
    for epoch in range(1,21):
        totals={m:0. for m in MODES};begin=time.time()
        for i in torch.randperm(len(fit),generator=generator).tolist():
            item=fit[i];bg=item['bg'].cuda();quality=item['quality'].cuda()
            for m in MODES:
                loss,norm=update(heads[m],opts[m],item['descriptors'][m].cuda(),bg,quality);totals[m]+=loss
                if m not in first:first[m]=norm
        history.append(dict(epoch=epoch,loss={m:v/len(fit) for m,v in totals.items()},seconds=time.time()-begin))
        base.checkpoint(entry/'latest.pt',dict(heads={m:h.state_dict() for m,h in heads.items()},optimizers={m:o.state_dict() for m,o in opts.items()},history=history,order_rng=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),split_sha256=sha(entry/'split.json'),spec=SPEC))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        if epoch in EPOCHS:
            outputs={role:predict(prepared[role]) for role in ('fit','heldout')}
            if epoch==20:outputs['validation']=predict(prepared['validation'])
            base.checkpoint(entry/f'predictions{epoch:03d}.pt',outputs)
            saved[str(epoch)]=dict(checkpoint_sha256=sha(entry/f'epoch{epoch:03d}.pt'),predictions_sha256=sha(entry/f'predictions{epoch:03d}.pt'))
        save(entry/'history.json',history);print('EPOCH',entry.name,epoch,flush=True)
    save(entry/'gpu_complete.json',dict(passed=True,spec=SPEC,split_sha256=sha(entry/'split.json'),origin_costs_sha256=sha(ORIGIN/entry.name/'costs.json'),curve_audit_sha256=sha(CURVE/'independent_final_audit.json'),kernel_sha256=sha(entry/'kernel_complete.json'),probe_video=probe['record']['video'],initial_digests=initial,final_digests={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()},parameters={m:sum(p.numel() for p in h.parameters()) for m,h in heads.items()},first_gradients=first,fit_steps_per_epoch=len(fit),saved=saved,descriptors_sha256=sha(entry/'descriptors.pt'),history_sha256=sha(entry/'history.json'),guard=guard,guard_sha256=sha(guard),readout_parameter_and_RNG_unchanged=True,preparation_seconds=preparation_seconds,seconds=time.time()-began,peak_mib=torch.cuda.max_memory_reserved()/2**20))

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
            frames={m:prediction_frame(rows) for m,rows in modes.items()};f=frames[MODES[0]];gt=truths[role];bg=background_mask(gt,f)
            assert all(f.drop(columns='pbg').equals(frames[m].drop(columns='pbg')) for m in MODES)
            engines={};positions={};baselines=[]
            for c in range(10):
                ids=np.flatnonzero(f.label.to_numpy()==c);positions[c]=ids;engines[c]=ClassAP(gt[gt.label==c],f.iloc[ids]);base=engines[c].evaluate(np.ones(len(ids),dtype=bool));baselines.append(base['map'])
                if c==0:
                    original=official_ap(gt[gt.label==c],f.iloc[ids]);assert np.max(np.abs(original-np.asarray(base['ap'])))<1e-7;official_checks.append(dict(epoch=epoch,role=role,label=c))
            records=[]
            for v,indices in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(indices);ids=ids[f.iloc[ids].score.to_numpy()>=.3]
                records.append(dict(video=v,labels=f.iloc[ids].label.tolist(),scores=f.iloc[ids].score.tolist(),bg=bg[ids].tolist(),pbg={m:frames[m].iloc[ids].pbg.tolist() for m in MODES}))
            metrics={}
            for m in (*MODES,'confidence'):
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
