"""冻结Prototype源缓存上的六方向、双头配对短预检；不读取任何目标预测/GT。"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from copy import deepcopy

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from tools.baseline_protocol import read, save, sha, code_identity

SOURCE = Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
PILOT = Path('/vepfs/vlms/proposal_foreground_source_pilot6_eb30065_1103')
SHIFTS = tuple('S%d' % i for i in range(1, 7))
MODES = ('control', 'ap_cost')
RATES = (.005, .01, .02, .05)
SPEC = dict(schema='prototype-background-source-AP-cost-v1', modes=MODES,
    source_only=True, target_read=False, detector_frozen=True, cutoff=.3,
    epochs=20, head_seed=311104, order_seed=123, lr=3e-4, wd=.05, clip=1.,
    same_architecture='HardBackgroundHead46274', same_quality_aux=.25,
    control='balanced_BG_BCE', ap_cost='.5*balanced_BG_BCE+.5*group_normalized_AP_cost_BCE',
    cost='per-video per-class mean_AP_.1:.5 delete-one difference; all low scores retained; not additive/global derivative',
    evaluation='all four predeclared equal-delete-count per-class budgets; no threshold/epoch selection',
    rates=RATES, selection='fixed20; no source_val used for fitting',
    budget='source cached readout only; no new detector forward or target inference')


def frame(item, video, grid=True):
    import pandas as pd
    boxes = item['boxes' if grid else 'segments'].numpy()
    return pd.DataFrame({'video-id':[video]*len(boxes), 't-start':boxes[:,0],
        't-end':boxes[:,1], 'label':item['labels'].numpy(), 'score':item['scores'].numpy(),
        'pbg':[0.]*len(boxes)})


def prepare(entry):
    import torch
    import pandas as pd
    import numpy as np
    from tools.background_ap_cost import deletion_costs
    from tools.hard_background_modules import targets
    from libs.evaluation.evaluator_anet import compute_average_precision_detection
    parent = SOURCE/entry.name
    index, truth, selection = (read(parent/n) for n in ('cache_index.json','source_truth.json','selection.json'))
    val = [r for r in read(PILOT/entry.name/'cache_index.json')['records'] if r['role']=='validation']
    assert set(truth) == set(selection['source_fit']) == {r['video'] for r in index['records']}
    assert not set(truth) & {r['video'] for r in val}
    rows = []; start=time.time(); checked=0
    for record in index['records']:
        assert sha(record['path']) == record['sha256']
        cpu = torch.load(record['path'], map_location='cpu'); video=record['video']
        pf = frame(cpu,video); gt=truth[video]
        g = pd.DataFrame({'video-id':[video]*len(gt['labels']),
            't-start':[v[0] for v in gt['segments']], 't-end':[v[1] for v in gt['segments']], 'label':gt['labels']})
        delta = deletion_costs(g,pf)
        bg,q = targets(cpu['boxes'],cpu['labels'],torch.tensor(gt['segments']).reshape(-1,2),torch.tensor(gt['labels']))
        hard = cpu['scores']>=.3
        # 首个非空视频的关键删框，独立原官方AP核验；合成测试另外覆盖同分与替补。
        if checked==0 and hard.any():
            for i in np.flatnonzero(hard.numpy())[:3]:
                c=int(cpu['labels'][i]); f=pf[pf.label==c]; t=g[g.label==c]
                before=compute_average_precision_detection(t,f,np.linspace(.1,.5,5))[0].mean() if len(t) else 0.
                after=compute_average_precision_detection(t,f.drop(i),np.linspace(.1,.5,5))[0].mean() if len(t) else 0.
                assert abs(delta[i]-(after-before))<1e-7
                checked+=1
        rows.append(dict(record=record,hard_indices=np.flatnonzero(hard.numpy()).tolist(),
            bg=bg[hard].tolist(),quality=q[hard].tolist(),delta=delta[hard.numpy()].tolist()))
    assert checked and len(rows)==len(truth)
    save(entry/'costs.json',dict(spec=SPEC,rows=rows,source_truth_sha256=sha(parent/'source_truth.json'),
        source_index_sha256=sha(parent/'cache_index.json'),validation=val,official_checks=checked,
        detector_digest=index['detector_digest'],seconds=time.time()-start,
        harmful_count=sum(sum(v<0 for v in r['delta']) for r in rows),
        beneficial_BG_count=sum(sum(b==1 and d>0 for b,d in zip(r['bg'],r['delta'])) for r in rows)))
    print('PREPARED',entry.name,len(rows),time.time()-start,flush=True)


def fit_items(entry):
    import torch
    rows=[]
    for r in read(entry/'costs.json')['rows']:
        if not r['hard_indices']: continue
        x=torch.load(r['record']['path'],map_location='cpu');ids=torch.tensor(r['hard_indices'])
        rows.append(dict(feature=x['feature'],valid=x['valid'],boxes=x['boxes'][ids],
            labels=x['labels'][ids],scores=x['scores'][ids],bg=torch.tensor(r['bg']),
            quality=torch.tensor(r['quality']),delta=torch.tensor(r['delta'],dtype=torch.float32)))
    return rows


def gpu(entry, kernel=False):
    import torch
    import random
    import numpy as np
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.hard_background_modules import HardBackgroundHead,step_loss
    from tools.background_ap_cost import step
    guard=first_cuda_guard(entry)
    torch.manual_seed(SPEC['head_seed']);random.seed(123);np.random.seed(123)
    torch.cuda.manual_seed(SPEC['head_seed'])
    initial=HardBackgroundHead().cuda();initial_digest=base.tensor_digest(initial.state_dict())
    heads={m:deepcopy(initial) for m in MODES};del initial
    opts={m:torch.optim.AdamW(h.parameters(),lr=SPEC['lr'],weight_decay=SPEC['wd']) for m,h in heads.items()}
    items=fit_items(entry);assert items
    if kernel:
        chosen=next((x for x in items if (x['delta']<0).any() and (x['bg']==1).any()),items[0])
        x={k:v.cuda() for k,v in chosen.items()}
        a=step(heads['control'],x,'control');b,_=step_loss(heads['control'],x,'conditional')
        assert torch.equal(a,b)
        grads_a=torch.autograd.grad(a,tuple(heads['control'].parameters()),retain_graph=True)
        grads_b=torch.autograd.grad(b,tuple(heads['control'].parameters()))
        assert all(torch.equal(u,v) for u,v in zip(grads_a,grads_b))
        reports={}
        for m in MODES:
            h,o=heads[m],opts[m];o.zero_grad();loss=step(h,x,m);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(h.parameters(),1.);assert norm>0 and torch.isfinite(norm);o.step()
            state=dict(head=h.state_dict(),optimizer=o.state_dict(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0))
            path=entry/(m+'_kernel.pt');base.checkpoint(path,state);saved=torch.load(path,map_location='cpu')
            fresh=HardBackgroundHead().cuda();fresh.load_state_dict(saved['head']);fo=torch.optim.AdamW(fresh.parameters());fo.load_state_dict(saved['optimizer'])
            assert base.tensor_digest(fresh.state_dict())==base.tensor_digest(h.state_dict())!=initial_digest
            assert base.tensor_digest(fo.state_dict())==base.tensor_digest(o.state_dict())
            torch.set_rng_state(saved['cpu']);torch.cuda.set_rng_state(saved['cuda'],0)
            assert torch.equal(torch.get_rng_state(),saved['cpu']) and torch.equal(torch.cuda.get_rng_state(0),saved['cuda'])
            # 重建后同一步更新逐tensor恢复核验。
            for model,opt in [(h,o),(fresh,fo)]:
                opt.zero_grad();step(model,x,m).backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            assert base.tensor_digest(h.state_dict())==base.tensor_digest(fresh.state_dict())
            reports[m]=dict(gradient=float(norm),state_restore_and_next_update_exact=True,checkpoint_sha256=sha(path))
        save(entry/'kernel_complete.json',dict(passed=True,reports=reports,control_old_loss_and_grad_exact=True,
            guard=guard,guard_sha256=sha(guard),peak_mib=torch.cuda.max_memory_reserved()/2**20,initial_digest=initial_digest))
        return
    order=torch.Generator().manual_seed(SPEC['order_seed']);history=[];first={}
    for epoch in range(1,21):
        start=time.time();totals={m:0. for m in MODES}
        for idx in torch.randperm(len(items),generator=order).tolist():
            x={k:v.cuda() for k,v in items[idx].items()}
            for m in MODES:
                opts[m].zero_grad(set_to_none=True);loss=step(heads[m],x,m)
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(heads[m].parameters(),1.)
                assert torch.isfinite(norm);opts[m].step();totals[m]+=float(loss.detach())
                if m not in first:first[m]=float(norm)
        history.append(dict(epoch=epoch,seconds=time.time()-start,loss={m:v/len(items) for m,v in totals.items()}))
        save(entry/'history.json',history)
        base.checkpoint(entry/'latest_heads.pt',dict(heads={m:h.state_dict() for m,h in heads.items()},
            optimizers={m:o.state_dict() for m,o in opts.items()},order_rng=order.get_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(0),history=history,spec=SPEC,costs_sha256=sha(entry/'costs.json')))
        print('EPOCH',entry.name,epoch,history[-1],flush=True)
    os.link(entry/'latest_heads.pt',entry/'fixed20_heads.pt')
    predictions={m:[] for m in MODES};costs=read(entry/'costs.json')
    for r in costs['validation']:
        assert sha(r['path'])==r['sha256'];cpu=torch.load(r['path'],map_location='cpu');hard=cpu['scores']>=.3
        for m,h in heads.items():
            h.eval();pbg=torch.zeros(len(hard))
            if hard.any():
                with torch.no_grad():pbg[hard]=h(cpu['feature'].cuda(),cpu['valid'].cuda(),cpu['boxes'][hard].cuda(),cpu['labels'][hard].cuda(),cpu['scores'][hard].cuda())[:,0].sigmoid().cpu()
            predictions[m].append(dict(video=r['video'],segments=cpu['segments'],scores=cpu['scores'],labels=cpu['labels'],pbg=pbg))
    base.checkpoint(entry/'source_val_predictions.pt',predictions)
    save(entry/'gpu_complete.json',dict(code=code_identity(),spec=SPEC,guard=guard,guard_sha256=sha(guard),
        initial_digest=initial_digest,final_digests={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()},
        first_gradient_norms=first,steps_per_mode=20*len(items),peak_mib=torch.cuda.max_memory_reserved()/2**20,
        artifacts={str(p):sha(p) for p in [entry/'costs.json',entry/'fixed20_heads.pt',entry/'history.json',entry/'source_val_predictions.pt']}))


def evaluate(entry):
    import torch
    import numpy as np
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.run_foreground_calibration_oracle import load_truth
    from libs.evaluation.evaluator_anet import compute_average_precision_detection
    done=read(entry/'gpu_complete.json')
    for p,h in done['artifacts'].items():assert sha(p)==h
    cfg=read(SOURCE/entry.name/'config.json');gt=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    predictions=torch.load(entry/'source_val_predictions.pt',map_location='cpu');frames={m:prediction_frame(v) for m,v in predictions.items()}
    assert frames['control'].drop(columns='pbg').equals(frames['ap_cost'].drop(columns='pbg'))
    results={}
    for m,f in frames.items():
        bg=background_mask(gt,f);rates=[];original=[]
        for rate in RATES:
            ap=[];ca=[];removed=nb=cb=cn=0
            for c in range(10):
                pos=np.flatnonzero(f.label.to_numpy()==c);sub=f.iloc[pos].reset_index(drop=True);t=gt[gt.label==c]
                engine=ClassAP(t,sub);base=engine.evaluate(np.ones(len(sub),dtype=bool))
                pool=np.flatnonzero(sub.score.to_numpy()>=.3);n=int(np.floor(rate*len(pool)))
                selected=sorted(pool,key=lambda i:(-float(sub.pbg.iloc[i]),int(i)))[:n]
                confidence=sorted(pool,key=lambda i:(float(sub.score.iloc[i]),int(i)))[:n]
                keep=np.ones(len(sub),dtype=bool);keep[selected]=False;ck=np.ones(len(sub),dtype=bool);ck[confidence]=False
                actual=engine.evaluate(keep);control=engine.evaluate(ck)
                # 完整官方AP复算每操作点，不用只看标签精度。
                official=compute_average_precision_detection(t,sub[keep],np.linspace(.1,.5,5))[0] if len(t) else np.zeros(5)
                assert np.max(np.abs(official-np.asarray(actual['ap'])))<1e-7
                ap.append(actual['map']);ca.append(control['map'])
                if rate==RATES[0]:original.append(base['map'])
                removed+=int(bg[pos[selected]].sum());nb+=n-int(bg[pos[selected]].sum())
                cb+=int(bg[pos[confidence]].sum());cn+=n-int(bg[pos[confidence]].sum())
            rates.append(dict(rate=rate,map=float(np.mean(ap)),confidence_map=float(np.mean(ca)),BG_removed=removed,nonBG_removed=nb,confidence_BG_removed=cb,confidence_nonBG_removed=cn))
        baseline=float(np.mean(original))
        for x in rates:x.update(delta_pp=100*(x['map']-baseline),confidence_delta_pp=100*(x['confidence_map']-baseline))
        results[m]=dict(baseline=baseline,all_fixed_budgets=rates)
    save(entry/'complete.json',dict(passed=True,shift=entry.name,spec=SPEC,results=results,source_only=True,
        gpu_complete_sha256=sha(entry/'gpu_complete.json'),artifacts=done['artifacts']))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False)
    save(root/'plan.json',dict(spec=SPEC,code=code_identity(),shifts=SHIFTS,prior_measured_four_head_seconds={'S1':119.52,'S5':804.63}))
    active={}
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for task in active.values():
            if task['p'].poll() is None:os.killpg(task['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage in ('prepare','kernel','train','evaluate'):
            pending=list(SHIFTS)
            while pending or active:
                for s,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[s]
                    if t['p'].returncode:raise RuntimeError(s+'/'+stage+' failed')
                devices=[None]*max(0,2-len(active)) if stage in ('prepare','evaluate') else [g for g in [2,0,3,1] if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    s=pending.pop(0);e=root/s;e.mkdir(exist_ok=True)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
                    if g is not None:
                        obs=snapshot(g);lease=e/(stage+'_lease.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=obs['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                        env.update(CUDA_VISIBLE_DEVICES=obs['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(e/(stage+'.log')).open('a');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(e)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[s]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={s:dict(pid=t['p'].pid,gpu=t['gpu']) for s,t in active.items()},pending=pending))
                time.sleep(3)
            if stage=='kernel':assert all(read(root/s/'kernel_complete.json')['passed'] for s in SHIFTS)
        results={s:read(root/s/'complete.json') for s in SHIFTS}
        save(root/'final.json',dict(spec=SPEC,results=results,complete_sha256={s:sha(root/s/'complete.json') for s in SHIFTS}))
        save(root/'state.json',dict(stage='complete',active={},owner_pid=os.getpid()))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['prepare','kernel','train','evaluate']);a=p.parse_args()
    if a.worker=='prepare':prepare(a.entry)
    elif a.worker=='kernel':gpu(a.entry,True)
    elif a.worker=='train':gpu(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry)
    else:run(a.root)
