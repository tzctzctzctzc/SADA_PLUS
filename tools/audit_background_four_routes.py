"""四路线独立审计：全输入NumPy重建、保存状态原GPU逐位重放、独立选删/AP。"""
import argparse
import os
from pathlib import Path
import subprocess
import signal
import sys
import time
import numpy as np

CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_background_four_routes import OLD,ORIGIN,SOURCE,ROUTES,MODES,SPEC,model_factory


def numpy_samples(feature,valid,boxes):
    length=boxes[:,1]-boxes[:,0];values=[];masks=[];positions=[]
    for a,b,n in ((boxes[:,0]-.25*length,boxes[:,0],4),(boxes[:,0],boxes[:,1],16),(boxes[:,1],boxes[:,1]+.25*length,4)):
        t=(np.arange(n,dtype=np.float32)+np.float32(.5))/np.float32(n)
        p=a[:,None]+(b-a)[:,None]*t
        lo=np.floor(p).astype(np.int64);hi=np.ceil(p).astype(np.int64);w=p-lo.astype(np.float32)
        l=np.clip(lo,0,feature.shape[-1]-1);h=np.clip(hi,0,feature.shape[-1]-1)
        masks.append((p>=0)&(p<=feature.shape[-1]-1)&valid[l]&valid[h])
        values.append((feature[:,l]*(1-w)[None]+feature[:,h]*w[None]).transpose(1,2,0));positions.append(p)
    return np.concatenate(values,1),np.concatenate(masks,1),np.concatenate(positions,1)


def inputs(root):
    import torch
    record=read(root/'prepare_complete.json');assert record['passed'] and sha(root/'prepared.pt')==record['prepared_sha256']
    assert sha(OLD/'independent_final_audit.json')==record['old_final_audit_sha256']
    assert sha(ORIGIN/'S1/costs.json')==record['costs_sha256'] and sha(SOURCE/'S1/source_truth.json')==record['truth_sha256']
    data=torch.load(root/'prepared.pt',map_location='cpu');old=torch.load(OLD/'S1/descriptors.pt',map_location='cpu')
    split=record['split'];truth=read(SOURCE/'S1/source_truth.json');cost=read(ORIGIN/'S1/costs.json')
    assert split['fit']==read(OLD/'S1/split.json')['fit'] and split['heldout']==read(OLD/'S1/split.json')['heldout']
    index={r['record']['video']:r['record'] for r in cost['rows']};index.update({r['video']:r for r in cost['validation']})
    checked=0;err=0.;gt_checked=0
    for role,rows in data['roles'].items():
        prior={x['video']:x for x in old[role]}
        for row in rows:
            v=row['video'];raw=torch.load(index[v]['path'],map_location='cpu');previous=prior[v]
            for k in ('scores','labels','segments'):assert torch.equal(row[k],previous[k])
            assert torch.equal(row['hard'],raw['scores']>=.3)
            seq,valid,pos=numpy_samples(raw['feature'].numpy(),raw['valid'].numpy(),raw['boxes'][row['hard']].numpy())
            assert np.array_equal(valid,row['valid'].numpy())
            assert np.allclose(seq,row['seq'].numpy(),atol=2e-5,rtol=1e-6)
            if seq.size:err=max(err,float(np.abs(seq-row['seq'].numpy()).max()))
            assert torch.allclose(row['linear'],previous['descriptors']['max_linear'],atol=2e-5,rtol=1e-6)
            assert torch.equal(row['metadata'],row['linear'][:,-12:])
            inside=(seq[:,4:20]*valid[:,4:20,None]).sum(1)/np.maximum(valid[:,4:20].sum(1,keepdims=True),1)
            assert np.allclose(inside,row['inner'],atol=2e-5,rtol=1e-6)
            if role!='validation':
                boxes=raw['boxes'][row['hard']].numpy();gt=np.asarray(truth[v]['segments'],dtype=np.float32).reshape(-1,2)
                inter=np.maximum(0,np.minimum(boxes[:,None,1],gt[None,:,1])-np.maximum(boxes[:,None,0],gt[None,:,0]))
                iou=inter/np.maximum((boxes[:,1]-boxes[:,0])[:,None]+(gt[:,1]-gt[:,0])[None]-inter,1e-8)
                bg=iou.max(1)<.01 if len(gt) else np.ones(len(boxes),bool)
                assert np.array_equal(bg,row['bg'].numpy().astype(bool))
            else:assert 'bg' not in row and 'quality' not in row
            if role=='fit':
                boxes=np.asarray(truth[v]['segments'],dtype=np.float32).reshape(-1,2);labels=truth[v]['labels']
                gs,gm,gp=numpy_samples(raw['feature'].numpy(),raw['valid'].numpy(),boxes)
                items=[x for x in data['GTclips'] if x['video']==v]
                eligible=np.flatnonzero(gm[:,4:20].sum(1)>=4);assert len(items)==len(eligible)
                for item,i in zip(items,eligible):
                    assert int(item['label'])==labels[i] and np.allclose(gs[i],item['samples'],atol=2e-5,rtol=1e-6) and np.array_equal(gm[i],item['valid'])
                    ctx=np.r_[gp[i,:4],gp[i,20:]]
                    clean=~((ctx[:,None]>=boxes[None,:,0])&(ctx[:,None]<=boxes[None,:,1])).any(1)&np.r_[gm[i,:4],gm[i,20:]]
                    assert np.array_equal(clean,item['bg_context'])
                    gt_checked+=1
            checked+=1
    assert gt_checked==len(data['GTclips'])==record['GTclips'] and checked==58
    save(root/'independent_input_audit.json',dict(passed=True,prepared_sha256=record['prepared_sha256'],all_original_arrays_exact=True,
        all_video_samples_independent_numpy=True,videos=checked,GTclips=gt_checked,all_GT_context_excludes_any_task=True,max_sample_error=err,
        validation_GT_absent=True,source_only=True,prepare_sha256=sha(root/'prepare_complete.json')))


def native(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.background_four_routes_math import knn_distance
    root=entry.parent;route=entry.name;guard=first_cuda_guard(entry)
    gpu=read(root/route/'gpu_complete.json');assert gpu['passed']
    assert sha(root/'prepared.pt')==gpu['prepared_sha256'] and sha(root/route/'predictions.pt')==gpu['predictions_sha256']
    data=torch.load(root/'prepared.pt',map_location='cpu');saved=torch.load(root/route/'predictions.pt',map_location='cpu')
    proof=[];cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    if route=='knn':
        fit=data['roles']['fit'];bank=torch.cat([r['inner'] for r in fit]).cuda();bg=torch.cat([r['bg'] for r in fit]).bool().cuda()
        videos=torch.cat([torch.full((len(r['bg']),),i,dtype=torch.long) for i,r in enumerate(fit)]).cuda()
        for role,rows in data['roles'].items():
            for row in rows:
                same=next((i for i,r in enumerate(fit) if r['video']==row['video']),None)
                eligible=torch.ones(len(bg),dtype=torch.bool,device='cuda') if same is None else videos!=same
                dfg=knn_distance(row['inner'].cuda(),bank[eligible&~bg]);dbg=knn_distance(row['inner'].cuda(),bank[eligible&bg])
                for mode,score in [('foreground_knn',dfg),('dual_memory_knn',dfg-dbg)]:
                    actual=next(r for r in saved[role][mode] if r['video']==row['video'])['pbg']
                    p=torch.zeros(len(row['scores']),dtype=torch.float64);p[row['hard']]=score.cpu();assert torch.equal(p,actual)
                    proof.append(dict(role=role,mode=mode,video=row['video'],bitwise_equal=True))
    else:
        path=root/route/'epoch020.pt';assert sha(path)==gpu['checkpoint_sha256'];state=torch.load(path,map_location='cpu')
        for mode in MODES[route]:
            model=model_factory(route,mode).cuda();model.load_state_dict(state['models'][mode]);model.eval()
            assert base.tensor_digest(model.state_dict())==gpu['final_digests'][mode]
            digest=base.tensor_digest(model.state_dict())
            for role,rows in data['roles'].items():
                with torch.no_grad():
                    for row in rows:
                        parts=[]
                        for start in range(0,int(row['hard'].sum()),128):
                            sl=slice(start,start+128)
                            if route=='pauc':
                                stats=state['auxiliary']['normalization'];pred=model((row['linear'][sl].cuda()-stats['mean'].cuda())/stats['std'].cuda())
                            else:pred=model(row['seq'][sl].cuda(),row['valid'][sl].cuda(),row['metadata'][sl].cuda())
                            parts.append(pred[:,0].sigmoid().cpu())
                        p=torch.zeros(len(row['scores']));p[row['hard']]=torch.cat(parts) if parts else torch.empty(0)
                        actual=next(r for r in saved[role][mode] if r['video']==row['video'])['pbg'];assert torch.equal(p,actual)
                        proof.append(dict(role=role,mode=mode,video=row['video'],bitwise_equal=True))
            assert base.tensor_digest(model.state_dict())==digest
            del model
    # 构造新模块会消费初始化RNG；实际每次推理本身无随机性，另由生产逐mode守卫。
    save(root/'independent_native'/f'{route}.json',dict(passed=True,all_predictions_bitwise_equal=True,rows=proof,
        optimizer_steps=0,guard=guard,guard_sha256=sha(guard),source_gpu_sha256=sha(root/route/'gpu_complete.json'),
        code=code_identity(),precision_environment={k:os.environ.get(k) for k in ('NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}))


def results(root,route):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,background_mask,prediction_frame
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    done=read(root/route/'complete.json');gpu=read(root/route/'gpu_complete.json');native=read(root/'independent_native'/f'{route}.json')
    assert sha(root/route/'gpu_complete.json')==done['gpu_complete_sha256'] and native['all_predictions_bitwise_equal']
    assert native['source_gpu_sha256']==done['gpu_complete_sha256']
    ng=read(native['guard']);nl=read(ng['lease']);al=read(root/'independent_audit_launch.json')
    assert sha(native['guard'])==native['guard_sha256'] and sha(ng['lease'])==ng['lease_sha256']
    assert ng['owner_pid']==nl['owner_pid']==al['pid'] and nl['commit']==al['head']
    assert ng['observed']['uuid']==nl['uuid'] and ng['observed']['free_mib']>=12288
    launch=read(root/'_launch.json')
    for p in (gpu['guard'],):
        g=read(p);lease=read(g['lease']);assert sha(p)==gpu['guard_sha256'] and sha(g['lease'])==g['lease_sha256']
        assert g['owner_pid']==lease['owner_pid']==launch['pid'] and lease['commit']==launch['head']
        assert g['observed']['uuid']==lease['uuid'] and g['observed']['free_mib']>=12288
    if route!='knn':
        history=read(root/route/'history.json');assert [x['epoch'] for x in history]==list(range(1,21));expected_steps=sum(h['steps'] for h in history)
        checkpoint_sha={}
        for epoch in (1,5,10,20):
            p=root/route/f'epoch{epoch:03d}.pt';state=torch.load(p,map_location='cpu');assert state['history']==history[:epoch]
            assert {'models','optimizers','optimizer_counters','order_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','auxiliary','spec'}<=set(state)
            assert set(state['models'])==set(MODES[route])==set(state['optimizers'])
            n=sum(h['steps'] for h in history[:epoch])
            for opt in state['optimizers'].values():assert opt['state'] and all(float(v['step'])==n for v in opt['state'].values())
            if route=='pauc':assert state['auxiliary']['u_pos'].numel()==5993 and torch.isfinite(state['auxiliary']['u_pos']).all()
            checkpoint_sha[str(epoch)]=sha(p)
        assert (root/route/'latest.pt').stat().st_ino==(root/route/'epoch020.pt').stat().st_ino
    split=read(root/'prepare_complete.json')['split'];gt=read(SOURCE/'S1/source_truth.json');cfg=read(SOURCE/'S1/config.json')
    truth={role:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[role] for b,c in zip(gt[v]['segments'],gt[v]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    outputs=torch.load(root/route/'predictions.pt',map_location='cpu');original=torch.load(root/'prepared.pt',map_location='cpu')['roles'];report={}
    for role,modes in outputs.items():
        frames={m:prediction_frame(rows) for m,rows in modes.items()};frame=next(iter(frames.values()));bg=background_mask(truth[role],frame)
        for m,rows in modes.items():
            assert len(rows)==len(original[role])
            for row,raw in zip(rows,original[role]):
                assert row['video']==raw['video']
                for k in ('segments','scores','labels'):assert torch.equal(row[k],raw[k])
                assert (row['pbg'][~raw['hard']]==0).all()
        engines={};positions={};base=[]
        for c in range(10):
            idx=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[idx]);base.append(engines[c].evaluate(np.ones(len(idx),bool))['ap'])
        # 原始AP从本次真值/原框重新算，不从删后报告当baseline。
        report[role]={}
        for m in (*MODES[route],'confidence'):
            f=frame if m=='confidence' else frames[m];record=done['results'][role][m];aucs=[];budgets=[]
            assert np.allclose(base,record['original_per_class_AP'],atol=1e-7,rtol=0)
            for v,ids in f.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(ids);idx=idx[f.iloc[idx].score.to_numpy()>=.3]
                a=independent_auc(bg[idx],1-f.iloc[idx].score.to_numpy() if m=='confidence' else f.iloc[idx].pbg.to_numpy())
                if a is not None:aucs.append(a)
            assert len(aucs)==record['AUC_valid_videos'] and abs(float(np.mean(aucs))-record['video_macro_AUC'])<1e-12
            for rate,reported in zip(SPEC['rates'],record['budgets']):
                selected=[]
                for c,idx in positions.items():
                    pool=[int(i) for i in idx if float(f.score.iloc[i])>=.3]
                    pool.sort(key=lambda i:(float(f.score.iloc[i]) if m=='confidence' else -float(f.pbg.iloc[i]),i))
                    selected.extend(pool[:math_floor(rate*len(pool))])
                selected.sort();assert selected==reported['selected']
                keep=np.ones(len(f),bool);keep[selected]=False
                ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)]
                delta=100*(np.mean(ap)-np.mean(base))
                assert np.allclose(ap,reported['per_class_AP'],atol=1e-7,rtol=0) and abs(delta-reported['delta_pp'])<1e-7
                assert int(bg[selected].sum())==reported['BG_removed'] and int((~bg[selected]).sum())==reported['nonBG_removed']
                budgets.append(dict(rate=rate,delta_pp=float(delta),BG_removed=reported['BG_removed'],nonBG_removed=reported['nonBG_removed']))
            report[role][m]=dict(video_macro_AUC=record['video_macro_AUC'],AUC_valid_videos=len(aucs),budgets=budgets)
    save(root/'independent_results'/f'{route}.json',dict(passed=True,complete_sha256=sha(root/route/'complete.json'),native_sha256=sha(root/'independent_native'/f'{route}.json'),
        source_only=True,shift='S1',results=report,all_fixed_budget_indices_and_AP_recomputed=True,all_original_arrays_exact=True))


def math_floor(x):return int(x)


def watch(root):
    from tools.baseline_gpu import snapshot
    inputs(root);active={};pending=list(ROUTES)
    def stop(sig,frame):
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    while not all((root/r/'gpu_complete.json').exists() for r in ROUTES):
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped; preserve all audit artifacts')
        time.sleep(3)
    while read(root/'state.json')['stage']=='train':time.sleep(3)
    while pending or active:
        for r,t in list(active.items()):
            if t['p'].poll() is None:continue
            t['log'].close();del active[r]
            if t['p'].returncode:raise RuntimeError(r+' native replay failed')
        for g in (0,1,2,3):
            if not pending or g in [t['gpu'] for t in active.values()] or snapshot(g)['free_mib']<12288:continue
            r=pending.pop(0);observed=snapshot(g);lease=root/r/'audit_lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
            log=(root/r/'native_audit.log').open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--native',str(root/r)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[r]=dict(p=p,gpu=g,log=log)
        time.sleep(3)
    while not (root/'final.json').exists():
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped before final')
        time.sleep(3)
    for r in ROUTES:results(root,r);print('AUDITED',r,flush=True)
    final=read(root/'final.json')
    for r in ROUTES:assert sha(root/r/'complete.json')==final['complete_sha256'][r]
    save(root/'independent_final_audit.json',dict(passed=True,source_only=True,shift='S1',exact_routes=list(ROUTES),input_audit_sha256=sha(root/'independent_input_audit.json'),
        final_sha256=sha(root/'final.json'),audits={r:sha(root/'independent_results'/f'{r}.json') for r in ROUTES}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',type=Path);p.add_argument('--native',type=Path);p.add_argument('--inputs',type=Path);a=p.parse_args()
    if a.native:native(a.native)
    elif a.inputs:inputs(a.inputs)
    else:watch(a.watch)
