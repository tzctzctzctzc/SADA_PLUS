"""全输入与混淆对独审、原GPU逐bit重放、完整固定预算AP复核。"""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_highscore_pairs import DIAG,SPEC,TASKS,OLD,SOURCE


def np_samples(f,valid,b):
    size=b[:,1]-b[:,0];values=[];masks=[]
    for a,z,n in ((b[:,0]-.25*size,b[:,0],4),(b[:,0],b[:,1],16),(b[:,1],b[:,1]+.25*size,4)):
        x=a[:,None]+(z-a)[:,None]*(np.arange(n,dtype=np.float32)+np.float32(.5))/np.float32(n)
        lo=np.floor(x).astype(int);hi=np.ceil(x).astype(int);w=x-lo.astype(np.float32)
        l=np.clip(lo,0,f.shape[-1]-1);h=np.clip(hi,0,f.shape[-1]-1)
        masks.append((x>=0)&(x<=f.shape[-1]-1)&valid[l]&valid[h])
        values.append((f[:,l]*(1-w)[None]+f[:,h]*w[None]).transpose(1,2,0))
    return np.concatenate(values,1),np.concatenate(masks,1)


def audit_inputs(root):
    import torch
    from tools.foreground_oracle_ap import prediction_frame
    from tools.highscore_confusion_diagnosis import ORIGIN
    reports={}
    for s in SPEC['shifts']:
        cost=read(DIAG/s/'fit_costs_and_pairs.json');assert sha(DIAG/s/'inputs.pt')==cost['inputs_sha256']
        roles=torch.load(DIAG/s/'inputs.pt',map_location='cpu');sourcecost=read(ORIGIN/s/'costs.json');split=read(OLD/s/'split.json')
        assert sha(OLD/s/'split.json')==cost['old_split_sha256'] and sha(SOURCE/s/'source_truth.json')==cost['source_truth_sha256']
        records={r['record']['video']:r['record'] for r in sourcecost['rows']};records.update({r['video']:r for r in sourcecost['validation']})
        videos=0;error=0.
        for role,rows in roles.items():
            if role in ('fit','heldout'):assert {r['video'] for r in rows}==set(split[role])
            for r in rows:
                raw=torch.load(records[r['video']]['path'],map_location='cpu');hard=raw['scores']>=.3
                for k in ('scores','labels'):assert torch.equal(raw[k],r[k])
                assert torch.equal(raw['segments' if role=='validation' else 'boxes'],r['segments']) and torch.equal(hard,r['hard'])
                x,v=np_samples(raw['feature'].numpy(),raw['valid'].numpy(),raw['boxes'][hard].numpy())
                assert np.allclose(x,r['seq'],atol=2e-5,rtol=1e-6) and np.array_equal(v,r['valid'])
                if x.size:error=max(error,float(np.abs(x-r['seq'].numpy()).max()))
                if role=='validation':assert 'bg' not in r
                videos+=1
        rows=roles['fit'];frame=prediction_frame([{k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=torch.zeros(len(r['scores']))) for r in rows]);ids=np.flatnonzero(frame.score.to_numpy()>=.3)
        assert ids.tolist()==cost['hard_original_indices']
        label=frame.label.to_numpy()[ids];score=frame.score.to_numpy()[ids];length=(frame['t-end']-frame['t-start']).to_numpy()[ids];vid=frame['video-id'].to_numpy()[ids];bg=np.asarray(cost['BG']);delta=np.asarray(cost['delta'])
        assert np.array_equal(bg,torch.cat([r['bg'] for r in rows]).numpy().astype(bool))
        sb=np.zeros(len(ids),int);lb=np.zeros(len(ids),int)
        for c in np.unique(label):
            z=np.flatnonzero(label==c);se=np.quantile(score[z],np.arange(1,10)/10);le=np.quantile(length[z],(.25,.5,.75))
            assert np.allclose(se,cost['pairing']['fit_only_edges'][str(c)]['score']) and np.allclose(le,cost['pairing']['fit_only_edges'][str(c)]['length'])
            sb[z]=np.searchsorted(se,score[z],side='right');lb[z]=np.searchsorted(le,length[z],side='right')
        features=torch.cat([r['inner'] for r in rows]).numpy();features/=np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
        expected=[]
        for b in np.flatnonzero(bg&(delta>0)):
            fg=np.flatnonzero((~bg)&(delta<0)&(label==label[b])&(sb==sb[b])&(lb==lb[b])&(vid!=vid[b]))
            if len(fg):
                similarities=features[fg]@features[b]
                order=sorted(range(len(fg)),key=lambda k:(-float(similarities[k]),int(fg[k])))[:4]
                expected.extend((int(b),int(fg[k])) for k in order)
        actual=[(p['BG'],p['foreground']) for p in cost['pairing']['pairs']];assert actual==expected
        reports[s]=dict(passed=True,videos=videos,max_sampling_error=error,input_sha256=cost['inputs_sha256'],pairs_sha256=sha(DIAG/s/'fit_costs_and_pairs.json'),all_pair_members_and_visual_neighbors_recomputed=True,pairs=len(actual))
    save(root/'independent_input_pair_audit.json',dict(passed=True,source_only=True,results=reports))


def native(entry):
    import torch
    from tools.highscore_pair_math import make_model
    from tools.run_proposal_foreground import first_cuda_guard
    from tools import baseline_engine as base
    guard=first_cuda_guard(entry);gpu=read(entry/'gpu_complete.json');shift=entry.name.split('_')[0]
    assert sha(entry/'epoch020.pt')==gpu['checkpoint_sha256'] and sha(entry/'predictions.pt')==gpu['predictions_sha256']
    state=torch.load(entry/'epoch020.pt',map_location='cpu');model=make_model().cuda();model.load_state_dict(state['model']);model.eval()
    assert base.tensor_digest(model.state_dict())==gpu['final_digest']
    roles=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');outputs=torch.load(entry/'predictions.pt',map_location='cpu');checked=0
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);before=base.tensor_digest(model.state_dict())
    with torch.no_grad():
        for role,rows in roles.items():
            for r,saved in zip(rows,outputs[role]):
                parts=[]
                for start in range(0,len(r['seq']),128):
                    sl=slice(start,start+128);parts.append(model(r['seq'][sl].cuda(),r['valid'][sl].cuda(),r['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                actual=torch.cat(parts) if parts else torch.empty(0);assert torch.equal(actual,saved['pbg'][r['hard']]);checked+=1
    assert before==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'independent_native.json',dict(passed=True,videos=checked,all_predictions_bitwise_equal=True,parameters_and_RNG_unchanged=True,optimizer_steps=0,guard=guard,guard_sha256=sha(guard),source_gpu_sha256=sha(entry/'gpu_complete.json')))


def audit_result(root,task):
    import torch,pandas as pd
    from tools import baseline_engine as base
    from tools.audit_background_ap_cost import IndependentAP,background
    from tools.foreground_oracle_ap import prediction_frame
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.audit_background_fit_gap import independent_auc
    e=root/task;shift,mode=task.split('_');done=read(e/'complete.json');gpu=read(e/'gpu_complete.json');launch=read(Path(str(root)+'_launch.json'));native=read(e/'independent_native.json')
    assert sha(e/'gpu_complete.json')==done['gpu_complete_sha256']==native['source_gpu_sha256']
    for d,owner,head in [(gpu,launch['pid'],launch['head']),(native,read(root/'independent_audit_launch.json')['pid'],read(root/'independent_audit_launch.json')['head'])]:
        guard=read(d['guard']);lease=read(guard['lease']);assert sha(d['guard'])==d['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256']
        assert guard['owner_pid']==lease['owner_pid']==owner and lease['commit']==head and guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
    history=read(e/'history.json');assert [h['epoch'] for h in history]==list(range(1,21));checkpoints={}
    for epoch in (1,5,10,20):
        p=e/f'epoch{epoch:03d}.pt';state=torch.load(p,map_location='cpu')
        assert {'model','optimizer','order_rng','pair_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','history','input_sha256','pairs_sha256','spec'}<=set(state)
        assert state['history']==history[:epoch] and state['input_sha256']==gpu['input_sha256'] and state['pairs_sha256']==gpu['pairs_sha256']
        n=sum(h['steps'] for h in history[:epoch]);assert all(float(v['step'])==n for v in state['optimizer']['state'].values())
        checkpoints[str(epoch)]=sha(p)
    assert checkpoints['20']==gpu['checkpoint_sha256'] and base.tensor_digest(state['model'])==gpu['final_digest']
    assert (e/'latest.pt').stat().st_ino==(e/'epoch020.pt').stat().st_ino
    roles=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');outputs=torch.load(e/'predictions.pt',map_location='cpu')
    split=read(OLD/shift/'split.json');truth=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json')
    gt={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(truth[v]['segments'],truth[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')};gt['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);result={}
    for role,rows in outputs.items():
        for row,original in zip(rows,roles[role]):
            assert row['video']==original['video']
            for k in ('scores','labels','segments'):assert torch.equal(row[k],original[k])
            assert (row['pbg'][~original['hard']]==0).all()
        f=prediction_frame(rows);bg=background(gt[role],f);positions={};engines={};baselines=[]
        for c in range(10):
            idx=np.flatnonzero(f.label.to_numpy()==c);positions[c]=idx;engines[c]=IndependentAP(gt[role][gt[role].label==c],f.iloc[idx].reset_index(drop=True));baselines.append(engines[c].ap(np.ones(len(idx),bool)).tolist())
        result[role]={}
        for method in (mode,'confidence'):
            record=done['results'][role][method];assert np.allclose(baselines,record['original_per_class_AP'],atol=1e-7,rtol=0);aucs=[];budgets=[]
            for _,idx in f.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[f.iloc[idx].score.to_numpy()>=.3];a=independent_auc(bg[idx],1-f.iloc[idx].score.to_numpy() if method=='confidence' else f.iloc[idx].pbg.to_numpy())
                if a is not None:aucs.append(a)
            assert len(aucs)==record['AUC_valid_videos'] and abs(float(np.mean(aucs))-record['video_macro_AUC'])<1e-12
            for rate,rep in zip(SPEC['rates'],record['budgets']):
                selected=[]
                for c,idx in positions.items():
                    pool=[int(i) for i in idx if float(f.score.iloc[i])>=.3];pool.sort(key=lambda i:(float(f.score.iloc[i]) if method=='confidence' else -float(f.pbg.iloc[i]),i));selected.extend(pool[:int(rate*len(pool))])
                selected.sort();assert selected==rep['selected'];keep=np.ones(len(f),bool);keep[selected]=False
                ap=[engines[c].ap(keep[positions[c]]).tolist() for c in range(10)];delta=100*(np.mean(ap)-np.mean(baselines))
                assert np.allclose(ap,rep['per_class_AP'],atol=1e-7,rtol=0) and abs(delta-rep['delta_pp'])<1e-7
                assert int(bg[selected].sum())==rep['BG_removed'] and int((~bg[selected]).sum())==rep['nonBG_removed']
                budgets.append(dict(rate=rate,delta_pp=float(delta),BG_removed=rep['BG_removed'],nonBG_removed=rep['nonBG_removed']))
            result[role][method]=dict(video_macro_AUC=record['video_macro_AUC'],AUC_valid_videos=len(aucs),budgets=budgets)
    save(root/'independent_results'/f'{task}.json',dict(passed=True,complete_sha256=sha(e/'complete.json'),native_sha256=sha(e/'independent_native.json'),checkpoints=checkpoints,source_only=True,results=result))


def watch(root):
    from tools.baseline_gpu import snapshot
    audit_inputs(root)
    while not all((root/t/'gpu_complete.json').exists() for t in TASKS):
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped; preserve audit state')
        time.sleep(3)
    while read(root/'state.json')['stage']=='train':time.sleep(3)
    active={};pending=list(TASKS)
    while pending or active:
        for task,t in list(active.items()):
            if t['p'].poll() is None:continue
            t['log'].close();del active[task]
            if t['p'].returncode:raise RuntimeError(task+' native replay failed')
        for task in list(pending):
            original_guard=read(read(root/task/'gpu_complete.json')['guard']);g=original_guard['observed']['index']
            if g in [t['gpu'] for t in active.values()] or snapshot(g)['free_mib']<12288:continue
            observed=snapshot(g);assert observed['uuid']==original_guard['observed']['uuid'];lease=root/task/'audit_lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
            log=(root/task/'native_audit.log').open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--native',str(root/task)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log);pending.remove(task)
        time.sleep(3)
    while not (root/'final.json').exists():
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped before final')
        time.sleep(3)
    for task in TASKS:audit_result(root,task);print('AUDITED',task,flush=True)
    for s in SPEC['shifts']:
        a=read(root/(s+'_control')/'gpu_complete.json');b=read(root/(s+'_pairs')/'gpu_complete.json');assert a['initial_digest']==b['initial_digest']
    save(root/'independent_final_audit.json',dict(passed=True,source_only=True,shifts=SPEC['shifts'],exact_tasks=list(TASKS),input_audit_sha256=sha(root/'independent_input_pair_audit.json'),final_sha256=sha(root/'final.json'),audits={t:sha(root/'independent_results'/f'{t}.json') for t in TASKS}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',type=Path);p.add_argument('--native',type=Path);a=p.parse_args()
    if a.native:native(a.native)
    else:watch(a.watch)
