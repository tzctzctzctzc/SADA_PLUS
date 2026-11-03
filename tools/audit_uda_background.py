"""UDA输入、无标签范围、EMA恢复和完整AP独审；目标训练GT始终不读取。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_uda_background import SPEC,TASKS,EXTRACT,DIAG,OLD,SOURCE
from tools.audit_layer_signals import guardcheck
from tools.audit_highscore_pairs import audit_inputs
from tools.prepare_uda_background import target_rows


def semantic_numpy(boxes,truth,labels):
    from tools.audit_background_ap_cost import overlaps
    out=np.full(len(boxes),-1,dtype=np.int64);matrix=overlaps(boxes,truth);task=(labels>=0)&(labels<10);other=(labels>=10)&(labels<97)
    tm=matrix[:,task].max(1) if task.any() else np.zeros(len(boxes))
    for mask,eligible in [(task,tm>=.01),(other,tm<.01)]:
        if mask.any():
            best=matrix[:,mask].argmax(1);score=matrix[:,mask].max(1);ok=eligible&(score>=.1);out[ok]=labels[mask][best[ok]]
    return out


def audit_scope(root):
    import torch
    audit_inputs(root);prepared=read(root/'source_prepare.json');reports={}
    for s in SPEC['shifts']:
        record=prepared['results'][s];entry=root/'inputs'/s;assert sha(entry/'source.pt')==record['source_sha256'];assert sha(entry/'source_fit_97truth.json')==record['source_fit_97truth_sha256']
        roles=torch.load(entry/'source.pt',map_location='cpu');old=torch.load(DIAG/s/'inputs.pt',map_location='cpu');truth=read(entry/'source_fit_97truth.json');split=read(OLD/s/'split.json')
        assert set(truth)==set(split['fit']);cfg=read(SOURCE/s/'config.json');assert sha(cfg['dataset']['json_file'])==record['source_manifest_sha256'];raw=read(cfg['dataset']['json_file'])['database'];semcounts={}
        for role,rows in roles.items():
            assert [r['video'] for r in rows]==[r['video'] for r in old[role]]
            for r,o in zip(rows,old[role]):
                for k in ('scores','labels','segments','seq','valid','metadata','hard'):assert torch.equal(r[k],o[k])
                if role=='fit':
                    gt=np.asarray(truth[r['video']]['segments'],dtype=np.float32).reshape(-1,2);labels=np.asarray(truth[r['video']]['labels']);expected=semantic_numpy(r['segments'][r['hard']].numpy(),gt,labels)
                    assert np.array_equal(expected,r['semantic'].numpy()) and not ((r['bg'].numpy()==0)&(expected>=10)).any()
                    for c in expected:semcounts[str(c)]=semcounts.get(str(c),0)+1
                    assert labels.tolist()==[a['label_id'] for a in raw[r['video']]['annotations']]
                else:assert 'semantic' not in r
        assert semcounts==record['semantic_counts'];sets={}
        for role in ('target_train','target_val'):
            manifest=entry/(role+'_unlabeled.json');assert sha(manifest)==record['unlabeled_manifest_sha256'][role];unlabeled=read(manifest)['database'];assert all('annotations' not in x for x in unlabeled.values());sets[role]=set(unlabeled)
            complete=read(root/(s+'_'+role)/'complete.json');assert sha(manifest)==complete['manifest_sha256'] and {r['video'] for r in complete['records']}==sets[role]
            for rec in complete['records']:
                assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');assert not {'bg','quality','semantic','gt','annotations','pbg'}&set(r)
        assert not sets['target_train']&sets['target_val']
        assert not (sets['target_train']|sets['target_val'])&{r['video'] for rows in roles.values() for r in rows}
        reports[s]=dict(source_members=record['source_members'],semantic_counts=semcounts,target_train_videos=len(sets['target_train']),target_val_videos=len(sets['target_val']),source_truth_fit_only=True,target_GT_in_training=False)
    save(root/'independent_UDA_scope.json',dict(passed=True,source_prepare_sha256=sha(root/'source_prepare.json'),reports=reports))


def native_features(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.audit_highscore_pairs import np_samples
    guard=first_cuda_guard(entry);s=entry.name.split('_')[0];done=read(entry/'complete.json');model=load_detector(SOURCE/s);cfg=read(SOURCE/s/'config.json');captured={}
    handle=model.neck_source.register_forward_hook(lambda m,args,out:captured.update(fpn=out));digest=base.tensor_digest(model.state_dict());assert digest==done['detector_digest'];cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);checked=[]
    with torch.no_grad():
        for rec in done['records']:
            assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');d=cfg['dataset'];path=Path(d['feat_folder'])/((d['file_prefix'] or '')+r['video']+d['file_ext']);raw=torch.from_numpy(np.load(path)['feats'].astype(np.float32)[::d['downsample_rate']].T.copy())
            video=[dict(feats=raw.cuda(),segments=torch.empty(0,2,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=r['video_metadata'])]
            result=model(video_list_src=video,video_list_tgt=video,mode='test')[0][0]
            for k in ('segments','scores','labels'):assert torch.equal(result[k].cpu(),r[k])
            f,m=captured['fpn'];x,valid=np_samples(f[0][0].cpu().numpy(),m[0][0,0].bool().cpu().numpy(),r['grid_boxes'][r['hard']].numpy())
            assert np.array_equal(valid,r['valid']) and np.allclose(x,r['seq'],atol=2e-5,rtol=1e-6)
            meta=r['video_metadata'];grid=(r['segments']*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride'];assert torch.equal(grid,r['grid_boxes'])
            score=r['scores'][r['hard']].numpy();label=r['labels'][r['hard']].numpy();box=r['grid_boxes'][r['hard']].numpy();metadata=np.zeros((len(score),12),np.float32);metadata[np.arange(len(score)),label]=1
            prob=np.clip(score,1e-5,1-1e-5);metadata[:,10]=np.log(prob/(1-prob))/5;metadata[:,11]=np.log1p((box[:,1]-box[:,0])/f[0].shape[-1]);assert np.allclose(metadata,r['metadata'],atol=2e-7,rtol=1e-6)
            checked.append(dict(video=r['video'],input_sha256=rec['sha256'],raw_sha256=sha(path),max_sampling_error=float(np.max(np.abs(x-r['seq'].numpy()))) if x.size else 0.))
    handle.remove();assert digest==base.tensor_digest(model.state_dict()) and all(p.grad is None for p in model.parameters()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'independent_native_features.json',dict(passed=True,records=checked,all_original_arrays_bitwise_equal=True,all_sampling_numpy_checked=True,source_complete_sha256=sha(entry/'complete.json'),guard=guard,guard_sha256=sha(guard),parameters_RNG_unchanged=True))


def native_head(entry):
    import torch
    from tools import baseline_engine as base
    from tools.uda_background_math import UDAHead
    from tools.prepare_uda_background import target_rows
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);s=entry.name.split('_')[0];gpu=read(entry/'gpu_complete.json');assert sha(entry/'epoch020.pt')==gpu['checkpoint_sha256'] and sha(entry/'predictions.pt')==gpu['predictions_sha256']
    state=torch.load(entry/'epoch020.pt',map_location='cpu');model=UDAHead().cuda();model.load_state_dict(state['ema']);model.eval();roles=torch.load(entry.parent/'inputs'/s/'source.pt',map_location='cpu');roles['target_val']=target_rows(entry.parent,s,'target_val');outputs=torch.load(entry/'predictions.pt',map_location='cpu')
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);digest=base.tensor_digest(model.state_dict());checked=0
    with torch.no_grad():
        for role,rows in roles.items():
            for r,o in zip(rows,outputs[role]):
                parts=[]
                for start in range(0,len(r['seq']),128):
                    sl=slice(start,start+128);parts.append(model(r['seq'][sl].cuda(),r['valid'][sl].cuda(),r['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                actual=torch.cat(parts) if parts else torch.empty(0);assert torch.equal(actual,o['pbg'][r['hard']]);checked+=1
    assert digest==gpu['final_ema_digest']==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'independent_native.json',dict(passed=True,mode_videos=checked,all_predictions_bitwise_equal=True,source_gpu_sha256=sha(entry/'gpu_complete.json'),guard=guard,guard_sha256=sha(guard),parameters_RNG_unchanged=True,optimizer_steps=0))


def native_density(entry):
    import torch
    from tools import baseline_engine as base
    from tools.prepare_uda_background import target_rows
    from tools.run_layer_signals import seed
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();s=entry.name.split('_')[0];complete=read(entry/'complete.json');assert sha(entry/'density.pt')==complete['artifact_sha256'];saved=torch.load(entry/'density.pt',map_location='cpu')
    source=torch.load(entry.parent/'inputs'/s/'source.pt',map_location='cpu')['fit'];target=target_rows(entry.parent,s,'target_train')
    def mean(rows):return torch.cat([(r['seq'][:,4:20]*r['valid'][:,4:20,None]).sum(1)/r['valid'][:,4:20].sum(1,keepdim=True).clamp_min(1) for r in rows])
    a=mean(source);b=mean(target);sc=torch.cat([r['metadata'][:,:10].argmax(1) for r in source]);tc=torch.cat([r['metadata'][:,:10].argmax(1) for r in target]);centers=torch.stack([a[sc==c].mean(0) if (sc==c).any() else a.mean(0) for c in range(10)]);std=(a-centers[sc]).std(0,unbiased=False).clamp_min(1e-5)
    assert torch.equal(centers,saved['source_class_means']) and torch.equal(std,saved['source_scale']);a=((a-centers[sc])/std).cuda();b=((b-centers[tc])/std).cuda()
    def weights(rows):
        x=torch.cat([torch.full((len(r['seq']),),1/max(1,len(r['seq']))) for r in rows]);return x/x.sum()
    sw=weights(source);tw=weights(target);g=torch.Generator().manual_seed(311107);model=torch.nn.Linear(512,1).cuda();torch.nn.init.zeros_(model.weight);torch.nn.init.zeros_(model.bias);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.05)
    for _ in range(100):
        si=torch.multinomial(sw,256,replacement=True,generator=g).cuda();ti=torch.multinomial(tw,256,replacement=True,generator=g).cuda();z=model(torch.cat((a[si],b[ti])))[:,0];y=torch.cat((torch.zeros(256,device='cuda'),torch.ones(256,device='cuda')))
        opt.zero_grad(set_to_none=True);torch.nn.functional.binary_cross_entropy_with_logits(z,y).backward();opt.step()
    assert base.tensor_digest(model.state_dict())==base.tensor_digest(saved['model']) and base.tensor_digest(opt.state_dict())==base.tensor_digest(saved['optimizer']) and torch.equal(g.get_state(),saved['order_rng'])
    with torch.no_grad():p=model(a)[:,0].sigmoid().cpu();ratio=(p/(1-p).clamp_min(1e-6)).clamp(.25,4)
    assert torch.equal(ratio,saved['weights'])
    save(entry/'independent_density.json',dict(passed=True,complete_sha256=sha(entry/'complete.json'),all100steps_model_optimizer_sampler_bitwise_equal=True,all_weights_bitwise_equal=True,no_target_GT=True,guard=guard,guard_sha256=sha(guard)))


AP_CACHE={}


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
        assert {'model','ema','optimizer','order_rng','pair_rng','target_stream','augmentation_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','history','global_step','source_sha256','target_train_complete_sha256','density_complete_sha256','spec'}<=set(state)
        assert state['history']==history[:epoch] and state['source_sha256']==read(root/'source_prepare.json')['results'][shift]['source_sha256']
        assert state['target_train_complete_sha256']==sha(root/(shift+'_target_train')/'complete.json') and state['density_complete_sha256']==sha(root/(shift+'_density')/'complete.json')
        n=sum(h['steps'] for h in history[:epoch]);assert state['global_step']==n and int(state['target_stream']['seen'].sum())==n*128 and all(float(v['step'])==n for v in state['optimizer']['state'].values())
        checkpoints[str(epoch)]=sha(p)
    assert checkpoints['20']==gpu['checkpoint_sha256'] and base.tensor_digest(state['ema'])==gpu['final_ema_digest'] and base.tensor_digest(state['model'])==gpu['final_student_digest']
    assert (e/'latest.pt').stat().st_ino==(e/'epoch020.pt').stat().st_ino
    roles=torch.load(root/'inputs'/shift/'source.pt',map_location='cpu');roles['target_val']=target_rows(root,shift,'target_val');outputs=torch.load(e/'predictions.pt',map_location='cpu')
    split=read(OLD/shift/'split.json');truth=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json')
    gt={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(truth[v]['segments'],truth[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')};gt['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);gt['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);result={}
    for role,rows in outputs.items():
        for row,original in zip(rows,roles[role]):
            assert row['video']==original['video']
            for k in ('scores','labels','segments'):assert torch.equal(row[k],original[k])
            assert (row['pbg'][~original['hard']]==0).all()
        f=prediction_frame(rows);bg=background(gt[role],f);positions={};engines={};baselines=[]
        if (shift,role) not in AP_CACHE:
            for c in range(10):
                idx=np.flatnonzero(f.label.to_numpy()==c);positions[c]=idx;engines[c]=IndependentAP(gt[role][gt[role].label==c],f.iloc[idx].reset_index(drop=True));baselines.append(engines[c].ap(np.ones(len(idx),bool)).tolist())
            AP_CACHE[(shift,role)]=(positions,engines,baselines)
        else:positions,engines,baselines=AP_CACHE[(shift,role)]
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
    save(root/'independent_results'/f'{task}.json',dict(passed=True,complete_sha256=sha(e/'complete.json'),native_sha256=sha(e/'independent_native.json'),checkpoints=checkpoints,UDA_training=True,target_train_GT_read=False,results=result))


def watch(root):
    import torch
    from tools import baseline_engine as base
    from tools.baseline_gpu import snapshot
    while not all((root/t/'complete.json').exists() for t in EXTRACT):
        if (root/'error.json').exists():raise RuntimeError('main failed')
        time.sleep(3)
    audit_scope(root)
    while not all((root/t/'gpu_complete.json').exists() for t in TASKS):
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped')
        time.sleep(3)
    while read(root/'state.json')['stage']=='train':time.sleep(3)
    launch=read(Path(str(root)+'_launch.json'));al=read(root/'independent_audit_launch.json')
    for stage,tasks in [('features',EXTRACT),('density',('S1_density','S2_density')),('head',TASKS)]:
        pending=list(tasks);active={}
        while pending or active:
            for task,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[task]
                if t['p'].returncode:raise RuntimeError(task+'/'+stage+' independent replay failed')
            for task in list(pending):
                key='gpu_complete.json' if stage=='head' else 'complete.json';src=read(root/task/key);guardcheck(src,launch['pid'],launch['head']);g=read(src['guard'])['observed']['index']
                if g in [t['gpu'] for t in active.values()] or snapshot(g)['free_mib']<12288:continue
                observed=snapshot(g);assert observed['uuid']==read(src['guard'])['observed']['uuid'];lease=root/task/('independent_'+stage+'_lease.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=observed['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease));log=(root/task/('independent_'+stage+'.log')).open('w')
                proc=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--native',str(root/task),'--stage',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=proc,gpu=g,log=log);pending.remove(task)
            time.sleep(3)
    while not (root/'final.json').exists():
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped')
        time.sleep(3)
    for task in TASKS:audit_result(root,task);print('AUDITED',task,flush=True)
    warm={}
    for s in SPEC['shifts']:
        warm[s]={}
        for mode in ('ontology','coral','teacher'):
            st=torch.load(root/(s+'_'+mode)/'epoch005.pt',map_location='cpu');warm[s][mode]=dict(model=base.tensor_digest(st['model']),ema=base.tensor_digest(st['ema']),optimizer=base.tensor_digest(st['optimizer']),order=base.tensor_digest(st['order_rng']),pair=base.tensor_digest(st['pair_rng']))
        assert warm[s]['ontology']==warm[s]['coral']==warm[s]['teacher']
        assert len({read(root/(s+'_'+m)/'gpu_complete.json')['initial_digest'] for m in SPEC['modes']})==1
    for task in EXTRACT:guardcheck(read(root/task/'independent_native_features.json'),al['pid'],al['head'])
    for task in ('S1_density','S2_density'):guardcheck(read(root/task/'independent_density.json'),al['pid'],al['head'])
    save(root/'independent_final_audit.json',dict(passed=True,UDA_training=True,target_train_GT_read=False,exact_tasks=TASKS,source_scope_sha256=sha(root/'independent_UDA_scope.json'),
        final_sha256=sha(root/'final.json'),audits={t:sha(root/'independent_results'/(t+'.json')) for t in TASKS},target_feature_audits={t:sha(root/t/'independent_native_features.json') for t in EXTRACT},
        density_audits={t:sha(root/t/'independent_density.json') for t in ('S1_density','S2_density')},shared_source_warmup5_bitwise_equal=warm))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',type=Path);p.add_argument('--native',type=Path);p.add_argument('--stage',choices=['features','density','head']);a=p.parse_args()
    if a.native:{'features':native_features,'density':native_density,'head':native_head}[a.stage](a.native)
    else:watch(a.watch)

