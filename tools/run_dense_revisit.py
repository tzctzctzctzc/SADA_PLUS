"""旧证据的新用法：全时序任务占据、97类覆盖、目标时序自监督、双删除效用。"""
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
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.dense_revisit_math import MODES
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
UDA=Path('/vepfs/vlms/highscore_uda_ideas_354cf35_1103')
DIAG=Path('/vepfs/vlms/highscore_confusion_diag_36c0916_1103')
LAYER=Path('/vepfs/vlms/highscore_layer_signals_bdcc520_1103')
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in MODES)
SPEC=dict(schema='highscore-full-sequence-revisit-v1',shifts=['S1','S2'],modes=MODES,tasks=TASKS,
    hypotheses=dict(dense10='train raw2304 temporal encoder on ALL sourceFIT task annotations and timeline cells; aggregate any-task evidence inside fixed proposals',
        dense97='dense10 plus all87 non-task verbs on fullFIT timeline, including annotations absent from high-score proposals',
        targetssl='dense97 plus targetTRAIN timestamp identity contrast across channel/timestamp-masked views; no predictedBG consistency or targetGT',
        utility='dense97 plus candidate dual beneficialBG/harmfulnonBG deletion marginal regression; score benefit minus harm latent, not BCE pBG'),
    encoder='raw2304->64 GELU; residual depthwise temporalconv dilation1/2/4 with pointwise64 and LayerNorm; train all encoder weights',
    labels='sourceFIT only97 multi-label union occupancy within center cells[i-.5,i+.5]; GT grid uses exact originalloader convention',
    dense_objective='per-class positive/negative separately normalized softoccupancy BCE, averaged over10 or97; no class-frequency tuning',
    task_readout='1-max(sigmoid tasklogit) over16 inside interpolation points and all10tasks; any task protects',
    utility='dense97 + balancedproposalBG BCE +.25 smoothL1(softplus(benefit,harm), log1p(10000*positive fullFIT deletion marginals)); two cost targets distinct; not additive AP',
    targetssl='dense97 +.1 symmetric timestampInfoNCE temperature.2;64 source-independent targettimestamps from two channelmask.1/timemask.1 views; inspired by TS2Vec, not fullhierarchy replication',
    epochs=20,checkpoints=[1,5,10,20],seed=311104,order_seed=123,target_seed=311108,augmentation_seed=311109,
    lr=3e-4,weight_decay=.05,clip=1.,rates=[.005,.01,.02,.05],budget_seconds=900,
    normalization='per-channel sourceFIT timestamp populationmean/std only; apply frozen to all roles; clampstd1e-4',
    train_schedule='one sourceFIT video/update; all fulltimelinecells each epoch; targetssl one cyclic targetTRAIN video/update, wholeencoder sees allits features',
    stop_rule='firstepoch projected20epoch>900sec scheduling pause with complete state, not scientificfailure',
    precision='nativefloat32 inheritedTF32, noAMP/quantization; originaldetector remains frozen, noforward/retrain',
    candidate_pool='originalpostNMSscore>=.3; all originalscore/class/endpoints and low-score replacements unchanged',
    evaluation='all8 heads frozen before targetvalCPU;fullfit/heldout/sourceval/targetval10classAP .1:.1:.5, all4fixedbudgets, matchedconfidence, per-videoAUC and BG-only/nonBG-only actualsubset effects',
    caveat='S1/S2 historicaldevelopment sets, head-onlysourceholdout, not independent untouched test or fullsixdirections; no targetTRAIN labels opened',
    target_train_GT_read=False,target_val_selection=False,
    references=['https://arxiv.org/abs/2106.10466','https://raw.githubusercontent.com/zhihanyue/ts2vec/main/models/losses.py'])


def seed():
    import torch
    random.seed(311104);np.random.seed(311104);torch.manual_seed(311104)
    torch.cuda.set_device(0);torch.cuda.manual_seed(311104)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def raw_load(record):
    import torch
    p=Path(record['raw_path']);st=p.stat();assert st.st_size==record['raw_size'] and st.st_mtime_ns==record['raw_mtime_ns']
    return torch.from_numpy(np.load(p)['feats'].astype(np.float32)[::record['downsample']].copy())


def prepare(root):
    import torch
    from tools import baseline_engine as base
    from tools.dense_revisit_math import occupancy
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity()))
    assert sha(UDA/'independent_final_audit.json')=='2cd7ca9625ccb80a3b3593636345212a1b37ad878e5e136d4834c1fd0534ab57'
    result={}
    for s in ('S1','S2'):
        started=time.monotonic();folder=root/'inputs'/s;folder.mkdir(parents=True)
        roles=torch.load(DIAG/s/'inputs.pt',map_location='cpu');cfg=read(SOURCE/s/'config.json');ds=cfg['dataset']
        source97=read(UDA/'inputs'/s/'source_fit_97truth.json');fitset=set(source97)
        assert fitset=={r['video'] for r in roles['fit']};bounds={}
        for domain in ('source','validation'):
            for rec in read(LAYER/(s+'_'+domain)/'complete.json')['records']:
                bound=torch.load(rec['path'],map_location='cpu');bounds[rec['video']]=bound
        cost=read(DIAG/s/'fit_costs_and_pairs.json');cursor=0
        for role,rows in roles.items():
            converted=[]
            for old in rows:
                b=bounds[old['video']];vid=old['video'];meta=b['metadata'];path=Path(b['raw_path']);stat=path.stat()
                r={k:old[k] for k in ('video','segments','scores','labels','hard')}
                r.update(grid_boxes=b['boxes'],video_metadata=meta,raw_path=str(path),raw_size=stat.st_size,raw_mtime_ns=stat.st_mtime_ns,
                         raw_sha256=b['raw_sha256'],downsample=ds['downsample_rate'])
                if role=='fit':
                    truth=source97[vid];raw=raw_load(r);r['dense']=occupancy(len(raw),torch.tensor(truth['segments']).reshape(-1,2),torch.tensor(truth['labels']))
                    n=int(old['hard'].sum());r['bg']=old['bg'];r['delta']=torch.tensor(cost['delta'][cursor:cursor+n],dtype=torch.float32)
                    assert np.array_equal(r['bg'].numpy().astype(bool),np.asarray(cost['BG'][cursor:cursor+n]));cursor+=n
                    r['annotations']=truth
                converted.append(r)
            roles[role]=converted
        assert cursor==len(cost['delta'])
        for role in ('target_train','target_val'):
            manifest=read(UDA/'inputs'/s/(role+'_unlabeled.json'));assert all('annotations' not in r for r in manifest['database'].values());rows=[]
            for rec in read(UDA/(s+'_'+role)/'complete.json')['records']:
                assert sha(rec['path'])==rec['sha256'];old=torch.load(rec['path'],map_location='cpu');assert not {'bg','quality','semantic','annotations'}&set(old)
                vid=old['video'];assert vid in manifest['database'];path=Path(ds['feat_folder'])/((ds['file_prefix'] or '')+vid+ds['file_ext']);st=path.stat()
                rows.append({k:old[k] for k in ('video','segments','scores','labels','hard','grid_boxes','video_metadata')}|dict(raw_path=str(path),raw_size=st.st_size,raw_mtime_ns=st.st_mtime_ns,downsample=ds['downsample_rate'],raw_sha256=sha(path)))
            assert set(manifest['database'])=={r['video'] for r in rows};roles[role]=rows
        members={k:[r['video'] for r in rows] for k,rows in roles.items()}
        for i,a in enumerate(members):
            for b in list(members)[i+1:]:assert not set(members[a])&set(members[b])
        total=0;sums=torch.zeros(2304,dtype=torch.float64);squares=sums.clone()
        for r in roles['fit']:
            x=raw_load(r).double();assert x.shape[1]==2304 and torch.isfinite(x).all();total+=len(x);sums+=x.sum(0);squares+=x.square().sum(0)
        mean=sums/total;std=(squares/total-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
        base.checkpoint(folder/'data.pt',dict(roles=roles,mean=mean.float(),std=std.float()))
        result[s]=dict(data_sha256=sha(folder/'data.pt'),members=members,source_fit_annotations=sum(len(r['annotations']['labels']) for r in roles['fit']),
            source_costs_sha256=sha(DIAG/s/'fit_costs_and_pairs.json'),source97_sha256=sha(UDA/'inputs'/s/'source_fit_97truth.json'),normalization_timestamps=total,seconds=time.monotonic()-started)
        print('DENSE_PREPARED',s,{k:len(v) for k,v in members.items()},result[s]['seconds'],flush=True)
    save(root/'prepare_complete.json',dict(passed=True,results=result,target_train_GT_read=False,previous_audit_sha256=sha(UDA/'independent_final_audit.json')))


def capture(model,opt,order,target_order,aug,history,step,entry):
    import torch
    return dict(model=deepcopy(model.state_dict()),optimizer=deepcopy(opt.state_dict()),order_rng=order.get_state(),target_order=target_order.state(),augmentation_rng=aug.get_state(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),history=deepcopy(history),step=step,spec=SPEC,
        inputs_sha256=read(entry.parent/'prepare_complete.json')['results'][entry.name.split('_')[0]]['data_sha256'])


def update(model,opt,r,x,mode,target,aug):
    import torch
    from tools.dense_revisit_math import dense_loss,utility_loss,temporal_ssl
    from tools.hard_background_modules import balanced_bg
    opt.zero_grad(set_to_none=True);logits,z=model(x);classes=10 if mode=='dense10' else 97
    dense=dense_loss(logits[:,:classes],r['dense'].to(x.device)[:,:classes]);loss=dense;terms=dict(dense=float(dense.detach()))
    if mode=='utility':
        hard=r['hard'];bg,_,u=model.candidate(logits,z,r['grid_boxes'][hard].to(x.device),r['labels'][hard].to(x.device),r['scores'][hard].to(x.device))
        if len(bg):
            ordinary=balanced_bg(torch.logit(bg.clamp(1e-5,1-1e-5)),r['bg'].to(x.device));cost=utility_loss(u,r['bg'].to(x.device),r['delta'].to(x.device));loss=loss+ordinary+.25*cost
            terms.update(candidate_BG=float(ordinary.detach()),utility=float(cost.detach()))
    if mode=='targetssl':
        # 完整targettrain视频前向；独立生成器保存精确增强随机流。
        views=[]
        for _ in range(2):
            cm=(torch.rand((1,target.shape[1]),device=x.device,generator=aug)>.1).float()/.9
            tm=(torch.rand((len(target),1),device=x.device,generator=aug)>.1).float();views.append(target*cm*tm)
        _,a=model(views[0]);_,b=model(views[1]);ids=torch.linspace(0,len(target)-1,min(64,len(target)),device=x.device).long()
        ssl=temporal_ssl(a[ids],b[ids]);loss=loss+.1*ssl;terms['SSL']=float(ssl.detach())
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
    return dict(loss=float(loss.detach()),grad=float(norm),**terms)


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.dense_revisit_math import DenseEvidence
    from tools.run_uda_background import TargetStream
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();s,mode=entry.name.split('_');started=time.monotonic();fp=entry.parent/'inputs'/s/'data.pt'
    assert sha(fp)==read(entry.parent/'prepare_complete.json')['results'][s]['data_sha256'];data=torch.load(fp,map_location='cpu');roles=data['roles'];mean=data['mean'];std=data['std']
    fit=roles['fit'];target=roles['target_train'];raw={}
    for r in fit+(target if mode=='targetssl' else []):raw[r['video']]=((raw_load(r)-mean)/std).pin_memory()
    needed=sum(x.numel()*x.element_size() for x in raw.values());resident=needed+4096*2**20<torch.cuda.mem_get_info()[0]
    if resident:raw={k:v.cuda() for k,v in raw.items()}
    def get(r):return raw[r['video']].to('cuda',non_blocking=True)
    def fresh():
        m=DenseEvidence().cuda();o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05);return m,o
    def restore(st,m,o,order,stream,aug):
        m.load_state_dict(st['model']);o.load_state_dict(st['optimizer']);order.set_state(st['order_rng']);stream.restore(st['target_order']);aug.set_state(st['augmentation_rng'])
        torch.set_rng_state(st['cpu_rng']);torch.cuda.set_rng_state(st['cuda_rng'],0);random.setstate(st['python_rng']);np.random.set_state(st['numpy_rng'])
    model,opt=fresh();order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));aug=torch.Generator(device='cuda').manual_seed(311109)
    r=fit[0];x=get(r);tx=get(target[0]) if mode=='targetssl' else None;before=base.tensor_digest(model.state_dict());update(model,opt,r,x,mode,tx,aug)
    assert before!=base.tensor_digest(model.state_dict())
    for name in ('project','blocks','activity'):assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if n.startswith(name))
    if mode=='utility':assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.utility.parameters())
    st=capture(model,opt,order,stream,aug,[],1,entry);base.checkpoint(entry/'kernel.pt',st);st=torch.load(entry/'kernel.pt',map_location='cpu')
    other,oo=fresh();restore(st,other,oo,order,stream,aug)
    def draw():return torch.randperm(len(fit),generator=order),stream.draw(1),torch.rand(4,device='cuda',generator=aug),torch.rand(4,device='cuda'),torch.rand(4)
    restore(st,model,opt,order,stream,aug);a=draw();restore(st,other,oo,order,stream,aug);b=draw();assert all(torch.equal(i,j) for i,j in zip(a,b))
    restore(st,model,opt,order,stream,aug);update(model,opt,r,x,mode,tx,aug);after=base.tensor_digest(model.state_dict());osd=base.tensor_digest(opt.state_dict())
    restore(st,other,oo,order,stream,aug);update(other,oo,r,x,mode,tx,aug)
    assert after==base.tensor_digest(other.state_dict()) and osd==base.tensor_digest(oo.state_dict())
    # 存储/搬运优化仅复用完全相同的归一化输入，不变数学；实测同一保存头的loss/grad/update。
    restore(st,model,opt,order,stream,aug);update(model,opt,r,x,mode,tx,aug);digest=base.tensor_digest(model.state_dict())
    restore(st,other,oo,order,stream,aug);copied=((raw_load(r)-mean)/std).cuda();assert torch.equal(x,copied);update(other,oo,r,copied,mode,tx,aug);assert digest==base.tensor_digest(other.state_dict())
    benchmark={};cpu_input=((raw_load(r)-mean)/std).pin_memory()
    with torch.no_grad():
        for label,move in [('resident',False),('cachedCPU_transfer',True)]:
            for _ in range(3):model(cpu_input.cuda(non_blocking=True) if move else x)
            torch.cuda.synchronize();tick=time.monotonic()
            for _ in range(20):model(cpu_input.cuda(non_blocking=True) if move else x)
            torch.cuda.synchronize();benchmark[label]=time.monotonic()-tick
    save(entry/'storage_benchmark.json',dict(seconds20forwards=benchmark,CPU_transfer_to_resident_ratio=benchmark['cachedCPU_transfer']/benchmark['resident'],scope='forwardstorage_microbenchmark, not endtoendtraining speedup',input_numerics_update_exact=True))
    save(entry/'kernel_complete.json',dict(passed=True,real_encoder_update=True,complete_model_optimizer_RNG_next_step_exact=True,draws_exact=True,resident_vs_CPUtransfer_update_exact=True,
        kernel_sha256=sha(entry/'kernel.pt'),guard_sha256=sha(guard),storage_benchmark_sha256=sha(entry/'storage_benchmark.json')))
    del model,opt,other,oo,st,x,copied;torch.cuda.empty_cache();seed();model,opt=fresh();order=torch.Generator().manual_seed(123);stream=TargetStream(len(target));aug=torch.Generator(device='cuda').manual_seed(311109);history=[];step=0
    initial=base.tensor_digest(model.state_dict())
    for epoch in range(1,21):
        tick=time.monotonic();records=[]
        for i in torch.randperm(len(fit),generator=order).tolist():
            r=fit[i];ti=int(stream.draw(1)[0]);tx=get(target[ti]) if mode=='targetssl' else None
            records.append(update(model,opt,r,get(r),mode,tx,aug));step+=1
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,seconds=seconds,steps=len(records),**{k:float(np.mean([v[k] for v in records])) for k in records[0]}))
        base.checkpoint(entry/'latest.pt',capture(model,opt,order,stream,aug,history,step,entry))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('DENSE_EPOCH',entry.name,epoch,seconds,history[-1]['loss'],flush=True)
        if epoch==1:
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=seconds*20,setup_seconds=time.monotonic()-started-seconds,normalized_input_bytes=needed,GPU_resident=resident,limit=900))
            if seconds*20>900:save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,checkpoint_sha256=sha(entry/'latest.pt')));return
    model.eval();digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout(m):
        out={}
        with torch.no_grad():
            for role in ('fit','heldout','validation','target_val'):
                out[role]=[]
                for r in roles[role]:
                    x=get(r) if r['video'] in raw else ((raw_load(r)-mean)/std).cuda();logits,z=m(x);h=r['hard']
                    bg,score,_=m.candidate(logits,z,r['grid_boxes'][h].cuda(),r['labels'][h].cuda(),r['scores'][h].cuda());p=torch.zeros(len(h));p[h]=(score if mode=='utility' else bg).cpu()
                    from tools.dense_revisit_math import sample_sequence
                    _,valid=sample_sequence(z,r['grid_boxes'][h].cuda());eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=(valid.sum(1)>=4).cpu()
                    out[role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p,eligible=eligible))
        return out
    out=readout(model);base.checkpoint(entry/'predictions.pt',out)
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    other=DenseEvidence().cuda();other.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['model']);other.eval();replay=readout(other)
    for role in out:
        for a,b in zip(out[role],replay[role]):assert torch.equal(a['pbg'],b['pbg']) and torch.equal(a['eligible'],b['eligible'])
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),initial_digest=initial,final_digest=digest,
        predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),kernel_sha256=sha(entry/'kernel_complete.json'),
        parameters=sum(p.numel() for p in model.parameters()),saved_predictions_GPU_replay_bitwise_equal=True,readout_parameters_RNG_unchanged=True,
        target_train_seen=int((stream.seen>0).sum()) if mode=='targetssl' else 0,target_train_GT_read=False,seconds=time.monotonic()-started))


def choose(frame,eligible,rate,confidence=False):
    selected=[]
    for c in range(10):
        pool=np.flatnonzero((frame.label.to_numpy()==c)&(frame.score.to_numpy()>=.3));n=int(np.floor(rate*len(pool)));pool=pool[eligible[pool]]
        key=frame.score.to_numpy()[pool] if confidence else -frame.pbg.to_numpy()[pool]
        selected.extend(pool[np.lexsort((pool,key))[:n]].tolist())
    return np.asarray(sorted(selected),dtype=np.int64)


def evaluate(root,shift):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    assert all((root/t/'gpu_complete.json').exists() for t in TASKS)
    data=torch.load(root/'inputs'/shift/'data.pt',map_location='cpu');roles=data['roles'];outputs={m:torch.load(root/(shift+'_'+m)/'predictions.pt',map_location='cpu') for m in MODES}
    gt=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);results={}
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m in MODES:
            rows=outputs[m][role]
            for a,b in zip(rows,roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][~b['hard']]==0).all()
            frames[m]=prediction_frame(rows);v=torch.cat([r['eligible'] for r in rows]).numpy()
            if eligible is None:eligible=v
            else:assert np.array_equal(eligible,v)
        frame=frames[MODES[0]];bg=background_mask(truth[role],frame);engines={};refs={};pos={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;engines[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[ids]);refs[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(engines[c].evaluate(np.ones(len(ids),bool))['ap'])
        def ap(remove):
            keep=np.ones(len(frame),bool);keep[remove]=False;a=[engines[c].evaluate(keep[pos[c]])['ap'] for c in range(10)];b=[refs[c].ap(keep[pos[c]]) for c in range(10)]
            assert np.allclose(a,b,atol=1e-7,rtol=0);return dict(per_class_AP=np.asarray(a).tolist(),delta_pp=float(100*(np.mean(a)-np.mean(baseline))))
        ap(np.array([],dtype=int));results[role]={}
        for m in (*MODES,'confidence'):
            f=frame if m=='confidence' else frames[m];value=1-f.score.to_numpy() if m=='confidence' else f.pbg.to_numpy();auc=[];budgets=[]
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:auc.append(a)
            for rate in SPEC['rates']:
                selected=choose(f,eligible,rate,m=='confidence');assert not (f.iloc[selected].score.to_numpy()<.3).any();a=ap(selected);b=ap(selected[bg[selected]]);c=ap(selected[~bg[selected]])
                budgets.append(dict(rate=rate,selected=selected.tolist(),BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum()),BG_only_delta_pp=b['delta_pp'],nonBG_only_delta_pp=c['delta_pp'],**a))
            results[role][m]=dict(baseline=float(np.mean(baseline)),original_per_class_AP=np.asarray(baseline).tolist(),video_macro_AUC=float(np.mean(auc)) if auc else None,AUC_valid_videos=len(auc),budgets=budgets)
        print('DENSE_EVALUATED',shift,role,flush=True)
    original=read(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/shift/'evaluation'/'target_val.json')['map_0.1_to_0.5'];assert abs(results['target_val']['dense10']['baseline']-original)<1e-7
    save(root/(shift+'_evaluation.json'),dict(passed=True,results=results,all_original_arrays_exact=True,all_AP_independent_NumPy_equal=True,target_train_GT_read=False,target_val_used_for_selection=False,subset_effects_not_additive=True,
        input_sha256=sha(root/'inputs'/shift/'data.pt'),GPU_sha256={m:sha(root/(shift+'_'+m)/'gpu_complete.json') for m in MODES}))


def run(root):
    from tools.baseline_gpu import snapshot
    prepare(root);active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        for stage,tasks in [('train',TASKS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' scheduling pause')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        observed=snapshot(g);lease=entry/'lease_train.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');cmd=[sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage];p=subprocess.Popen(cmd,env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),completed=done,pending=pending,active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        save(root/'final.json',dict(passed=True,spec=SPEC,evaluation_sha256={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')},all8GPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['train','evaluate']);a=p.parse_args()
    if a.worker=='train':train(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
