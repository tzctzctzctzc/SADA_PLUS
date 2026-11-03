"""冻结检测器的逐层信号诊断；只在源fit训练固定线性probe，不选阈值/epoch。"""
import argparse
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
from tools.layer_signal_math import STAGES,MODES,FAMILIES,raw_projection,duration_levels,pool_signals,conditional_auc,video_geometry
from tools.highscore_confusion_diagnosis import OLD,SOURCE,ORIGIN
DIAG=Path('/vepfs/vlms/highscore_confusion_diag_36c0916_1103')
EXTRACT=('S1_source','S1_validation','S2_source','S2_validation')
PROBES=('S1_early','S1_temporal','S2_early','S2_temporal')
SPEC=dict(schema='highscore-layer-signal-source-v1',source_only=True,target_read=False,
    shifts=['S1','S2'],stages=STAGES,modes=MODES,source_split='unchanged head-holdout-v1',
    raw='fixed seed311106 Gaussian2304to512 projection; not lossless; no motion-channel semantics asserted',
    pooling='same original4/16/4 interpolation, valid inner16 mean; raw grid and feature level boxes/stride, offset0',
    scale='predicted half-duration distance to original regression ranges, finest tie; heuristic not actual NMS provenance',
    centered='subtract unlabeled per-video valid feature mean; removes action-distribution as well as scene information',
    probe='visual512 linear1 alone; metadata12 linear1 separately; no quality objective; zero initialization',
    standardization='sourceFIT per-channel mean and population std clamp1e-5, apply unchanged to heldout/validation',
    objective='balanced BG BCE, no threshold/epoch/architecture selection',epochs=20,checkpoints=[1,5,10,20],batch=256,
    seed=311104,order_seed=123,lr=3e-4,weight_decay=.05,clip=1.,rates=[.005,.01,.02,.05],
    contrasts=['raw_RP512 versus FPN0','embedding to SGP to FPN0','duration_scale versus FPN0','centered versus original','visual versus metadata'],
    budget_seconds_per_worker=900,precision='inherit originalfloat32/TF32 environment, no AMP/quantization',
    caveat='readout capacity diagnostic, not information-theoretic proof; head-only holdout; no target/generalization claim')


def seed():
    import torch
    random.seed(311104);np.random.seed(311104);torch.manual_seed(311104)
    torch.cuda.set_device(0);torch.cuda.manual_seed(311104)
    assert torch.cuda.initial_seed()==311104


def extract(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_foreground_calibration_oracle import load_detector,native_features
    guard=first_cuda_guard(entry);seed();shift,role=entry.name.split('_');began=time.monotonic()
    cfg=read(SOURCE/shift/'config.json');cost=read(ORIGIN/shift/'costs.json');index=read(SOURCE/shift/'cache_index.json')
    records=index['records'] if role=='source' else cost['validation'];assert records
    cp=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/shift/'best.pt'
    expected=read(SOURCE/'plan.json')['jobs'][shift]['checkpoint_sha256'];assert sha(cp)==expected
    model=load_detector(SOURCE/shift);digest=base.tensor_digest(model.state_dict());assert digest==index['detector_digest']
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);projection=raw_projection().cuda();done=[]
    base.checkpoint(entry/'projection.pt',projection.cpu());(entry/'videos').mkdir();first=True
    for rec in records:
        started=time.monotonic();assert sha(rec['path'])==rec['sha256'];old=torch.load(rec['path'],map_location='cpu')
        meta=old['metadata'];vid=rec['video'];assert vid==meta['video_id'];ds=cfg['dataset']
        rawpath=Path(ds['feat_folder'])/((ds['file_prefix'] or '')+vid+ds['file_ext'])
        raw=torch.from_numpy(np.load(rawpath)['feats'].astype(np.float32)[::ds['downsample_rate']].T.copy())
        assert raw.shape[0]==2304 and raw.shape[1]==meta['video_length']
        video=[dict(feats=raw.cuda(),segments=torch.empty(0,2,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=meta)]
        with torch.no_grad():
            x,mask=model.preprocessing(video,mode='test');x=model.dropout(x)
            backbone,masks,embedding=model.backbone_source(x,mask);fpn,fmasks=model.neck_source(backbone,masks)
            assert torch.equal(fpn[0][0].cpu(),old['feature']) and torch.equal(fmasks[0][0,0].bool().cpu(),old['valid'])
            if first:
                nf,nm,_=native_features(model,video,True)
                assert torch.equal(nf,fpn[0][0]) and torch.equal(nm,fmasks[0][0,0].bool());first=False
            hard=old['scores']>=.3;boxes=old['boxes'][hard].cuda()
            # raw先在原采样坐标池化，再固定投影；无需对整段2304维重复大矩阵计算。
            rawmean,rawglobal,rawstats=pool_signals(x[0],mask[0,0].bool(),boxes)
            means={'raw_RP512':rawmean@projection};globals={'raw_RP512':rawglobal@projection};stats={'raw_RP512':rawstats}
            for name,feature,valid in [('embedding',embedding[0],mask[0,0].bool()),('SGP',backbone[0][0],masks[0][0,0].bool()),('FPN0',fpn[0][0],fmasks[0][0,0].bool())]:
                means[name],globals[name],stats[name]=pool_signals(feature,valid,boxes)
            levels=duration_levels(boxes,model.reg_range);allmeans=[];allglobals=[];allstats=[]
            for level,(f,m,stride) in enumerate(zip(fpn,fmasks,model.fpn_strides)):
                a,b,c=pool_signals(f[0],m[0,0].bool(),boxes,stride);allmeans.append(a);allglobals.append(b);allstats.append(c)
            stack=torch.stack(allmeans,1);gs=torch.stack(allglobals,0);st=torch.stack(allstats,1);ix=torch.arange(len(boxes),device='cuda')
            means['duration_scale']=stack[ix,levels];globals['duration_scale']=gs[levels];stats['duration_scale']=st[ix,levels]
            reps={}
            for name in STAGES:
                reps[name]=means[name].cpu();reps[name+'_centered']=(means[name]-globals[name]).cpu()
            reps['metadata']=torch.cat((torch.nn.functional.one_hot(old['labels'][hard].long(),10).float(),torch.logit(old['scores'][hard].clamp(1e-5,1-1e-5))[:,None]/5,torch.log1p((old['boxes'][hard,1]-old['boxes'][hard,0])/old['feature'].shape[-1])[:,None]),1)
        out=entry/'videos'/(vid+'.pt')
        base.checkpoint(out,dict(video=vid,representations=reps,statistics={k:v.cpu() for k,v in stats.items()},all_FPN_inner=stack.cpu(),levels=levels.cpu(),boxes=old['boxes'],metadata=meta,
            scores=old['scores'],labels=old['labels'],segments=old['segments'],hard=hard,source_cache=rec,raw_path=str(rawpath),raw_sha256=sha(rawpath),raw_shape=list(raw.shape)))
        done.append(dict(video=vid,path=str(out),sha256=sha(out),seconds=time.monotonic()-started));print('EXTRACTED',entry.name,vid,len(boxes),done[-1]['seconds'],flush=True)
        save(entry/'progress.json',dict(done=len(done),total=len(records),elapsed=time.monotonic()-began,records=done))
        if len(done)==3:
            projected=(time.monotonic()-began)*len(records)/3
            save(entry/'throughput.json',dict(first3_end_to_end_seconds=time.monotonic()-began,projected_seconds=projected,limit=900))
            if projected>900:
                save(entry/'budget_pause.json',dict(scientific_failure=False,reason='user short-experiment time budget',automatic_restart_authorized=False,completed_records=done));return
    assert digest==base.tensor_digest(model.state_dict()) and all(p.grad is None for p in model.parameters())
    assert torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'complete.json',dict(passed=True,records=done,checkpoint_sha256=expected,detector_digest=digest,guard=guard,guard_sha256=sha(guard),
        projection_sha256=sha(entry/'projection.pt'),all_old_FPN0_bitwise_equal=True,first_native_forward_equal=True,parameters_and_RNG_unchanged=True,seconds=time.monotonic()-began))


def roles_for(root,shift):
    import torch
    split=read(OLD/shift/'split.json');roles={k:[] for k in ('fit','heldout','validation')}
    original=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');lookup={r['video']:r for rows in original.values() for r in rows}
    for domain in ('source','validation'):
        complete=read(root/(shift+'_'+domain)/'complete.json');assert complete['passed']
        for rec in complete['records']:
            assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');v=r['video'];old=lookup[v]
            role='validation' if domain=='validation' else ('fit' if v in split['fit'] else 'heldout')
            assert role=='validation' or v in split[role]
            for k in ('scores','labels','hard'):assert torch.equal(old[k],r[k])
            assert torch.equal(old['segments'],r['segments'] if role=='validation' else r['boxes'])
            r['segments']=old['segments'];r['bg']=old.get('bg');roles[role].append(r)
    for role in roles:roles[role].sort(key=lambda r:r['video'])
    return roles


def newheads(modes,dimensions):
    import torch
    heads=torch.nn.ModuleDict({m:torch.nn.Linear(dimensions[m],1) for m in modes}).cuda()
    for h in heads.values():torch.nn.init.zeros_(h.weight);torch.nn.init.zeros_(h.bias)
    return heads


def probe(entry):
    import torch
    from tools import baseline_engine as base
    from tools.hard_background_modules import balanced_bg
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);seed();shift,family=entry.name.split('_');modes=FAMILIES[family];began=time.monotonic();roles=roles_for(entry.parent,shift)
    normal={};data={};dimensions={};bg=torch.cat([r['bg'] for r in roles['fit']]).cuda()
    for m in modes:
        fit=torch.cat([r['representations'][m] for r in roles['fit']]);mean=fit.mean(0);std=fit.std(0,unbiased=False).clamp_min(1e-5)
        normal[m]=dict(mean=mean,std=std);dimensions[m]=fit.shape[1];data[m]=((fit-mean)/std).cuda()
    def step(heads,opts,ids):
        losses={}
        for m in modes:
            opts[m].zero_grad(set_to_none=True);loss=balanced_bg(heads[m](data[m][ids])[:,0],bg[ids]);assert torch.isfinite(loss)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(heads[m].parameters(),1.);assert torch.isfinite(norm);opts[m].step();losses[m]=float(loss.detach())
        return losses
    def snapshot(heads,opts,order,history):
        return dict(heads=heads.state_dict(),optimizers={m:o.state_dict() for m,o in opts.items()},normalization=normal,dimensions=dimensions,modes=modes,order_rng=order.get_state(),history=history,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),spec=SPEC,
            extraction_SHA={d:sha(entry.parent/(shift+'_'+d)/'complete.json') for d in ('source','validation')})
    heads=newheads(modes,dimensions);opts={m:torch.optim.AdamW(heads[m].parameters(),lr=3e-4,weight_decay=.05) for m in modes};order=torch.Generator().manual_seed(123)
    initial=base.tensor_digest(heads.state_dict());ids=torch.arange(min(128,len(bg)),device='cuda');step(heads,opts,ids)
    assert base.tensor_digest(heads.state_dict())!=initial
    base.checkpoint(entry/'kernel.pt',snapshot(heads,opts,order,[]));saved=torch.load(entry/'kernel.pt',map_location='cpu')
    other=newheads(modes,dimensions);other.load_state_dict(saved['heads']);oo={m:torch.optim.AdamW(other[m].parameters(),lr=3e-4,weight_decay=.05) for m in modes}
    for m in modes:oo[m].load_state_dict(saved['optimizers'][m])
    torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0);step(heads,opts,ids)
    torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0);step(other,oo,ids)
    assert base.tensor_digest(heads.state_dict())==base.tensor_digest(other.state_dict())
    assert base.tensor_digest({m:o.state_dict() for m,o in opts.items()})==base.tensor_digest({m:o.state_dict() for m,o in oo.items()})
    save(entry/'kernel_complete.json',dict(passed=True,real_update=True,disk_optimizer_nextstep_bitwise_equal=True,explicit_cuda_seed=311104,kernel_sha256=sha(entry/'kernel.pt')))
    seed();heads=newheads(modes,dimensions);opts={m:torch.optim.AdamW(heads[m].parameters(),lr=3e-4,weight_decay=.05) for m in modes};order=torch.Generator().manual_seed(123);history=[]
    for epoch in range(1,21):
        tick=time.monotonic();losses=[];permutation=torch.randperm(len(bg),generator=order)
        for ids in permutation.split(256):losses.append(step(heads,opts,ids.cuda()))
        torch.cuda.synchronize();seconds=time.monotonic()-tick;history.append(dict(epoch=epoch,seconds=seconds,steps=len(losses),loss={m:float(np.mean([l[m] for l in losses])) for m in modes}))
        base.checkpoint(entry/'latest.pt',snapshot(heads,opts,order,history))
        if epoch in SPEC['checkpoints']:os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('EPOCH',entry.name,epoch,seconds,flush=True)
        if epoch==1 and seconds*20>900:
            save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,projected=seconds*20));return
    heads.eval();digest=base.tensor_digest(heads.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    def readout(hs):
        out={}
        with torch.no_grad():
            for role,rows in roles.items():
                out[role]={m:[] for m in modes}
                for r in rows:
                    for m in modes:
                        x=((r['representations'][m]-normal[m]['mean'])/normal[m]['std']).cuda();p=torch.zeros(len(r['scores']))
                        p[r['hard']]=torch.cat([hs[m](part)[:,0].sigmoid().cpu() for part in x.split(256)]) if len(x) else torch.empty(0)
                        out[role][m].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
        return out
    out=readout(heads);base.checkpoint(entry/'predictions.pt',out)
    assert digest==base.tensor_digest(heads.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    replay=newheads(modes,dimensions);replay.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['heads']);again=readout(replay)
    for role in out:
        for m in modes:
            for a,b in zip(out[role][m],again[role][m]):assert torch.equal(a['pbg'],b['pbg'])
    save(entry/'gpu_complete.json',dict(passed=True,modes=modes,parameters={m:sum(p.numel() for p in heads[m].parameters()) for m in modes},guard=guard,guard_sha256=sha(guard),
        kernel_sha256=sha(entry/'kernel_complete.json'),predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),final_digest=digest,
        all_predictions_disk_GPU_bitwise_equal=True,readout_parameters_RNG_unchanged=True,seconds=time.monotonic()-began))


def evaluate(entry):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.audit_background_ap_cost import IndependentAP,selected_indices
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    shift,family=entry.name.split('_');outputs=torch.load(entry/'predictions.pt',map_location='cpu');roles=roles_for(entry.parent,shift)
    gt=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json');truth={}
    for role in ('fit','heldout'):
        truth[role]=pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label'])
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    edges=read(DIAG/shift/'fit_costs_and_pairs.json')['pairing']['fit_only_edges'];report={}
    for role,methods in outputs.items():
        first=next(iter(methods.values()));frame=prediction_frame(first);bg=background_mask(truth[role],frame);hard=frame.score.to_numpy()>=.3
        engines={};refs={};positions={};baseline=[]
        for c in range(10):
            idx=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[idx]);refs[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[idx].reset_index(drop=True))
            a=engines[c].evaluate(np.ones(len(idx),bool))['ap'];assert np.allclose(a,refs[c].ap(np.ones(len(idx),bool)),atol=1e-7,rtol=0);baseline.append(a)
        videos=frame['video-id'].to_numpy()[hard];labels=frame.label.to_numpy()[hard];scores=frame.score.to_numpy()[hard]
        boxes=torch.cat([r['boxes'][r['hard']] for r in roles[role]]).numpy();lengths=boxes[:,1]-boxes[:,0];report[role]={}
        for mode in (*methods,'confidence'):
            rows=methods[mode] if mode!='confidence' else first;f=prediction_frame(rows);assert (f.score.to_numpy()==frame.score.to_numpy()).all()
            value=1-f.score.to_numpy() if mode=='confidence' else f.pbg.to_numpy();aucs=[]
            for _,idx in f.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[hard[idx]];a=independent_auc(bg[idx],value[idx])
                if a is not None:aucs.append(a)
            budgets=[]
            for rate in SPEC['rates']:
                selected=selected_indices(f,rate,mode=='confidence');keep=np.ones(len(f),bool);keep[selected]=False
                ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];reference=[refs[c].ap(keep[positions[c]]).tolist() for c in range(10)]
                assert np.allclose(ap,reference,atol=1e-7,rtol=0)
                budgets.append(dict(rate=rate,delta_pp=float(100*(np.mean(ap)-np.mean(baseline))),selected=selected.tolist(),BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum()),per_class_AP=ap))
            report[role][mode]=dict(baseline=float(np.mean(baseline)),video_macro_AUC=float(np.mean(aucs)) if aucs else None,AUC_valid_videos=len(aucs),
                conditional=conditional_auc(bg[hard],value[hard],labels,scores,lengths,videos,edges),budgets=budgets)
        geometry={};stats={}
        for stage in FAMILIES[family]:
            if stage=='metadata':continue
            rep=torch.cat([r['representations'][stage] for r in roles[role]]).numpy();geometry[stage]=video_geometry(rep,labels,scores,lengths,videos,edges,boxes)
        for stage in STAGES:
            a=torch.cat([r['statistics'][stage] for r in roles[role]]).numpy()
            stats[stage]={str(k):dict(BG_mean=a[bg[hard]].mean(0).tolist() if bg[hard].any() else None,nonBG_mean=a[~bg[hard]].mean(0).tolist() if (~bg[hard]).any() else None) for k in ['norm_change_context_coverage']}
        save(entry/(role+'_geometry.json'),dict(geometry=geometry,descriptive_statistics=stats,not_causal=True));print('EVALUATED',entry.name,role,flush=True)
    save(entry/'complete.json',dict(passed=True,source_only=True,gpu_complete_sha256=sha(entry/'gpu_complete.json'),results=report))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),diagnostic_sha256=sha(DIAG/'complete.json')));active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        for stage,tasks in [('extract',EXTRACT),('probe',PROBES),('evaluate',PROBES)]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' scheduling pause not scientific failure')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in (0,1,2,3) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        observed=snapshot(g);lease=entry/('lease_'+stage+'.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                        env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),completed=done,pending=pending,active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        save(root/'final.json',dict(passed=True,source_only=True,spec=SPEC,complete_sha256={t:sha(root/t/'complete.json') for t in PROBES}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['extract','probe','evaluate']);a=p.parse_args()
    if a.worker:globals()[a.worker](a.entry)
    else:run(a.root)
