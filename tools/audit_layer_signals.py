"""独立原生forward hook逐层重放、保存头重放、全固定操作点AP核验。"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_layer_signals import EXTRACT,PROBES,SPEC,STAGES,FAMILIES,OLD,SOURCE,DIAG


def inner(feature,mask,boxes,stride=1):
    """独立写出16个区间内中心坐标；不调用生产pool_signals。"""
    import torch
    b=boxes/stride;x=b[:,0,None]+(b[:,1]-b[:,0])[:,None]*(torch.arange(16,device='cuda',dtype=feature.dtype)+.5)/16
    lo=x.floor().long();hi=x.ceil().long();a=lo.clamp(0,feature.shape[-1]-1);z=hi.clamp(0,feature.shape[-1]-1);w=x-lo
    valid=(x>=0)&(x<=feature.shape[-1]-1)&mask[a]&mask[z]
    values=(feature[:,a]*(1-w)[None]+feature[:,z]*w[None]).permute(1,2,0)
    return (values*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)


def guardcheck(report,owner,head):
    guard=read(report['guard']);lease=read(guard['lease'])
    assert sha(report['guard'])==report['guard_sha256'] and sha(guard['lease'])==guard['lease_sha256']
    assert guard['owner_pid']==lease['owner_pid']==owner and lease['commit']==head
    assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['index']==lease['index'] and guard['observed']['free_mib']>=12288


def extract_native(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);shift=entry.name.split('_')[0];complete=read(entry/'complete.json');model=load_detector(SOURCE/shift)
    assert base.tensor_digest(model.state_dict())==complete['detector_digest'];captured={}
    handles=[model.backbone_source.register_forward_hook(lambda m,args,out:captured.update(raw=args[0],mask=args[1],backbone=out)),
             model.neck_source.register_forward_hook(lambda m,args,out:captured.update(fpn=out))]
    projection=torch.load(entry/'projection.pt',map_location='cpu');g=torch.Generator().manual_seed(311106)
    assert torch.equal(projection,torch.randn(2304,512,generator=g)/np.sqrt(512));projection=projection.cuda()
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);digest=base.tensor_digest(model.state_dict());checked=[]
    with torch.no_grad():
        for rec in complete['records']:
            assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');old=torch.load(r['source_cache']['path'],map_location='cpu')
            assert sha(r['raw_path'])==r['raw_sha256'];cfg=read(SOURCE/shift/'config.json')['dataset']
            raw=torch.from_numpy(np.load(r['raw_path'])['feats'].astype(np.float32)[::cfg['downsample_rate']].T.copy())
            video=[dict(feats=raw.cuda(),segments=torch.empty(0,2,device='cuda'),labels=torch.empty(0,dtype=torch.long,device='cuda'),metadata=r['metadata'])]
            # 调用完整正式forward，从hook读取层输出；不同于生产手工路径。
            model(video_list_src=video,video_list_tgt=video,mode='test')
            bf,bm,embedding=captured['backbone'];fpn,fm=captured['fpn'];boxes=r['boxes'][r['hard']].cuda();n=len(boxes)
            assert torch.equal(fpn[0][0].cpu(),old['feature']) and torch.equal(fm[0][0,0].bool().cpu(),old['valid'])
            stages={'raw_RP512':(captured['raw'][0],captured['mask'][0,0].bool()),'embedding':(embedding[0],captured['mask'][0,0].bool()),'SGP':(bf[0][0],bm[0][0,0].bool()),'FPN0':(fpn[0][0],fm[0][0,0].bool())}
            for name,(f,v) in stages.items():
                a=inner(f,v,boxes);b=f[:,v].mean(1)
                if name=='raw_RP512':a=a@projection;b=b@projection
                assert torch.equal(a.cpu(),r['representations'][name]) and torch.equal((a-b).cpu(),r['representations'][name+'_centered'])
            half=(boxes[:,1]-boxes[:,0]).cpu().numpy()/2;ranges=np.asarray(model.reg_range);dist=np.maximum(ranges[None,:,0]-half[:,None],0)+np.maximum(half[:,None]-ranges[None,:,1],0);levels=dist.argmin(1)
            assert np.array_equal(levels,r['levels'].numpy());means=[];globals=[]
            for f,m,stride in zip(fpn,fm,model.fpn_strides):means.append(inner(f[0],m[0,0].bool(),boxes,stride));globals.append(f[0][:,m[0,0].bool()].mean(1))
            stack=torch.stack(means,1);globalstack=torch.stack(globals);ids=torch.arange(n,device='cuda');lev=torch.from_numpy(levels).cuda()
            assert torch.equal(stack.cpu(),r['all_FPN_inner']) and torch.equal(stack[ids,lev].cpu(),r['representations']['duration_scale'])
            assert torch.equal((stack[ids,lev]-globalstack[lev]).cpu(),r['representations']['duration_scale_centered'])
            # 非视觉元数据独立重算。
            label=r['labels'][r['hard']].numpy();score=r['scores'][r['hard']].numpy();b=r['boxes'][r['hard']].numpy();meta=np.zeros((n,12),np.float32);meta[np.arange(n),label]=1
            p=np.clip(score,1e-5,1-1e-5);meta[:,10]=np.log(p/(1-p))/5;meta[:,11]=np.log1p((b[:,1]-b[:,0])/old['feature'].shape[-1])
            assert np.allclose(meta,r['representations']['metadata'],atol=2e-7,rtol=1e-6)
            checked.append(dict(video=r['video'],descriptor_sha256=rec['sha256'],all_visual_modes_bitwise_equal=True))
    for h in handles:h.remove()
    assert digest==base.tensor_digest(model.state_dict()) and all(p.grad is None for p in model.parameters()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'independent_native.json',dict(passed=True,records=checked,full_native_forward_hook_replay=True,all_visual_arrays_bitwise_equal=True,
        source_complete_sha256=sha(entry/'complete.json'),parameters_RNG_unchanged=True,optimizer_steps=0,guard=guard,guard_sha256=sha(guard)))


def probe_native(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);gpu=read(entry/'gpu_complete.json');assert sha(entry/'epoch020.pt')==gpu['checkpoint_sha256'] and sha(entry/'predictions.pt')==gpu['predictions_sha256']
    state=torch.load(entry/'epoch020.pt',map_location='cpu');saved=torch.load(entry/'predictions.pt',map_location='cpu');shift=entry.name.split('_')[0]
    hs=torch.nn.ModuleDict({m:torch.nn.Linear(state['dimensions'][m],1) for m in state['modes']}).cuda();hs.load_state_dict(state['heads']);hs.eval()
    cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);digest=base.tensor_digest(hs.state_dict());checked=0;videos={}
    for domain in ('source','validation'):
        for rec in read(entry.parent/(shift+'_'+domain)/'complete.json')['records']:videos[rec['video']]=rec
    with torch.no_grad():
        for role,methods in saved.items():
            for m,rows in methods.items():
                for row in rows:
                    rec=videos[row['video']];assert sha(rec['path'])==rec['sha256'];r=torch.load(rec['path'],map_location='cpu');norm=state['normalization'][m]
                    x=((r['representations'][m]-norm['mean'])/norm['std']).cuda();a=torch.cat([hs[m](part)[:,0].sigmoid().cpu() for part in x.split(256)]) if len(x) else torch.empty(0)
                    assert torch.equal(a,row['pbg'][r['hard']]) and (row['pbg'][~r['hard']]==0).all();checked+=1
    assert digest==gpu['final_digest']==base.tensor_digest(hs.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    save(entry/'independent_native.json',dict(passed=True,mode_videos=checked,all_predictions_bitwise_equal=True,parameters_RNG_unchanged=True,optimizer_steps=0,
        source_gpu_sha256=sha(entry/'gpu_complete.json'),guard=guard,guard_sha256=sha(guard)))


def conditional(bg,value,classes,scores,lengths,videos,edges):
    # 与生产按cell循环不同：逐BG显式构建不同视频且同三条件的前景集合。
    bins=np.array([(int(c),np.searchsorted(edges[str(int(c))]['score'],s,side='right'),np.searchsorted(edges[str(int(c))]['length'],l,side='right')) for c,s,l in zip(classes,scores,lengths)])
    output={};pairs=0;anchors=0
    for i in np.flatnonzero(bg):
        ids=np.flatnonzero((~bg)&(videos!=videos[i])&(bins==bins[i]).all(1))
        if not len(ids):continue
        a=sum(1. if value[i]>value[j] else .5 if value[i]==value[j] else 0. for j in ids)/len(ids)
        output.setdefault(videos[i],[]).append(a);pairs+=len(ids);anchors+=1
    return dict(video_macro_conditional_AUC=float(np.mean([np.mean(v) for v in output.values()])) if output else None,valid_videos=len(output),BG_anchors=anchors,pairs=pairs)


def audit_result(root,task):
    import torch,pandas as pd
    from tools import baseline_engine as base
    from tools.audit_background_ap_cost import IndependentAP,background
    from tools.foreground_oracle_ap import prediction_frame
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.audit_background_fit_gap import independent_auc
    e=root/task;shift,family=task.split('_');done=read(e/'complete.json');gpu=read(e/'gpu_complete.json');native=read(e/'independent_native.json');launch=read(Path(str(root)+'_launch.json'));al=read(root/'independent_audit_launch.json')
    guardcheck(gpu,launch['pid'],launch['head']);guardcheck(native,al['pid'],al['head']);assert sha(e/'gpu_complete.json')==done['gpu_complete_sha256']==native['source_gpu_sha256']
    assert sha(e/'predictions.pt')==gpu['predictions_sha256'];history=read(e/'history.json');assert [r['epoch'] for r in history]==list(range(1,21));checkpoints={}
    for ep in (1,5,10,20):
        p=e/f'epoch{ep:03d}.pt';st=torch.load(p,map_location='cpu');assert st['history']==history[:ep]
        assert {'heads','optimizers','order_rng','python_rng','numpy_rng','cpu_rng','cuda_rng','normalization','dimensions','modes','spec','extraction_SHA'}<=set(st)
        for d in ('source','validation'):assert st['extraction_SHA'][d]==sha(root/(shift+'_'+d)/'complete.json')
        steps=sum(h['steps'] for h in history[:ep]);assert all(float(v['step'])==steps for opt in st['optimizers'].values() for v in opt['state'].values());checkpoints[str(ep)]=sha(p)
    assert checkpoints['20']==gpu['checkpoint_sha256'] and base.tensor_digest(st['heads'])==gpu['final_digest'] and (e/'latest.pt').stat().st_ino==(e/'epoch020.pt').stat().st_ino
    split=read(OLD/shift/'split.json');original=torch.load(DIAG/shift/'inputs.pt',map_location='cpu');gt=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json')
    truth={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(gt[v]['segments'],gt[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')};truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    outputs=torch.load(e/'predictions.pt',map_location='cpu');records={}
    for d in ('source','validation'):
        cc=read(root/(shift+'_'+d)/'complete.json');nn=read(root/(shift+'_'+d)/'independent_native.json');assert sha(root/(shift+'_'+d)/'complete.json')==nn['source_complete_sha256'];guardcheck(cc,launch['pid'],launch['head']);guardcheck(nn,al['pid'],al['head'])
        for rec in cc['records']:records[rec['video']]=rec
    for m in st['modes']:
        fit=torch.cat([torch.load(records[v]['path'],map_location='cpu')['representations'][m] for v in sorted(split['fit'])]);norm=st['normalization'][m]
        assert torch.equal(fit.mean(0),norm['mean']) and torch.equal(fit.std(0,unbiased=False).clamp_min(1e-5),norm['std'])
    edges=read(DIAG/shift/'fit_costs_and_pairs.json')['pairing']['fit_only_edges'];results={}
    for role,methods in outputs.items():
        lookup={r['video']:r for r in original[role]};first=next(iter(methods.values()));f=prediction_frame(first);bg=background(truth[role],f);hard=f.score.to_numpy()>=.3;positions={};engines={};baseline=[]
        assert {r['video'] for r in first}==set(lookup)
        boxes=[]
        for row in first:
            src=lookup[row['video']];r=torch.load(records[row['video']]['path'],map_location='cpu');boxes.append(r['boxes'][r['hard']].numpy())
            for k in ('scores','labels','segments'):assert torch.equal(row[k],src[k])
            if role in ('fit','heldout'):
                z=np.flatnonzero((f['video-id'].to_numpy()==row['video'])&hard);assert np.array_equal(bg[z],src['bg'].numpy().astype(bool))
        b=np.concatenate(boxes);length=b[:,1]-b[:,0];vid=f['video-id'].to_numpy()[hard];scores=f.score.to_numpy()[hard];classes=f.label.to_numpy()[hard]
        for c in range(10):
            idx=np.flatnonzero(f.label.to_numpy()==c);positions[c]=idx;engines[c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[idx].reset_index(drop=True));baseline.append(engines[c].ap(np.ones(len(idx),bool)).tolist())
        results[role]={}
        for method in (*methods,'confidence'):
            frame=prediction_frame(methods[method] if method!='confidence' else first);value=1-frame.score.to_numpy() if method=='confidence' else frame.pbg.to_numpy();record=done['results'][role][method]
            for row in (methods[method] if method!='confidence' else first):
                src=lookup[row['video']]
                for k in ('scores','labels','segments'):assert torch.equal(row[k],src[k])
                assert (row['pbg'][~src['hard']]==0).all()
            auc=[]
            for _,idx in frame.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[hard[idx]];a=independent_auc(bg[idx],value[idx])
                if a is not None:auc.append(a)
            assert len(auc)==record['AUC_valid_videos'] and abs(np.mean(auc)-record['video_macro_AUC'])<1e-12
            cc=conditional(bg[hard],value[hard],classes,scores,length,vid,edges)
            for k,v in cc.items():assert record['conditional'][k]==v or (v is not None and record['conditional'][k] is not None and abs(record['conditional'][k]-v)<1e-12)
            budgets=[]
            for rate,rep in zip(SPEC['rates'],record['budgets']):
                selected=[]
                for c,idx in positions.items():
                    pool=[int(i) for i in idx if float(frame.score.iloc[i])>=.3];pool.sort(key=lambda i:(float(frame.score.iloc[i]) if method=='confidence' else -float(frame.pbg.iloc[i]),i));selected.extend(pool[:int(len(pool)*rate)])
                selected.sort();assert selected==rep['selected'];keep=np.ones(len(frame),bool);keep[selected]=False;ap=[engines[c].ap(keep[positions[c]]).tolist() for c in range(10)];delta=100*(np.mean(ap)-np.mean(baseline))
                assert np.allclose(ap,rep['per_class_AP'],atol=1e-7,rtol=0) and abs(delta-rep['delta_pp'])<1e-7 and abs(np.mean(baseline)-record['baseline'])<1e-7
                assert int(bg[selected].sum())==rep['BG_removed'] and int((~bg[selected]).sum())==rep['nonBG_removed']
                budgets.append(dict(rate=rate,delta_pp=float(delta),BG_removed=rep['BG_removed'],nonBG_removed=rep['nonBG_removed']))
            results[role][method]=dict(video_macro_AUC=float(np.mean(auc)),AUC_valid_videos=len(auc),conditional=cc,budgets=budgets)
    save(root/'independent_results'/(task+'.json'),dict(passed=True,source_only=True,complete_sha256=sha(e/'complete.json'),native_sha256=sha(e/'independent_native.json'),checkpoints=checkpoints,results=results))


def watch(root):
    from tools.baseline_gpu import snapshot
    for stage,tasks,key in [('extract',EXTRACT,'complete.json'),('probe',PROBES,'gpu_complete.json')]:
        while not all((root/t/key).exists() for t in tasks):
            if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped')
            time.sleep(3)
        # 不和生产GPU共享同一张卡；生产结束此阶段才重放。
        while read(root/'state.json')['stage'] in (['extract','probe'] if stage=='extract' else ['probe']):time.sleep(3)
        active={};pending=list(tasks)
        while pending or active:
            for task,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[task]
                if t['p'].returncode:raise RuntimeError(task+' independent native failed')
            for task in list(pending):
                report=read(root/task/key);guard=read(report['guard']);g=guard['observed']['index']
                if g in [t['gpu'] for t in active.values()] or snapshot(g)['free_mib']<12288:continue
                ob=snapshot(g);assert ob['uuid']==guard['observed']['uuid'];lease=root/task/('independent_'+stage+'_lease.json');save(lease,dict(owner_pid=os.getpid(),index=g,uuid=ob['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=ob['uuid'],PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease));log=(root/task/('independent_'+stage+'.log')).open('w')
                p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--native',str(root/task),'--stage',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log);pending.remove(task)
            time.sleep(3)
    while not (root/'final.json').exists():
        if (root/'error.json').exists() or (root/'stop.json').exists():raise RuntimeError('main stopped')
        time.sleep(3)
    for task in PROBES:audit_result(root,task);print('AUDITED',task,flush=True)
    save(root/'independent_final_audit.json',dict(passed=True,source_only=True,exact_tasks=PROBES,all219_layer_replay_native_hooks=True,
        final_sha256=sha(root/'final.json'),audits={t:sha(root/'independent_results'/(t+'.json')) for t in PROBES},extraction_native_sha256={t:sha(root/t/'independent_native.json') for t in EXTRACT}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?');p.add_argument('--native',type=Path);p.add_argument('--stage',choices=['extract','probe']);a=p.parse_args()
    if a.native:(extract_native if a.stage=='extract' else probe_native)(a.native)
    else:watch(a.root)
