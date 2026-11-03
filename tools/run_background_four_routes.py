"""四路线S1完整源短预检；独立输入、真实GPU门禁、配对对照、全部固定预算。"""
import argparse
from copy import deepcopy
import math
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

OLD=Path('/vepfs/vlms/background_linear_temporal6_4932c27_1103')
ORIGIN=Path('/vepfs/vlms/background_ap_cost_source6_bd16507_1103')
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
ROUTES=('pauc','asl','knn','causal')
MODES={'pauc':('bce_control','pauc'), 'asl':('ase_control','ase_ascl'),
       'knn':('foreground_knn','dual_memory_knn'), 'causal':('attention_unmasked_control','causal_hybrid')}
SPEC=dict(schema='background-four-literature-routes-source-S1-v1',shifts=['S1'],routes=ROUTES,
    epochs=20,checkpoints=[1,5,10,20],readout_epoch=20,rates=[.005,.01,.02,.05],head_seed=311104,order_seed=123,
    lr=3e-4,wd=.05,clip=1.,batch=128,source_only=True,target_read=False,
    source_split='unchanged head-holdout-v1: S1 40 fit/11 heldout; full sourceval7',
    sample='original4/16/4 frozen FPN sample coordinates and validity; all original predictions including low scores retained',
    semantics='pure BG any-task maxIoU<.01; wrong predicted class covering any task action is protected nonBG',
    pauc=dict(beta=.01,eta=.001,margin=1.,positive='BG',negative='all nonBG',optimizer='official SOPA Adam both modes',
              input='max_linear descriptor, per-dimension standardization fitted on source fit only',
              sampling='full fit negative bank every step, random permutation of all fit positives each epoch; no per-video tail approximation'),
    asl=dict(width=64,GT_batch=8,temperature=.1,ASCL_weight=1.,auxiliary_weight=1.,
             mechanisms='class Gaussian cls/start/end sensitivity; instance cls/IoU quality sensitivity; weighted focal classification/DIoU localization; same-class cls/loc contrast versus other classes and real any-task-clean context',
             adaptation='sourceGT per-instance24-point evidence and candidate BG readout; small local encoder instead of full ASL detector; no-GT in inference',
             control='identical ASE/auxiliary heads; only ASCL gradient disabled'),
    knn=dict(k=50,input='raw512 inner-valid-mean float32 L2 normalized; float64 distance accumulation and saved scores',search='exact L2; query-video excluded from both source-fit banks even on fit readout',
             control='distance to foreground bank',adaptation='dual score d_foreground-d_background; no learned classifier, no distance threshold selection'),
    causal=dict(width=64,state=16,expand=2,heads=4,kernel=4,drop_path=0.,
                implementation='official HybridCausalBlock and MixtureCausalBlock AST unchanged; original selective_scan_ref and float32 attention math reference',
                adaptation='one full hybrid block on each candidate24-point evidence; no new detection endpoints or FPN pyramid',
                control='same block/parameters/selective SSM, only attention masking disabled',
                limitation='isolates attention mask within full hybrid; not whole CausalTAD system reproduction or causal identification; no native fused-kernel equivalence claim'),
    precision='float32 weights/inputs; inherited original TF32 settings; no AMP/quantization; reference selective scan casts as official source',
    budget=dict(max_estimated_training_seconds_per_route=900,policy='probe throughput first, preserve state if projected budget exceeded; no silent epochs/samples change'),
    selection='no epoch/threshold/route winner selection; all controls, modes, roles and rates reported; detector saw head-heldout source videos')


def seed():
    import torch
    random.seed(123);np.random.seed(123);torch.manual_seed(311104);torch.cuda.manual_seed_all(311104)


def prepare(root):
    import torch
    from tools import baseline_engine as base
    from tools.background_linear_temporal import region_samples,descriptors
    root.mkdir(parents=True,exist_ok=False)
    assert read(OLD/'independent_final_audit.json')['passed']
    assert sha(OLD/'independent_final_audit.json')=='30f33e64042d4e2a2791f841e616a9d6511e52dc1e877aa8c9e6cf38da5986ce'
    s='S1';cost=read(ORIGIN/s/'costs.json');split=read(OLD/s/'split.json');truth=read(SOURCE/s/'source_truth.json')
    prepared={r:[] for r in ('fit','heldout','validation')};gtclips=[];provenance=[]
    for role,records in [('source',[x['record'] for x in cost['rows']]),('validation',cost['validation'])]:
        byvideo={x['record']['video']:x for x in cost['rows']}
        for rec in records:
            raw=torch.load(rec['path'],map_location='cpu');v=rec['video'];hard=raw['scores']>=.3
            dest='validation' if role=='validation' else ('fit' if v in split['fit'] else 'heldout')
            values,masks=region_samples(raw['feature'],raw['valid'],raw['boxes'][hard])
            seq=torch.cat(values,1);valid=torch.cat(masks,1)
            desc=descriptors(raw['feature'],raw['valid'],raw['boxes'][hard],raw['labels'][hard],raw['scores'][hard])['max_linear']
            item=dict(video=v,segments=raw['segments' if dest=='validation' else 'boxes'],scores=raw['scores'],labels=raw['labels'],
                      hard=hard,seq=seq,valid=valid,metadata=desc[:,-12:],linear=desc,
                      inner=(values[1]*masks[1][...,None]).sum(1)/masks[1].sum(1,keepdim=True).clamp_min(1))
            if role=='source':
                row=byvideo[v];assert torch.where(hard)[0].tolist()==row['hard_indices']
                item.update(bg=torch.tensor(row['bg']),quality=torch.tensor(row['quality']))
            prepared[dest].append(item)
            provenance.append(dict(video=v,role=dest,path=rec['path'],old_record=rec,stat_size=Path(rec['path']).stat().st_size))
            if dest=='fit':
                boxes=torch.tensor(truth[v]['segments'],dtype=torch.float32).reshape(-1,2)
                labels=torch.tensor(truth[v]['labels'],dtype=torch.long)
                vals,oks=region_samples(raw['feature'],raw['valid'],boxes)
                length=boxes[:,1]-boxes[:,0]
                positions=torch.cat([a[:,None]+(b-a)[:,None]*(torch.arange(n)+.5)/n for a,b,n in
                    ((boxes[:,0]-.25*length,boxes[:,0],4),(boxes[:,0],boxes[:,1],16),(boxes[:,1],boxes[:,1]+.25*length,4))],1)
                context_pos=torch.cat((positions[:,:4],positions[:,20:]),1)
                inside=((context_pos[:,:,None]>=boxes[None,None,:,0])&(context_pos[:,:,None]<=boxes[None,None,:,1])).any(-1)
                clean=~inside & torch.cat((oks[0],oks[2]),1)
                sampled_gt=torch.cat(vals,1);valid_gt=torch.cat(oks,1)
                for i in range(len(boxes)):
                    if oks[1][i].sum()>=4:
                        gtclips.append(dict(video=v,label=int(labels[i]),samples=sampled_gt[i],valid=valid_gt[i],bg_context=clean[i]))
    assert {x['video'] for x in prepared['fit']}==set(split['fit'])
    assert {x['video'] for x in prepared['heldout']}==set(split['heldout'])
    assert not set(split['fit']+split['heldout'])&{x['video'] for x in prepared['validation']}
    assert all(x['video'] in split['fit'] for x in gtclips)
    base.checkpoint(root/'prepared.pt',dict(roles=prepared,GTclips=gtclips))
    # 不读目标，validation无任何标签；heldout标签仅保存在CPU输入工件，train只消费fit。
    save(root/'prepare_complete.json',dict(passed=True,prepared_sha256=sha(root/'prepared.pt'),split=split,
        old_final_audit_sha256=sha(OLD/'independent_final_audit.json'),costs_sha256=sha(ORIGIN/s/'costs.json'),truth_sha256=sha(SOURCE/s/'source_truth.json'),
        counts={r:dict(videos=len(rows),hard=sum(int(x['hard'].sum()) for x in rows)) for r,rows in prepared.items()},
        GTclips=len(gtclips),raw_cache_reused_from_audited_origin=True,raw_provenance=provenance))
    save(root/'plan.json',dict(spec=SPEC,code=code_identity(),source_provenance={p.name:read(p) for p in (CODE/'tools/four_route_sources').glob('*_provenance.json')}))


def model_factory(route,mode):
    from tools.background_four_routes_math import ASLAdapter,ProposalEncoder
    from tools.background_linear_temporal import LinearBackground
    if route=='pauc':return LinearBackground('max_linear')
    if route=='asl':return ASLAdapter(contrast=mode=='ase_ascl')
    return ProposalEncoder('causal',causal=mode=='causal_hybrid')


def capture(models,opts,generator,history,aux=None):
    import torch
    return dict(models={m:h.state_dict() for m,h in models.items()},optimizers={m:o.state_dict() for m,o in opts.items()},
                optimizer_counters={m:{k:getattr(o,k) for k in ('T','steps','lr','mode') if hasattr(o,k)} for m,o in opts.items()},
                order_rng=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(0),history=history,spec=SPEC,auxiliary=aux)


def gradient_step(model,opt,batch,route,auxiliary=None,gtbatch=None):
    import torch
    from tools.background_four_routes_math import head_objective
    opt.zero_grad(set_to_none=True)
    if route=='pauc':
        logits=model(batch['linear'])
        if auxiliary is None:loss=head_objective(logits,batch['bg'],batch['quality'])
        else:
            from tools.hard_background_modules import quality_loss
            loss=auxiliary(logits[:,0].sigmoid(),batch['bg'],batch['index'])+.25*quality_loss(logits[:,1],batch['quality'])
    else:
        logits=model(batch['seq'],batch['valid'],batch['metadata'])
        loss=head_objective(logits,batch['bg'],batch['quality'])
        if route=='asl':loss=loss+model.auxiliary(*gtbatch)[0]
    assert torch.isfinite(loss)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(norm)
    opt.step();return float(loss.detach()),float(norm)


def concatenate(rows,keys,device):
    import torch
    return {k:torch.cat([r[k] for r in rows]).to(device) for k in keys}


def take(data,ids):return {k:x[ids] for k,x in data.items()}


def kernel(entry,route,fit,gtdata):
    import torch
    from tools import baseline_engine as base
    from tools.background_four_routes_math import official_pauc,official_sopa
    seed();m=MODES[route][-1];model=model_factory(route,m).cuda()
    opt=official_sopa(model.parameters(),'cuda') if route=='pauc' else torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    pos=torch.where(fit['bg']==1)[0][:16];neg=torch.where(fit['bg']==0)[0][:16]
    batch=take(fit,torch.cat((pos,neg)))
    aux=official_pauc(len(fit['bg']),int(fit['bg'].sum()),'cuda') if route=='pauc' else None
    gtb=tuple(v[:4] for v in gtdata) if route=='asl' else None
    generator=torch.Generator().manual_seed(123)
    initial=base.tensor_digest(model.state_dict())
    t=time.monotonic();loss,norm=gradient_step(model,opt,batch,route,aux,gtb);torch.cuda.synchronize();first_seconds=time.monotonic()-t
    assert norm>0 and initial!=base.tensor_digest(model.state_dict())
    state=capture({m:model},{m:opt},generator,[],dict(u_pos=aux.u_pos if aux else None))
    base.checkpoint(entry/'kernel_state.pt',state)
    restored=torch.load(entry/'kernel_state.pt',map_location='cpu');fresh=model_factory(route,m).cuda();fresh.load_state_dict(restored['models'][m])
    fo=official_sopa(fresh.parameters(),'cuda') if route=='pauc' else torch.optim.AdamW(fresh.parameters(),lr=3e-4,weight_decay=.05)
    fo.load_state_dict(restored['optimizers'][m]);fa=official_pauc(len(fit['bg']),int(fit['bg'].sum()),'cuda') if aux else None
    for k,v in restored['optimizer_counters'][m].items():setattr(fo,k,v)
    if fa is not None:fa.u_pos.copy_(restored['auxiliary']['u_pos'].cuda())
    def restore():
        torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state(restored['cuda_rng'],0);random.setstate(restored['python_rng']);np.random.set_state(restored['numpy_rng']);generator.set_state(restored['order_rng'])
    restore();draw=(torch.rand(3),torch.rand(3,device='cuda'),random.random(),np.random.rand(),torch.randperm(11,generator=generator))
    restore();repeat=(torch.rand(3),torch.rand(3,device='cuda'),random.random(),np.random.rand(),torch.randperm(11,generator=generator))
    assert all(torch.equal(draw[i],repeat[i]) for i in (0,1,4)) and draw[2:4]==repeat[2:4]
    restore();gradient_step(model,opt,batch,route,aux,gtb)
    restore();gradient_step(fresh,fo,batch,route,fa,gtb)
    assert base.tensor_digest(model.state_dict())==base.tensor_digest(fresh.state_dict())
    assert base.tensor_digest(opt.state_dict())==base.tensor_digest(fo.state_dict())
    if aux:assert torch.equal(aux.u_pos,fa.u_pos)
    save(entry/'kernel_complete.json',dict(passed=True,loss=loss,gradient=norm,full_state_next_step_exact=True,RNG_draws_exact=True,
        official_auxiliary_state_saved=route=='pauc',checkpoint_sha256=sha(entry/'kernel_state.pt'),first_small_step_seconds=first_seconds,
        parameters=sum(p.numel() for p in model.parameters())))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.background_four_routes_math import official_pauc,official_sopa
    route=entry.name;root=entry.parent;guard=first_cuda_guard(entry);started=time.monotonic()
    assert read(root/'prepare_complete.json')['prepared_sha256']==sha(root/'prepared.pt')
    data=torch.load(root/'prepared.pt',map_location='cpu');roles=data['roles']
    if route=='knn':return knn(entry,roles,guard)
    keys=['linear','bg','quality'] if route=='pauc' else ['seq','valid','metadata','bg','quality']
    fit=concatenate(roles['fit'],keys,'cuda');fit['index']=torch.arange(len(fit['bg']),device='cuda')
    stats={}
    if route=='pauc':
        mean=fit['linear'].mean(0);std=fit['linear'].std(0).clamp_min(1e-3);fit['linear']=(fit['linear']-mean)/std
        stats=dict(mean=mean,std=std)
    clips=data['GTclips'];gtdata=tuple(torch.stack([x[k] if k!='label' else torch.tensor(x[k]) for x in clips]).cuda()
        for k in ('samples','valid','bg_context','label')) if route=='asl' else None
    kernel(entry,route,fit,gtdata)
    seed();generator=torch.Generator().manual_seed(123)
    models={m:model_factory(route,m).cuda() for m in MODES[route]}
    # 相同模式的随机初始化必须逐tensor配对；contrast与attention flag不在state_dict中。
    models[MODES[route][1]].load_state_dict(models[MODES[route][0]].state_dict())
    initial={m:base.tensor_digest(h.state_dict()) for m,h in models.items()};assert len(set(initial.values()))==1
    opts={m:(official_sopa(h.parameters(),'cuda') if route=='pauc' else torch.optim.AdamW(h.parameters(),lr=3e-4,weight_decay=.05)) for m,h in models.items()}
    aux=official_pauc(len(fit['bg']),int(fit['bg'].sum()),'cuda') if route=='pauc' else None
    pos=torch.where(fit['bg']==1)[0].cpu();neg=torch.where(fit['bg']==0)[0]
    history=[];timed=[];steps=0;peak=0
    for epoch in range(1,21):
        before=time.monotonic();losses={m:0. for m in models};count=0
        ids=pos[torch.randperm(len(pos),generator=generator)] if route=='pauc' else torch.randperm(len(fit['bg']),generator=generator)
        for part in ids.split(128):
            batch_ids=torch.cat((part.cuda(),neg)) if route=='pauc' else part.cuda()
            batch=take(fit,batch_ids)
            gtb=None
            if route=='asl':
                gi=torch.randperm(len(gtdata[0]),generator=generator)[:8].cuda();gtb=tuple(v[gi] for v in gtdata)
            for m,model in models.items():
                loss,_=gradient_step(model,opts[m],batch,route,aux if m=='pauc' else None,gtb);losses[m]+=loss
            torch.cuda.synchronize();steps+=1;count+=1
        seconds=time.monotonic()-before;timed.append(seconds)
        history.append(dict(epoch=epoch,loss={m:v/count for m,v in losses.items()},seconds=seconds,steps=count))
        auxiliary=dict(u_pos=aux.u_pos if aux else None,normalization=stats,initial_digests=initial)
        base.checkpoint(entry/'latest.pt',capture(models,opts,generator,history,auxiliary))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('EPOCH',route,epoch,seconds,flush=True)
        if epoch==1:
            estimate=seconds*20
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=estimate,limit=900,steps=count,
                policy='full planned dataset and20 epochs, no post-hoc sample or epoch change'))
            if estimate>900:
                save(entry/'budget_pause.json',dict(reason='user short-experiment scheduling budget',scientific_failure=False,automatic_restart_authorized=False,
                    estimated_training_seconds=estimate,state_sha256=sha(entry/'latest.pt'),completed_epoch=epoch))
                return
    outputs={}
    for role,rows in roles.items():
        outputs[role]={m:[] for m in models}
        for m,model in models.items():
            model.eval();digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
            with torch.no_grad():
                for row in rows:
                    parts=[]
                    for start in range(0,int(row['hard'].sum()),128):
                        sl=slice(start,start+128)
                        if route=='pauc':out=model((row['linear'][sl].cuda()-stats['mean'])/stats['std'])
                        else:out=model(row['seq'][sl].cuda(),row['valid'][sl].cuda(),row['metadata'][sl].cuda())
                        parts.append(out[:,0].sigmoid().cpu())
                    p=torch.zeros(len(row['scores']));p[row['hard']]=torch.cat(parts) if parts else torch.empty(0)
                    outputs[role][m].append({k:row[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
            assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    base.checkpoint(entry/'predictions.pt',outputs)
    save(entry/'gpu_complete.json',dict(passed=True,route=route,guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),
        predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),prepared_sha256=sha(root/'prepared.pt'),
        initial_digests=initial,final_digests={m:base.tensor_digest(h.state_dict()) for m,h in models.items()},readout_parameters_and_RNG_unchanged=True,
        elapsed_seconds=time.monotonic()-started,optimizer_steps=steps,peak_mib=torch.cuda.max_memory_reserved()/2**20,spec=SPEC))


def knn(entry,roles,guard):
    import torch
    from tools.background_four_routes_math import knn_distance
    from tools import baseline_engine as base
    began=time.monotonic();fit=roles['fit'];bank=torch.cat([r['inner'] for r in fit]).cuda();bg=torch.cat([r['bg'] for r in fit]).bool().cuda()
    video=torch.cat([torch.full((len(r['bg']),),i,dtype=torch.long) for i,r in enumerate(fit)]).cuda()
    outputs={};checks=[]
    # 独立cdist核实际原特征上的精确topk，使用容差而非声称TF32与CPU逐bit。
    q=bank[:7];test=bank[:89]
    actual=knn_distance(q,test,k=50);normq=torch.nn.functional.normalize(q,dim=1);normb=torch.nn.functional.normalize(test,dim=1)
    ref=torch.cdist(normq.double(),normb.double()).square().sort(1).values[:,49]
    assert torch.allclose(actual,ref,atol=1e-12,rtol=1e-12)
    save(entry/'kernel_complete.json',dict(passed=True,exact_topk_vs_cdist=True,max_abs_error=float((actual-ref).abs().max()),optimizer_steps=0))
    for role,rows in roles.items():
        outputs[role]={m:[] for m in MODES['knn']}
        for r in rows:
            same=next((i for i,x in enumerate(fit) if x['video']==r['video']),None)
            eligible=torch.ones(len(bg),dtype=torch.bool,device='cuda') if same is None else video!=same
            assert same is None or not (eligible&(video==same)).any()
            query=r['inner'].cuda();fgd=knn_distance(query,bank[eligible&~bg]);bgd=knn_distance(query,bank[eligible&bg])
            checks.append(dict(role=role,video=r['video'],excluded_same_video=same is not None,FG_bank=int((eligible&~bg).sum()),BG_bank=int((eligible&bg).sum())))
            for m,score in [('foreground_knn',fgd),('dual_memory_knn',fgd-bgd)]:
                p=torch.zeros(len(r['scores']),dtype=torch.float64);p[r['hard']]=score.cpu()
                outputs[role][m].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
    base.checkpoint(entry/'predictions.pt',outputs);save(entry/'memory_identity.json',dict(queries=checks,fit_videos=[r['video'] for r in fit],k=50,bank_membership='only source fit; all same-video entries excluded for each query'))
    save(entry/'gpu_complete.json',dict(passed=True,route='knn',guard=guard,guard_sha256=sha(guard),kernel_sha256=sha(entry/'kernel_complete.json'),
        predictions_sha256=sha(entry/'predictions.pt'),memory_sha256=sha(entry/'memory_identity.json'),prepared_sha256=sha(entry.parent/'prepared.pt'),
        optimizer_steps=0,elapsed_seconds=time.monotonic()-began,spec=SPEC))


def evaluate(entry):
    import torch,pandas as pd
    from tools.audit_background_ap_cost import IndependentAP,make_frame,background,selected_indices
    from tools.audit_background_fit_gap import independent_auc
    from tools.background_ap_cost import official_ap
    from tools.run_foreground_calibration_oracle import load_truth
    root=entry.parent;route=entry.name;gpu=read(entry/'gpu_complete.json');assert gpu['passed'] and sha(entry/'predictions.pt')==gpu['predictions_sha256']
    out=torch.load(entry/'predictions.pt',map_location='cpu');split=read(root/'prepare_complete.json')['split'];source=read(SOURCE/'S1/source_truth.json');cfg=read(SOURCE/'S1/config.json')
    truth={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(source[v]['segments'],source[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split'])
    result={};artifacts={}
    original=torch.load(root/'prepared.pt',map_location='cpu')['roles']
    for role,modes in out.items():
        result[role]={};reference={x['video']:x for x in original[role]};frames={m:make_frame(rows) for m,rows in modes.items()}
        for m,rows in modes.items():
            assert len(rows)==len(reference) and {r['video'] for r in rows}==set(reference)
            for row in rows:
                raw=reference[row['video']]
                for k in ('scores','segments','labels'):assert torch.equal(row[k],raw[k])
                assert (row['pbg'][~raw['hard']]==0).all()
        frame=next(iter(frames.values()));bg=background(truth[role],frame);engines={};fastengines={};positions={};baseline=[];official=[]
        from tools.foreground_oracle_ap import ClassAP
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);positions[c]=ids
            engines[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True))
            fastengines[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[ids])
            ap=engines[c].ap(np.ones(len(ids),dtype=bool));baseline.append(ap.tolist())
            check=official_ap(truth[role][truth[role].label==c],frame.iloc[ids]);assert np.max(np.abs(ap-check))<1e-7;official.append(c)
        for m in (*MODES[route],'confidence'):
            f=frame if m=='confidence' else frames[m];aucs=[];budgets=[]
            for v,idx in f.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[f.iloc[idx].score.to_numpy()>=.3]
                a=independent_auc(bg[idx],1-f.iloc[idx].score.to_numpy() if m=='confidence' else f.iloc[idx].pbg.to_numpy())
                if a is not None:aucs.append(a)
            for rate in SPEC['rates']:
                chosen=selected_indices(f,rate,m=='confidence');assert (f.iloc[chosen].score>=.3).all()
                keep=np.ones(len(f),dtype=bool);keep[chosen]=False
                ap=[engines[c].ap(keep[positions[c]]).tolist() for c in range(10)]
                # NumPy独立matcher之外，对全部操作点用旧Numba ClassAP再重算。
                fast=[fastengines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)]
                assert np.max(np.abs(np.array(ap)-np.array(fast)))<1e-7
                budgets.append(dict(rate=rate,map=float(np.mean(ap)),delta_pp=float(100*(np.mean(ap)-np.mean(baseline))),
                    BG_removed=int(bg[chosen].sum()),nonBG_removed=int((~bg[chosen]).sum()),selected=chosen.tolist(),per_class_AP=ap))
            result[role][m]=dict(baseline=float(np.mean(baseline)),original_per_class_AP=baseline,video_macro_AUC=float(np.mean(aucs)) if aucs else None,AUC_valid_videos=len(aucs),total_videos=len(reference),budgets=budgets)
        print('EVALUATED',route,role,flush=True)
    save(entry/'complete.json',dict(passed=True,source_only=True,shift='S1',results=result,gpu_complete_sha256=sha(entry/'gpu_complete.json'),
        prepared_sha256=sha(root/'prepared.pt'),all_original_arrays_exact=True,all_low_scores_protected=True,official_all_baselines=True,
        independent_numpy_vs_numba_all_AP=True,spec=SPEC))


def run(root):
    from tools.baseline_gpu import snapshot
    assert (root/'prepare_complete.json').exists() and not (root/'state.json').exists()
    active={};done=[];pending=list(ROUTES)
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage in ('train','evaluate'):
            pending=list(ROUTES)
            while pending or active:
                for r,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[r]
                    if t['p'].returncode:raise RuntimeError(r+'/'+stage+' failed')
                    if (root/r/'budget_pause.json').exists():raise RuntimeError(r+' preserved at scheduling budget pause; no scientific failure')
                    done.append(r+'/'+stage)
                choices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in (0,1,2,3) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in choices:
                    if not pending:break
                    r=pending.pop(0);e=root/r;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        observed=snapshot(g);lease=e/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                        env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(e/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(e)],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[r]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,pending=pending,completed=done,owner_pid=os.getpid(),active={r:dict(pid=t['p'].pid,gpu=t['gpu']) for r,t in active.items()}));time.sleep(3)
        results={r:read(root/r/'complete.json') for r in ROUTES};assert all(x['passed'] for x in results.values())
        save(root/'final.json',dict(source_only=True,shift='S1',results=results,complete_sha256={r:sha(root/r/'complete.json') for r in ROUTES},spec=SPEC))
        save(root/'state.json',dict(stage='complete',active={},owner_pid=os.getpid()))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--run',type=Path);p.add_argument('--worker',choices=['train','evaluate']);p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.run:run(a.run)
    elif a.worker=='train':train(a.entry)
    else:evaluate(a.entry)
