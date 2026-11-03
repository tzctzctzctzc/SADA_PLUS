"""四卡只读：原始全维与已学时序特征，完整片段bank和两个预声明读出。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
DENSE=Path('/vepfs/vlms/highscore_dense_revisit_5ef4cca_1103')
TASKS=tuple(s+'_'+m for s in ('S1','S2') for m in ('raw','learned'))
SPEC=dict(schema='full-GT-semantic-memory-revisit-v1',tasks=TASKS,shifts=['S1','S2'],representations=['raw','learned'],readouts=['foreground','dual'],
    raw='full2304 rawSlowFast sourceFIT-normalized; no randomprojection or oldFPN',learned='same rawfeatures through fixed new dense10 fulltimeline-trained64 encoder, no newupdates',
    bank='ALL sourceFIT97verb GT intervals; task0..9 anchors alwaysretain, non-task10..96 only if maxIoUwithANYtask<.01; plus original highscore pureBG proposals as unknownclass97',
    descriptor='16 true interpolation points->4 orderedvalidmeans; perbin and concatenatedL2norm; invalid<4points excluded',
    source_exclusion='query physicalvideo excluded from all98classes, including fit queries',metric='exact float64 squaredL2; nearest5 average within each98class, closest among10task versus closest among88BGclasses',
    readouts='foreground: distance from closest taskclass; dual: foregrounddistance-BGdistance; fixed2mechanistic comparisons, no threshold/temp/ksearch',
    budget_seconds=900,rates=[.005,.01,.02,.05],source_fit_only=True,target_train_GT_read=False,target_train_model_updates=False,
    caveat='allGT bank changes support as well as metric; raw-vslearned also dimensions differ, no singlefactor causality;S1/S2 historicaldevelopment,not untouched test or fullsixdirections')


def gpu(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    from tools.run_dense_revisit import seed,raw_load
    from tools.dense_revisit_math import DenseEvidence
    from tools.semantic_memory_math import ordered_descriptor,semantic_distances
    guard=first_cuda_guard(entry);seed();s,kind=entry.name.split('_');began=time.monotonic();assert sha(DENSE/'independent_final_audit.json')=='3cf9fcc7eb3f06e2bc557af745816847bbde67f2725ab5e2820a86a0664f0141'
    data=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu');roles=data['roles'];mean=data['mean'];std=data['std'];vids={v:i for i,v in enumerate(sorted({r['video'] for rows in roles.values() for r in rows}))}
    model=None
    if kind=='learned':
        model=DenseEvidence().cuda();state=torch.load(DENSE/(s+'_dense10')/'epoch020.pt',map_location='cpu');model.load_state_dict(state['model']);model.eval();model.requires_grad_(False)
    checkpoint_sha=sha(DENSE/(s+'_dense10')/'epoch020.pt');md=base.tensor_digest(model.state_dict()) if model else None;cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0);cache={}
    def feature(r):
        x=((raw_load(r)-mean)/std).cuda()
        if model:
            logits,z=model(x)
            if r['video'] not in verified:
                old=old_lookup[r['video']];h=r['hard'];p,_,_=model.candidate(logits,z,r['grid_boxes'][h].cuda(),r['labels'][h].cuda(),r['scores'][h].cuda())
                assert torch.equal(p.cpu(),old['pbg'][h]);verified.add(r['video'])
            return z
        return x
    old_lookup={r['video']:r for rows in torch.load(DENSE/(s+'_dense10')/'predictions.pt',map_location='cpu').values() for r in rows};verified=set();bank=[];labels=[];bv=[];counts={}
    with torch.no_grad():
        for r in roles['fit']:
            z=feature(r);ann=r['annotations'];b=torch.tensor(ann['segments'],device='cuda',dtype=torch.float32).reshape(-1,2);c=torch.tensor(ann['labels'],device='cuda',dtype=torch.long);task=b[c<10]
            keep=c<10
            if len(task):
                inter=(torch.minimum(b[:,None,1],task[None,:,1])-torch.maximum(b[:,None,0],task[None,:,0])).clamp_min(0);union=(b[:,1]-b[:,0])[:,None]+(task[:,1]-task[:,0])[None]-inter
                keep|=(c>=10)&((inter/union.clamp_min(1e-12)).max(1).values<.01)
            else:keep|=c>=10
            b=b[keep];c=c[keep];hardboxes=r['grid_boxes'][r['hard']][r['bg'].bool()].cuda();b=torch.cat((b,hardboxes));c=torch.cat((c,torch.full((len(hardboxes),),97,device='cuda',dtype=torch.long)))
            for start in range(0,len(b),128):
                v,valid=ordered_descriptor(z,b[start:start+128]);bank.append(v[valid].cpu());labels.append(c[start:start+128][valid].cpu());bv.append(torch.full((int(valid.sum()),),vids[r['video']],dtype=torch.long))
            cache[r['video']]=z.cpu()
        bank=torch.cat(bank).cuda();labels=torch.cat(labels).cuda();bv=torch.cat(bv).cuda();assert (labels<10).any() and (labels>=10).any();counts={str(int(c)):int((labels==c).sum()) for c in labels.unique()}
        base.checkpoint(entry/'bank.pt',dict(features=bank.cpu(),labels=labels.cpu(),video=bv.cpu(),video_ids=vids,source_input_sha256=sha(DENSE/'inputs'/s/'data.pt'),checkpoint_sha256=checkpoint_sha,spec=SPEC))
        # 真实完整bank的一条GPU距离/逐类最近5，与独立NumPy全距离比较；保留自视频排除。
        query=bank[:3];a,b=semantic_distances(query,bank,labels,bv,int(bv[0]));q=query.cpu().double().numpy();bb=bank.cpu().double().numpy();cl=labels.cpu().numpy();vv=bv.cpu().numpy()
        dd=np.maximum(0,(q*q).sum(1)[:,None]+(bb*bb).sum(1)[None]-2*q@bb.T);columns=[]
        for c in range(98):
            ix=np.flatnonzero((cl==c)&(vv!=int(bv[0])))
            columns.append(np.sort(dd[:,ix],axis=1)[:,:min(5,len(ix))].mean(1) if len(ix) else np.full(len(q),np.inf))
        cc=np.stack(columns,1);fg=cc[:,:10].min(1);bg=cc[:,10:].min(1);assert np.allclose(a.cpu(),fg,atol=1e-10,rtol=1e-10) and np.allclose(b.cpu(),fg-bg,atol=1e-10,rtol=1e-10)
        bank=bank.double();bank_norm=bank.square().sum(1);cached=semantic_distances(query,bank,labels,bv,int(bv[0]),bank_norm=bank_norm)
        assert torch.equal(a,cached[0]) and torch.equal(b,cached[1])
        save(entry/'kernel_complete.json',dict(passed=True,exact_float64_GPU_and_independent_NumPy_fullbank3queries=True,self_video_excluded=True,bank_sha256=sha(entry/'bank.pt'),guard_sha256=sha(guard)))
        output={m:{} for m in ('foreground','dual')};processed=0;total=sum(len(roles[k]) for k in ('fit','heldout','validation','target_val'));querybegan=time.monotonic()
        for role in ('fit','heldout','validation','target_val'):
            for m in output:output[m][role]=[]
            for r in roles[role]:
                z=cache[r['video']].cuda() if r['video'] in cache else feature(r);h=r['hard'];b=r['grid_boxes'][h].cuda();scores=[];valids=[]
                for start in range(0,len(b),128):
                    query,valid=ordered_descriptor(z,b[start:start+128]);a,c=semantic_distances(query,bank,labels,bv,vids[r['video']],bank_norm=bank_norm);scores.append(torch.stack((a,c),1).cpu());valids.append(valid.cpu())
                values=torch.cat(scores) if scores else torch.empty(0,2,dtype=torch.float64);valid=torch.cat(valids) if valids else torch.empty(0,dtype=torch.bool);eligible=torch.zeros(len(h),dtype=torch.bool);eligible[h]=valid
                for col,m in enumerate(output):
                    p=torch.zeros(len(h),dtype=torch.float64);p[h]=values[:,col];p[~eligible]=0
                    output[m][role].append({k:r[k] for k in ('video','segments','scores','labels')}|dict(pbg=p,eligible=eligible))
                processed+=1
                if processed==3:
                    projected=(time.monotonic()-querybegan)*total/3;save(entry/'throughput.json',dict(first3_seconds=time.monotonic()-querybegan,projected_query_seconds=projected,setup_seconds=querybegan-began,total_videos=total,limit=900))
                    if projected>900:base.checkpoint(entry/'partial_predictions.pt',output);save(entry/'budget_pause.json',dict(scientific_failure=False,automatic_restart_authorized=False,bank_sha256=sha(entry/'bank.pt')));return
        assert torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0));assert model is None or md==base.tensor_digest(model.state_dict())
        assert kind=='raw' or len(verified)==total
        base.checkpoint(entry/'predictions.pt',output)
    save(entry/'gpu_complete.json',dict(passed=True,guard=guard,guard_sha256=sha(guard),bank_sha256=sha(entry/'bank.pt'),predictions_sha256=sha(entry/'predictions.pt'),kernel_sha256=sha(entry/'kernel_complete.json'),counts=counts,
        all_learned_original_dense_predictions_bitwise_replayed=kind=='learned',readonly_parameters_RNG_unchanged=True,zero_optimizer_steps=True,source_input_sha256=sha(DENSE/'inputs'/s/'data.pt'),checkpoint_sha256=checkpoint_sha,seconds=time.monotonic()-began))


def evaluate(root,s):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import prediction_frame,ClassAP
    from tools.audit_background_ap_cost import IndependentAP
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.run_dense_revisit import choose,SOURCE
    assert all((root/t/'gpu_complete.json').exists() for t in TASKS)
    roles=torch.load(DENSE/'inputs'/s/'data.pt',map_location='cpu')['roles'];out={}
    for kind in ('raw','learned'):
        for m,roles_out in torch.load(root/(s+'_'+kind)/'predictions.pt',map_location='cpu').items():out[kind+'_'+m]=roles_out
    cfg=read(SOURCE/s/'config.json');gt=read(SOURCE/s/'source_truth.json');truth={role:pd.DataFrame([(r['video'],float(b[0]),float(b[1]),int(c)) for r in roles[role] for b,c in zip(gt[r['video']]['segments'],gt[r['video']]['labels'])],columns=['video-id','t-start','t-end','label']) for role in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);truth['target_val']=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split']);results={}
    from tools.foreground_oracle_ap import background_mask
    for role in ('fit','heldout','validation','target_val'):
        frames={};eligible=None
        for m in out:
            rows=out[m][role]
            for a,b in zip(rows,roles[role]):
                assert a['video']==b['video']
                for k in ('segments','scores','labels'):assert torch.equal(a[k],b[k])
                assert (a['pbg'][~b['hard']]==0).all()
            frames[m]=prediction_frame(rows);v=torch.cat([r['eligible'] for r in rows]).numpy()
            if eligible is None:eligible=v
            else:assert np.array_equal(eligible,v)
        frame=next(iter(frames.values()));bg=background_mask(truth[role],frame);pos={};fast={};refs={};baseline=[]
        for c in range(10):
            ids=np.flatnonzero(frame.label.to_numpy()==c);pos[c]=ids;fast[c]=ClassAP(truth[role][truth[role].label==c],frame.iloc[ids]);refs[c]=IndependentAP(truth[role][truth[role].label==c],frame.iloc[ids].reset_index(drop=True));baseline.append(refs[c].ap(np.ones(len(ids),bool)))
        def effect(selected):
            keep=np.ones(len(frame),bool);keep[selected]=False;a=np.array([refs[c].ap(keep[pos[c]]) for c in range(10)]);b=[fast[c].evaluate(keep[pos[c]])['ap'] for c in range(10)];assert np.allclose(a,b,atol=1e-7,rtol=0)
            return float(100*(np.mean(a)-np.mean(baseline)))
        results[role]={}
        for m in (*out,'confidence'):
            f=frame if m=='confidence' else frames[m];value=1-f.score.to_numpy() if m=='confidence' else f.pbg.to_numpy();auc=[];budgets=[]
            for _,ids in f.groupby('video-id',sort=False).indices.items():
                ids=np.asarray(ids);ids=ids[f.iloc[ids].score.to_numpy()>=.3];a=independent_auc(bg[ids],value[ids])
                if a is not None:auc.append(a)
            for rate in SPEC['rates']:
                selected=choose(f,eligible,rate,m=='confidence');a=effect(selected);b=effect(selected[bg[selected]]);c=effect(selected[~bg[selected]])
                budgets.append(dict(rate=rate,selected=selected.tolist(),delta_pp=a,BG_only_delta_pp=b,nonBG_only_delta_pp=c,BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum())))
            results[role][m]=dict(baseline=float(np.mean(baseline)),video_macro_AUC=float(np.mean(auc)),AUC_valid_videos=len(auc),budgets=budgets)
        print('SEMANTIC_MEMORY_EVALUATED',s,role,flush=True)
    save(root/(s+'_evaluation.json'),dict(passed=True,results=results,all_original_arrays_equal=True,all_AP_independent_numpy_equal=True,target_train_GT_read=False,subset_effects_not_additive=True))


def run(root):
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),dense_independent_audit_sha256=sha(DENSE/'independent_final_audit.json')));active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    try:
        for stage,tasks in [('gpu',TASKS),('evaluate',('S1','S2'))]:
            pending=list(tasks)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' schedulingpause')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in range(4) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        v=snapshot(g);lease=entry/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=v['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']));env.update(CUDA_VISIBLE_DEVICES=v['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()},pending=pending,completed=done));time.sleep(3)
        save(root/'final.json',dict(passed=True,allGPU_sha256={t:sha(root/t/'gpu_complete.json') for t in TASKS},evaluations={s:sha(root/(s+'_evaluation.json')) for s in ('S1','S2')}));save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
    except Exception as ex:
        save(root/'error.json',dict(error=repr(ex)))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--entry',type=Path);p.add_argument('--worker',choices=['gpu','evaluate']);a=p.parse_args()
    if a.worker=='gpu':gpu(a.entry)
    elif a.worker=='evaluate':evaluate(a.entry.parent,a.entry.name)
    else:run(a.root)
