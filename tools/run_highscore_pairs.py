"""S1/S2四卡短实验：同架构、同输入顺序、同完整预算，只改变配对目标。"""
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
from tools.run_background_four_routes import seed,concatenate
from tools.highscore_confusion_diagnosis import OLD,SOURCE
from tools.highscore_pair_math import make_model,build_batch,objective

DIAG=Path('/vepfs/vlms/highscore_confusion_diag_36c0916_1103')
TASKS=('S1_control','S1_pairs','S2_control','S2_pairs')
SPEC=dict(schema='highscore-AP-matched-visual-pairs-v1',shifts=['S1','S2'],tasks=TASKS,
    source_only=True,target_read=False,epochs=20,checkpoints=[1,5,10,20],readout_epoch=20,seed=311104,order_seed=123,pair_seed=311105,
    lr=3e-4,weight_decay=.05,clip=1.,batch=128,pair_batch=64,pair_weight=.25,margin=.5,rates=[.005,.01,.02,.05],
    encoder='same64dim ProposalEncoder as priorASL adaptation without GT sensitivity/auxiliary; original frozen512dim FPN4/16/4 samples',
    baseline='balanced_BG_BCE+.25 quality; pair loss computed but gradient weight0',
    treatment='baseline+.25 AP-weighted softplus(BG-versus-criticalFG margin) on matched real high-score proposals',
    pair_metadata='each pair shares the mean original score/relative-length metadata; same predicted class; ordinary BCE uses actual metadata',
    pair_selection='fixed diagnostic sourcefit only: same predicted class, score decile, grid-length quartile, differentvideo; up to4 closest raw visual FG with harmful full-fit AP deletion cost per beneficial BG',
    pair_importance='positive full-fit BG deletion marginal + absolute negative full-fit FG deletion marginal, normalized within batch; not additive AP derivative',
    scientific_difference='oldrank was originalscoretop32 weighted average pairs; oldAPcost was per-video weighted BCE; current pairs are cross-video, confound matched, visual-near, full-fit AP-critical and share nuisance metadata',
    comparability='control and treatment same initialization/primary and pair indices/forward shapes/AdamW/RNG/budget; no val or target feedback',
    precision='float32 inheritedTF32 environment, no quantization/AMP change',
    efficiency='immutable CPU sample cache read once then fit tensors copied toGPU once, cache retains all original predictions',
    budget='firstepoch end-to-end step timing; pause with complete state if projected20epoch training>900seconds',
    caveat='head-only video holdout; detector saw source training videos; no assertion of target or six-direction gains')


def state(model,opt,order,pair_rng,history,source_sha,pairs_sha):
    import torch
    return dict(model=model.state_dict(),optimizer=opt.state_dict(),order_rng=order.get_state(),pair_rng=pair_rng.get_state(),
        python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),
        history=history,input_sha256=source_sha,pairs_sha256=pairs_sha,spec=SPEC)


def update(model,opt,data,primary,pair_indices,pair_table,weights,enabled):
    import torch
    opt.zero_grad(set_to_none=True);batch=build_batch(data,primary,pair_table[pair_indices])
    loss,terms=objective(model,batch,len(primary),weights[pair_indices],enabled)
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm)
    opt.step();return float(loss.detach()),float(norm),{k:float(v) for k,v in terms.items()}


def kernel(entry,data,table,weights,enabled):
    import torch
    from tools import baseline_engine as base
    seed();m=make_model().cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.05)
    order=torch.Generator().manual_seed(123);pair_rng=torch.Generator().manual_seed(311105)
    primary=torch.arange(min(64,len(data['bg'])),device='cuda');pi=torch.arange(min(16,len(table)),device='cuda')
    before=base.tensor_digest(m.state_dict());loss,norm,_=update(m,opt,data,primary,pi,table,weights,enabled)
    assert norm>0 and before!=base.tensor_digest(m.state_dict())
    base.checkpoint(entry/'kernel_state.pt',state(m,opt,order,pair_rng,[],None,None));saved=torch.load(entry/'kernel_state.pt',map_location='cpu')
    fresh=make_model().cuda();fresh.load_state_dict(saved['model']);fo=torch.optim.AdamW(fresh.parameters());fo.load_state_dict(saved['optimizer'])
    def restore():
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],0);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);order.set_state(saved['order_rng']);pair_rng.set_state(saved['pair_rng'])
    restore();draw=(torch.rand(3),torch.rand(3,device='cuda'),torch.randperm(17,generator=order),torch.randperm(19,generator=pair_rng),random.random(),np.random.rand())
    restore();again=(torch.rand(3),torch.rand(3,device='cuda'),torch.randperm(17,generator=order),torch.randperm(19,generator=pair_rng),random.random(),np.random.rand())
    assert all(torch.equal(draw[i],again[i]) for i in range(4)) and draw[4:]==again[4:]
    restore();update(m,opt,data,primary,pi,table,weights,enabled)
    restore();update(fresh,fo,data,primary,pi,table,weights,enabled)
    assert base.tensor_digest(m.state_dict())==base.tensor_digest(fresh.state_dict()) and base.tensor_digest(opt.state_dict())==base.tensor_digest(fo.state_dict())
    batch=build_batch(data,primary,table[pi]);n=len(primary);p=len(pi)
    assert torch.equal(batch['metadata'][n:n+p],batch['metadata'][n+p:])
    zero={k:v.clone() for k,v in batch.items()};zero['seq'].zero_()
    with torch.no_grad():z=m(zero['seq'],zero['valid'],zero['metadata'])
    # 有效mask不同也会产生覆盖信息，所以共享mask再核零视觉配对不可由metadata区分。
    zero['valid'][n+p:]=zero['valid'][n:n+p]
    with torch.no_grad():z=m(zero['seq'],zero['valid'],zero['metadata'])
    assert torch.equal(z[n:n+p],z[n+p:])
    save(entry/'kernel_complete.json',dict(passed=True,gradient=norm,loss=loss,full_state_next_step_exact=True,RNG_exact=True,
        shared_pair_metadata_exact=True,zero_visual_same_mask_pair_logits_equal=True,state_sha256=sha(entry/'kernel_state.pt')))


def train(entry):
    import torch
    from tools import baseline_engine as base
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);shift,mode=entry.name.split('_');enabled=mode=='pairs';began=time.monotonic()
    assert read(DIAG/'complete.json')['passed'];cost=read(DIAG/shift/'fit_costs_and_pairs.json');path=DIAG/shift/'inputs.pt'
    assert sha(path)==cost['inputs_sha256'];roles=torch.load(path,map_location='cpu')
    data=concatenate(roles['fit'],('seq','valid','metadata','bg','quality'),'cuda')
    assert np.array_equal(data['bg'].cpu().numpy().astype(bool),cost['BG'])
    pairs=cost['pairing']['pairs'];assert len(pairs)>0
    table=torch.tensor([[p['BG'],p['foreground']] for p in pairs],device='cuda')
    weights=torch.tensor([p['BG_benefit']+p['FG_harm'] for p in pairs],device='cuda')
    assert (data['bg'][table[:,0]]==1).all() and (data['bg'][table[:,1]]==0).all() and (weights>0).all()
    kernel(entry,data,table,weights,enabled)
    seed();model=make_model().cuda();initial=base.tensor_digest(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.05)
    order=torch.Generator().manual_seed(123);pair_rng=torch.Generator().manual_seed(311105);history=[]
    for epoch in range(1,21):
        begin=time.monotonic();losses=[];primary_loss=[];pair_loss=[]
        permutation=torch.randperm(len(data['bg']),generator=order)
        for ids in permutation.split(128):
            pi=torch.randperm(len(table),generator=pair_rng)[:64].cuda()
            loss,norm,terms=update(model,opt,data,ids.cuda(),pi,table,weights,enabled)
            losses.append(loss);primary_loss.append(terms['ordinary']);pair_loss.append(terms['pair'])
        torch.cuda.synchronize();seconds=time.monotonic()-begin
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),ordinary=float(np.mean(primary_loss)),pair=float(np.mean(pair_loss)),seconds=seconds,steps=len(losses)))
        base.checkpoint(entry/'latest.pt',state(model,opt,order,pair_rng,history,cost['inputs_sha256'],sha(DIAG/shift/'fit_costs_and_pairs.json')))
        if epoch in (1,5,10,20):os.link(entry/'latest.pt',entry/f'epoch{epoch:03d}.pt')
        save(entry/'history.json',history);print('EPOCH',entry.name,epoch,seconds,flush=True)
        if epoch==1:
            save(entry/'throughput.json',dict(first_epoch_seconds=seconds,projected_training_seconds=20*seconds,limit=900))
            if 20*seconds>900:
                save(entry/'budget_pause.json',dict(scientific_failure=False,reason='user short-experiment scheduling constraint',automatic_restart_authorized=False,epoch=1,state_sha256=sha(entry/'latest.pt')));return
    model.eval();outputs={};digest=base.tensor_digest(model.state_dict());cr=torch.get_rng_state();gr=torch.cuda.get_rng_state(0)
    with torch.no_grad():
        for role,rows in roles.items():
            outputs[role]=[]
            for row in rows:
                parts=[]
                for start in range(0,len(row['seq']),128):
                    sl=slice(start,start+128);parts.append(model(row['seq'][sl].cuda(),row['valid'][sl].cuda(),row['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                p=torch.zeros(len(row['scores']));p[row['hard']]=torch.cat(parts) if parts else torch.empty(0)
                outputs[role].append({k:row[k] for k in ('video','segments','scores','labels')}|dict(pbg=p))
    assert digest==base.tensor_digest(model.state_dict()) and torch.equal(cr,torch.get_rng_state()) and torch.equal(gr,torch.cuda.get_rng_state(0))
    base.checkpoint(entry/'predictions.pt',outputs)
    # 同一GPU进程从磁盘重建模型并逐位重放全部预测，不改精度、不只依赖CPU近似。
    replay=make_model().cuda();replay.load_state_dict(torch.load(entry/'epoch020.pt',map_location='cpu')['model']);replay.eval()
    with torch.no_grad():
        for role,rows in roles.items():
            for row,saved in zip(rows,outputs[role]):
                parts=[]
                for start in range(0,len(row['seq']),128):
                    sl=slice(start,start+128);parts.append(replay(row['seq'][sl].cuda(),row['valid'][sl].cuda(),row['metadata'][sl].cuda())[:,0].sigmoid().cpu())
                actual=torch.cat(parts) if parts else torch.empty(0);assert torch.equal(actual,saved['pbg'][row['hard']])
    save(entry/'gpu_complete.json',dict(passed=True,shift=shift,mode=mode,spec=SPEC,guard=guard,guard_sha256=sha(guard),
        kernel_sha256=sha(entry/'kernel_complete.json'),input_sha256=cost['inputs_sha256'],pairs_sha256=sha(DIAG/shift/'fit_costs_and_pairs.json'),
        initial_digest=initial,final_digest=digest,parameters=sum(p.numel() for p in model.parameters()),history_sha256=sha(entry/'history.json'),
        predictions_sha256=sha(entry/'predictions.pt'),checkpoint_sha256=sha(entry/'epoch020.pt'),all_saved_predictions_disk_GPU_bitwise_equal=True,
        readout_parameter_and_RNG_unchanged=True,seconds=time.monotonic()-began))


def evaluate(entry):
    import torch,pandas as pd
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,background_mask
    from tools.audit_background_ap_cost import IndependentAP,selected_indices
    from tools.audit_background_fit_gap import independent_auc
    from tools.run_foreground_calibration_oracle import load_truth
    shift,mode=entry.name.split('_');gpu=read(entry/'gpu_complete.json');assert sha(entry/'predictions.pt')==gpu['predictions_sha256']
    outputs=torch.load(entry/'predictions.pt',map_location='cpu');roles=torch.load(DIAG/shift/'inputs.pt',map_location='cpu')
    split=read(OLD/shift/'split.json');gt=read(SOURCE/shift/'source_truth.json');cfg=read(SOURCE/shift/'config.json')
    truth={r:pd.DataFrame([(v,float(b[0]),float(b[1]),int(c)) for v in split[r] for b,c in zip(gt[v]['segments'],gt[v]['labels'])],columns=['video-id','t-start','t-end','label']) for r in ('fit','heldout')}
    truth['validation']=load_truth(cfg['dataset']['json_file'],cfg['src_val_split']);report={}
    for role,rows in outputs.items():
        for row,original in zip(rows,roles[role]):
            assert row['video']==original['video']
            for k in ('scores','labels','segments'):assert torch.equal(row[k],original[k])
            assert (row['pbg'][~original['hard']]==0).all()
        f=prediction_frame(rows);bg=background_mask(truth[role],f);engines={};refs={};positions={};base=[]
        for c in range(10):
            idx=np.flatnonzero(f.label.to_numpy()==c);positions[c]=idx;engines[c]=ClassAP(truth[role][truth[role].label==c],f.iloc[idx]);refs[c]=IndependentAP(truth[role][truth[role].label==c],f.iloc[idx].reset_index(drop=True))
            ap=engines[c].evaluate(np.ones(len(idx),bool))['ap'];assert np.allclose(ap,refs[c].ap(np.ones(len(idx),bool)),atol=1e-7,rtol=0);base.append(ap)
        report[role]={}
        for method in (mode,'confidence'):
            auc=[];budgets=[]
            for _,idx in f.groupby('video-id',sort=False).indices.items():
                idx=np.asarray(idx);idx=idx[f.iloc[idx].score.to_numpy()>=.3]
                a=independent_auc(bg[idx],1-f.iloc[idx].score.to_numpy() if method=='confidence' else f.iloc[idx].pbg.to_numpy())
                if a is not None:auc.append(a)
            for rate in SPEC['rates']:
                selected=selected_indices(f,rate,method=='confidence');keep=np.ones(len(f),bool);keep[selected]=False
                ap=[engines[c].evaluate(keep[positions[c]])['ap'] for c in range(10)];reference=[refs[c].ap(keep[positions[c]]).tolist() for c in range(10)]
                assert np.allclose(ap,reference,atol=1e-7,rtol=0)
                budgets.append(dict(rate=rate,delta_pp=float(100*(np.mean(ap)-np.mean(base))),map=float(np.mean(ap)),per_class_AP=ap,selected=selected.tolist(),BG_removed=int(bg[selected].sum()),nonBG_removed=int((~bg[selected]).sum())))
            report[role][method]=dict(baseline=float(np.mean(base)),original_per_class_AP=base,video_macro_AUC=float(np.mean(auc)) if auc else None,AUC_valid_videos=len(auc),total_videos=len(rows),budgets=budgets)
        print('EVALUATED',entry.name,role,flush=True)
    save(entry/'complete.json',dict(passed=True,source_only=True,shift=shift,mode=mode,gpu_complete_sha256=sha(entry/'gpu_complete.json'),all_original_arrays_exact=True,all_fixed_AP_numpy_numba_equal=True,results=report))


def run(root):
    from tools.baseline_gpu import snapshot
    assert read(DIAG/'complete.json')['passed'];root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(spec=SPEC,code=code_identity(),diagnostic_sha256=sha(DIAG/'complete.json')))
    active={};done=[]
    def stop(sig,frame):
        save(root/'stop.json',dict(signal=sig,automatic_restart_authorized=False))
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for s in (signal.SIGTERM,signal.SIGINT):signal.signal(s,stop)
    try:
        for stage in ('train','evaluate'):
            pending=list(TASKS)
            while pending or active:
                for task,t in list(active.items()):
                    if t['p'].poll() is None:continue
                    t['log'].close();del active[task]
                    if t['p'].returncode:raise RuntimeError(task+'/'+stage+' failed')
                    if (root/task/'budget_pause.json').exists():raise RuntimeError(task+' scheduling pause, not scientific failure')
                    done.append(task+'/'+stage)
                devices=[None]*max(0,2-len(active)) if stage=='evaluate' else [g for g in (0,1,2,3) if g not in [t['gpu'] for t in active.values()] and snapshot(g)['free_mib']>=12288]
                for g in devices:
                    if not pending:break
                    task=pending.pop(0);entry=root/task;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONHASHSEED='123')
                    if g is not None:
                        observed=snapshot(g);lease=entry/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=g,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
                        env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
                    log=(entry/(stage+'.log')).open('w');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--entry',str(entry),'--worker',stage],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);active[task]=dict(p=p,gpu=g,log=log)
                save(root/'state.json',dict(stage=stage,owner_pid=os.getpid(),completed=done,pending=pending,active={k:dict(pid=t['p'].pid,gpu=t['gpu']) for k,t in active.items()}));time.sleep(3)
        for s in SPEC['shifts']:
            a=read(root/(s+'_control')/'gpu_complete.json');b=read(root/(s+'_pairs')/'gpu_complete.json');assert a['initial_digest']==b['initial_digest']
        save(root/'final.json',dict(passed=True,spec=SPEC,source_only=True,results={t:read(root/t/'complete.json') for t in TASKS},complete_sha256={t:sha(root/t/'complete.json') for t in TASKS}))
        save(root/'state.json',dict(stage='complete',owner_pid=os.getpid(),active={}))
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
