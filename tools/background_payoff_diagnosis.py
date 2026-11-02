"""背景小头收益诊断：冻结预测/源读出，细网格、软重排及Oracle分解。"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_foreground_calibration_oracle import SHIFTS
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
CAPACITY=Path('/vepfs/vlms/foreground_calibration_capacity6_e4c7032_1103')
SPEC=dict(source_readout='all_original_source_fit_videos_frozen_head_no_optimizer',
    fine_thresholds='original_grid_union_51_target_train_quantiles_plus_0.001_0.005_0.995_0.999_0.9999_tails',
    protection=[.1,.3,.5,.7,1.01],soft_lambdas=[0.,.25,.5,1.,2.,4.],
    soft_score='original_score_times_clip(1-pbg,1e-8,1)**lambda',
    choices='target_train_AP_max_global_and_perclass; ties_keep_more_or_lambda_smaller',
    validation_Oracle='CPU_only_perclass_fine_val_threshold_upper_readout; never_used_as_deploy_rule',
    error_splits='learned_delete_BG_only_vs_nonBG_only; nonadditive_AP_interventions',
    score_bins=[0,.01,.05,.1,.3,.5,1.01],
    scope='target_informed_exploratory_diagnosis_not_formal_UDA_no_detector_training')


def auc(y,score):
    import numpy as np
    from scipy.stats import rankdata
    y=np.asarray(y,dtype=bool);score=np.asarray(score,dtype=float)
    p=int(y.sum());n=len(y)-p
    if not p or not n:return None
    return float((rankdata(score)[y].sum()-p*(p+1)/2)/(p*n))


def fine_rules(frame):
    import numpy as np
    values=frame.pbg.to_numpy(dtype=float)
    quantiles=np.unique(np.r_[np.linspace(0,1,51),.001,.005,.995,.999,.9999])
    thresholds=np.unique(np.r_[np.arange(0,1.001,.025),np.quantile(values,quantiles),1.01]) if len(values) else np.array([1.01])
    return [(float(t),float(p)) for t in thresholds for p in SPEC['protection']]


def summarize_ap(engine,keep,scores=None):
    import numpy as np
    from tools.foreground_oracle_ap import _matched_ap,TIOUS
    if scores is None:return engine.evaluate(keep)
    indices=np.flatnonzero(keep);order=indices[np.asarray(scores)[indices].argsort()[::-1]]
    ap,tp=_matched_ap(order,engine.pointers,engine.matches,engine.overlaps,engine.n_gt,TIOUS)
    return dict(map=float(ap.mean()),ap=ap.tolist(),tp=tp.astype(int).tolist(),kept=int(keep.sum()),gt=engine.n_gt)


def pick(results,rules):
    return max(range(len(results)),key=lambda i:(results[i]['map'],results[i]['kept'],rules[i][0],-rules[i][1]))


def macro(results):
    import numpy as np
    return float(np.mean([r['map'] for r in results]))


def distribution(frame,truth):
    import numpy as np
    from tools.foreground_oracle_ap import background_mask
    bg=background_mask(truth,frame);rows=[]
    for lower in [0,.1,.3,.5]:
        keep=frame.score.to_numpy()>=lower;video_aucs=[];video_base=[]
        for vid,idx in frame[keep].groupby('video-id').indices.items():
            positions=np.flatnonzero(keep)[idx]
            a=auc(bg[positions],frame.pbg.to_numpy()[positions]);b=auc(bg[positions],-frame.score.to_numpy()[positions])
            if a is not None:video_aucs.append(a);video_base.append(b)
        rows.append(dict(min_score=lower,candidates=int(keep.sum()),background=int(bg[keep].sum()),
            pooled_head_auc=auc(bg[keep],frame.pbg.to_numpy()[keep]),
            pooled_negative_confidence_auc=auc(bg[keep],-frame.score.to_numpy()[keep]),
            video_macro_head_auc=float(np.mean(video_aucs)) if video_aucs else None,
            video_macro_negative_confidence_auc=float(np.mean(video_base)) if video_base else None,
            videos_with_both_types=len(video_aucs)))
    return dict(auc_by_score=rows,pbg_quantiles=np.quantile(frame.pbg,[0,.01,.1,.5,.9,.95,.975,.99,.999,1]).tolist(),
                candidates=len(frame),background=int(bg.sum())),bg


def source_gpu(entry):
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import ProposalForeground
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);s=entry.name;parent=SOURCE/s
    complete=read(parent/'fit_complete.json');head_path=parent/'head_fixed20.pt'
    assert sha(head_path)==complete['artifacts'][str(head_path)]
    index=read(parent/'cache_index.json');selection=read(parent/'selection.json')
    assert {r['video'] for r in index['records']}==set(selection['source_fit'])
    head=ProposalForeground(512).cuda();head.load_state_dict(torch.load(head_path,map_location='cpu')['state_dict']);head.eval()
    initial=base.tensor_digest(head.state_dict());rows=[];torch.cuda.reset_peak_memory_stats()
    for record in index['records']:
        assert sha(record['path'])==record['sha256']
        item=torch.load(record['path'],map_location='cpu')
        with torch.no_grad():
            probability=head(item['feature'].cuda(),item['valid'].cuda(),item['boxes'].cuda()).softmax(-1)[:,0].cpu()
        rows.append(dict(video=record['video'],segments=item['segments'],labels=item['labels'],scores=item['scores'],pbg=probability))
    assert initial==base.tensor_digest(head.state_dict())
    base.checkpoint(entry/'source_predictions.pt',rows)
    save(entry/'source_gpu_complete.json',dict(guard=guard,guard_sha256=sha(guard),head_unchanged=True,optimizer_steps=0,
        source_videos=len(rows),source_index_sha256=sha(parent/'cache_index.json'),head_sha256=sha(head_path),
        predictions_sha256=sha(entry/'source_predictions.pt'),peak_mib=torch.cuda.max_memory_reserved()/2**20))


def analyze(entry):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import ClassAP,prediction_frame,keep_rule
    from tools.run_foreground_calibration_oracle import load_truth
    s=entry.name;parent=SOURCE/s;cfg=read(parent/'config.json')
    train_path=parent/'target_train_predictions.pt';val_path=parent/'target_val_predictions.pt'
    assert sha(train_path)==read(parent/'target_train_gpu_complete.json')['predictions_sha256']
    assert sha(val_path)==read(parent/'target_val_gpu_complete.json')['predictions_sha256']
    train=prediction_frame(torch.load(train_path,map_location='cpu'));val=prediction_frame(torch.load(val_path,map_location='cpu'))
    train_gt=load_truth(parent/'target_train_truth_cpu.json',cfg['tgt_train_split'])
    val_gt=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])
    assert not set(train['video-id'])&set(val['video-id'])
    train_dist,train_bg=distribution(train,train_gt);val_dist,val_bg=distribution(val,val_gt)
    source_done=read(entry/'source_gpu_complete.json');assert sha(entry/'source_predictions.pt')==source_done['predictions_sha256']
    source=prediction_frame(torch.load(entry/'source_predictions.pt',map_location='cpu'))
    source_gt=load_truth(cfg['dataset']['json_file'],cfg['src_train_split'])
    source_dist,_=distribution(source,source_gt)
    classes=sorted(int(c) for c in train_gt.label.unique());detail={}
    global_rules=fine_rules(train);global_train={i:[] for i in range(len(global_rules))}
    soft_train_global={v:[] for v in SPEC['soft_lambdas']}
    frozen=read(CAPACITY/s/'unconstrained_frozen.json')
    for c in classes:
        pos_train=np.flatnonzero(train.label.to_numpy()==c);pos_val=np.flatnonzero(val.label.to_numpy()==c)
        a=train.iloc[pos_train].reset_index(drop=True);b=val.iloc[pos_val].reset_index(drop=True)
        et=ClassAP(train_gt[train_gt.label==c],a);ev=ClassAP(val_gt[val_gt.label==c],b)
        all_t=np.ones(len(a),dtype=bool);all_v=np.ones(len(b),dtype=bool)
        baseline_t=et.evaluate(all_t);baseline_v=ev.evaluate(all_v)
        for i,rule in enumerate(global_rules):global_train[i].append(et.evaluate(keep_rule(a,rule)))
        rules=fine_rules(a);results=[et.evaluate(keep_rule(a,rule)) for rule in rules];chosen=pick(results,rules)
        selected_rule=rules[chosen];fine_val=ev.evaluate(keep_rule(b,selected_rule))
        soft=[]
        for lam in SPEC['soft_lambdas']:
            ts=a.score.to_numpy()*np.clip(1-a.pbg.to_numpy(),1e-8,1)**lam
            vs=b.score.to_numpy()*np.clip(1-b.pbg.to_numpy(),1e-8,1)**lam
            tr=summarize_ap(et,all_t,ts);vr=summarize_ap(ev,all_v,vs)
            soft.append(dict(lam=lam,train=tr,val=vr));soft_train_global[lam].append(tr)
        soft_index=max(range(len(soft)),key=lambda i:(soft[i]['train']['map'],-soft[i]['lam']))
        # 仅界定当前头排序的离线读出容量，val GT选的点绝不作为部署参数。
        vrules=fine_rules(b);vresults=[ev.evaluate(keep_rule(b,r)) for r in vrules];vi=pick(vresults,vrules)
        selected_old=frozen['per_class'][str(c)]['rule'];old_keep=keep_rule(b,selected_old);bg=val_bg[pos_val]
        bg_only=ev.evaluate(old_keep|~bg);nonbg_only=ev.evaluate(old_keep|bg)
        oracle=ev.evaluate(~bg)
        bins=[]
        for lo,hi in zip(SPEC['score_bins'][:-1],SPEC['score_bins'][1:]):
            mask=(b.score.to_numpy()>=lo)&(b.score.to_numpy()<hi)&bg
            bins.append(dict(low=lo,high=hi,background=int(mask.sum()),result=ev.evaluate(~mask)))
        detail[str(c)]=dict(baseline_train=baseline_t,baseline_val=baseline_v,fine_train_grid=results,fine_rules=rules,
            fine_selected_rule=selected_rule,fine_selected_train=results[chosen],fine_selected_val=fine_val,
            soft_grid=soft,soft_selected_lambda=soft[soft_index]['lam'],soft_selected_val=soft[soft_index]['val'],
            validation_threshold_Oracle_only=vresults[vi],validation_threshold_Oracle_rule=vrules[vi],
            old_rule=selected_old,old_selected_val=ev.evaluate(old_keep),old_BG_only_delete=bg_only,
            old_nonBG_only_delete=nonbg_only,perfect_background_Oracle=oracle,background_score_bin_oracles=bins)
        print('CLASS_ANALYZED',s,c,flush=True)
    common=max(global_train,key=lambda i:(macro(global_train[i]),sum(r['kept'] for r in global_train[i]),global_rules[i][0],-global_rules[i][1]))
    soft_global=max(soft_train_global,key=lambda l:(macro(soft_train_global[l]),-l))
    global_val=[];soft_global_val=[]
    for c in classes:
        b=val[val.label==c].reset_index(drop=True);ev=ClassAP(val_gt[val_gt.label==c],b)
        global_val.append(ev.evaluate(keep_rule(b,global_rules[common])))
        soft_global_val.append(summarize_ap(ev,np.ones(len(b),dtype=bool),b.score.to_numpy()*np.clip(1-b.pbg.to_numpy(),1e-8,1)**soft_global))
    baseline=macro([d['baseline_val'] for d in detail.values()])
    fields=['fine_selected_val','soft_selected_val','validation_threshold_Oracle_only','old_selected_val',
            'old_BG_only_delete','old_nonBG_only_delete','perfect_background_Oracle']
    summary={name:dict(map=macro([d[name] for d in detail.values()]),
        delta_pp=100*(macro([d[name] for d in detail.values()])-baseline)) for name in fields}
    summary['fine_global_val']=dict(map=macro(global_val),delta_pp=100*(macro(global_val)-baseline),rule=global_rules[common])
    summary['soft_global_val']=dict(map=macro(soft_global_val),delta_pp=100*(macro(soft_global_val)-baseline),lam=soft_global)
    bin_summary=[]
    for i,(lo,hi) in enumerate(zip(SPEC['score_bins'][:-1],SPEC['score_bins'][1:])):
        bin_summary.append(dict(low=lo,high=hi,background=sum(d['background_score_bin_oracles'][i]['background'] for d in detail.values()),
            delta_pp=100*(macro([d['background_score_bin_oracles'][i]['result'] for d in detail.values()])-baseline)))
    save(entry/'analysis.json',dict(shift=s,spec=SPEC,baseline_map=baseline,summary=summary,source=source_dist,
        target_train=train_dist,target_val=val_dist,classes=detail,background_bin_oracles=bin_summary,
        fine_global_train_grid=dict(rules=global_rules,results={str(i):v for i,v in global_train.items()}),
        target_train_selected_before_validation_scoring=True,
        validation_oracle_is_target_informed_CPU_development_not_formal_UDA=True))
    save(entry/'complete.json',dict(shift=s,code=code_identity(),spec=SPEC,summary=summary,
        artifacts={str(p):sha(p) for p in [entry/'analysis.json',entry/'source_predictions.pt',entry/'source_gpu_complete.json',
            train_path,val_path,parent/'target_train_truth_cpu.json',CAPACITY/s/'unconstrained_frozen.json',parent/'head_fixed20.pt']}))


def controller(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(code=code_identity(),spec=SPEC,source=str(SOURCE),
        previous_validation_results_already_seen=True,interpretation='target-informed_exploratory_diagnosis'))
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=list(SHIFTS);cpu=[];active={};done=[]
    def stop(sig,frame):
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def spawn(s,stage,gpu=None):
        entry=root/s;entry.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        if gpu is not None:
            obs=snapshot(gpu);lease=entry/'lease.json';save(lease,dict(owner_pid=os.getpid(),index=gpu,uuid=obs['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env.update(CUDA_VISIBLE_DEVICES=obs['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
        log=(entry/(stage+'.log')).open('a')
        p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(entry)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[s]=dict(p=p,stage=stage,gpu=gpu,log=log)
    try:
        while pending or cpu or active:
            for s,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[s]
                if t['p'].returncode:raise RuntimeError(s+'/'+t['stage']+' failed')
                if t['stage']=='source_gpu':cpu.append(s)
                else:done.append(s)
            used={t['gpu'] for t in active.values()}
            for gpu in (2,0,3,1):
                if pending and gpu not in used and snapshot(gpu)['free_mib']>=12288:spawn(pending.pop(0),'source_gpu',gpu)
            # 分析CPU最多一路，与原a4完整AP共享不超过2；旧capacity已全部结束。
            if cpu and not any(t['stage']=='analyze' for t in active.values()):spawn(cpu.pop(0),'analyze')
            save(root/'state.json',dict(pid=os.getpid(),completed=done,pending=pending,cpu_pending=cpu,
                active={s:dict(pid=t['p'].pid,stage=t['stage'],gpu=t['gpu']) for s,t in active.items()}))
            if pending or cpu or active:time.sleep(3)
        rows=[read(root/s/'complete.json') for s in SHIFTS]
        save(root/'final.json',dict(count=6,results=rows,mean_delta_pp={k:sum(r['summary'][k]['delta_pp'] for r in rows)/6 for k in rows[0]['summary']},
                                  target_informed_exploration=True,no_detector_training=True))
    except Exception as e:
        save(root/'error.json',dict(error=repr(e)));raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--worker',choices=['source_gpu','analyze']);p.add_argument('--entry',type=Path)
    a=p.parse_args()
    if a.run:controller(a.run)
    else:
        os.chdir('/vepfs/vlms/SADA_PLUS')
        (source_gpu if a.worker=='source_gpu' else analyze)(a.entry)
