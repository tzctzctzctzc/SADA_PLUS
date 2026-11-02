"""四个小头源预检共享原Prototype和提取；严格只过滤原score>=.3的纯背景。"""
import argparse,os,sys,time,subprocess,signal
from pathlib import Path
from copy import deepcopy
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.hard_background_modules import MODES
SOURCE=Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
OLD=Path('/vepfs/vlms/proposal_foreground_source_pilot6_eb30065_1103')
SHIFTS=tuple('S%d'%i for i in range(1,7))
SPEC=dict(schema='prototype-high-score-strict-background-source-pilot-v1',modes=list(MODES),
    cutoff=.3,background='any_task_class_max_tIoU<.01',fit='all_source_train_real_postNMS_candidates_score>=.3_only',
    epochs=20,seed=123,head_seed=311104,lr=3e-4,wd=.05,clip=1.,
    losses=dict(quality=.25,rank=.25,context=.1,perturb_supervised=.25,perturb_consistency=.1),
    jitter=.1,source_gt_only_for_head_fitting=True,target_gt='CPU_target_train_threshold_calibration_only',
    detector_frozen=True,original_Prototype_source_best_EMA=True,formal_joint_training=False,
    filtering='postNMS_delete_only_score>=.3; scores_labels_endpoints_of_retained_predictions_unchanged',
    threshold_objective='complete_train_AP_.1_to_.5_max_no_TP_constraint_global_and_per_class',
    new_mechanisms_are_hypotheses_not_claimed_novel=True)


def kernel(entry):
    import torch
    from tools import baseline_engine as base
    from tools.hard_background_modules import HardBackgroundHead,targets,step_loss
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);parent=SOURCE/entry.name;truth=read(parent/'source_truth.json')
    chosen=None
    for record in read(parent/'cache_index.json')['records']:
        assert sha(record['path'])==record['sha256'];cpu=torch.load(record['path'],map_location='cpu');hard=cpu['scores']>=SPEC['cutoff']
        if not hard.any():continue
        gt=torch.tensor(truth[record['video']]['segments'],dtype=torch.float32).reshape(-1,2);gl=torch.tensor(truth[record['video']]['labels'],dtype=torch.long)
        labels=cpu['labels'][hard];bg,q=targets(cpu['boxes'][hard],labels,gt,gl)
        if not any(((bg==0)&(labels==c)).any() and ((bg==1)&(labels==c)).any() for c in labels.unique()):continue
        chosen=dict(feature=cpu['feature'],valid=cpu['valid'],boxes=cpu['boxes'][hard],scores=cpu['scores'][hard],labels=labels,bg=bg,quality=q,truth=gt,truth_labels=gl)
        video=record['video'];break
    assert chosen is not None
    item={k:v.cuda() for k,v in chosen.items()};reports={};seed=HardBackgroundHead().cuda();initial=deepcopy(seed.state_dict())
    for mode in MODES:
        head=HardBackgroundHead().cuda();head.load_state_dict(initial);opt=torch.optim.AdamW(head.parameters(),lr=SPEC['lr'],weight_decay=SPEC['wd'])
        for _ in range(2):
            opt.zero_grad(set_to_none=True);loss,stats=step_loss(head,item,mode,torch.full_like(item['boxes'],.05))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),1.);assert torch.isfinite(norm) and norm>0;opt.step()
        path=entry/(mode+'_kernel.pt');base.checkpoint(path,dict(head=head.state_dict(),optimizer=opt.state_dict(),CPU=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state(0)))
        saved=torch.load(path,map_location='cpu');new=HardBackgroundHead().cuda();new.load_state_dict(saved['head'])
        no=torch.optim.AdamW(new.parameters());no.load_state_dict(saved['optimizer'])
        assert base.tensor_digest(no.state_dict())==base.tensor_digest(opt.state_dict())
        assert base.tensor_digest(new.state_dict())==base.tensor_digest(head.state_dict())!=base.tensor_digest(initial)
        torch.set_rng_state(saved['CPU']);torch.cuda.set_rng_state(saved['CUDA'],0)
        assert torch.equal(torch.get_rng_state(),saved['CPU']) and torch.equal(torch.cuda.get_rng_state(0),saved['CUDA'])
        reports[mode]=dict(gradient_norm=float(norm),mechanisms=stats,state_sha256=sha(path))
    # 新共享native前向也必须真实验收，不能等源头训完才发现目标读出接口错。
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.proposal_foreground_math import seconds_to_grid
    cfg=read(parent/'config.json');model=load_detector(parent);before=base.tensor_digest(model.state_dict());captured=[]
    handle=model.cls_head_target.register_forward_pre_hook(lambda module,args:captured.append(args))
    ds=base.dataset(cfg,parent/'target_train_unlabeled.json',cfg['tgt_train_split'],False,True)
    v=base.move_to_device([ds[0]],'cuda:0');assert not v[0]['segments'].numel()
    with torch.no_grad():result=model(video_list_src=v,video_list_tgt=v,mode='test',domain='target',curr_epoch=0)[1][0]
    assert len(captured)==3;f,masks=captured[1]
    row=dict(video=v[0]['metadata']['video_id'],feature=f[0][0].detach(),valid=masks[0][0,0].bool(),
        boxes=seconds_to_grid(result['segments'],v[0]['metadata']).cuda(),segments=result['segments'],scores=result['scores'],labels=result['labels'])
    out=predict(head,row);assert (out['pbg'][out['scores']<.3]==0).all()
    assert all(torch.equal(out[k],result[k]) for k in ['segments','scores','labels'])
    handle.remove();assert before==base.tensor_digest(model.state_dict())
    save(entry/'kernel_complete.json',dict(passed=True,video=video,guard=guard,guard_sha256=sha(guard),reports=reports,
        all_four_same_initialization=True,all_four_head_optimizer_CPU_CUDA_RNG_restore=True,
        original_native_target_GT_free_and_detector_unchanged=True,low_score_candidates_unchanged=True,
        peak_mib=torch.cuda.max_memory_reserved()/2**20,not_formal_training=True))


def predict(head,item):
    import torch
    hard=item['scores']>=SPEC['cutoff'];bg=torch.zeros(len(hard));quality=torch.zeros(len(hard))
    if hard.any():
        with torch.no_grad():
            out=head(item['feature'],item['valid'],item['boxes'][hard.to(item['boxes'].device)],
                     item['labels'][hard].to(item['feature'].device),item['scores'][hard].to(item['feature'].device)).sigmoid().cpu()
        bg[hard]=out[:,0];quality[hard]=out[:,1]
    return dict(video=item['video'],segments=item['segments'],scores=item['scores'],labels=item['labels'],pbg=bg,quality=quality)


def gpu(entry):
    import gc,torch,numpy as np,random
    from tools import baseline_engine as base
    from tools.hard_background_modules import HardBackgroundHead,targets,step_loss
    from tools.proposal_foreground_math import seconds_to_grid
    from tools.run_foreground_calibration_oracle import load_detector
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);s=entry.name;parent=SOURCE/s;cfg=read(parent/'config.json')
    fitdone=read(parent/'fit_complete.json');index=read(parent/'cache_index.json');truth=read(parent/'source_truth.json')
    selection=read(parent/'selection.json');assert set(truth)==set(selection['source_fit'])
    assert sha(parent/'cache_index.json')==fitdone['artifacts'][str(parent/'cache_index.json')]
    fit=[];skipped=[]
    for record in index['records']:
        assert sha(record['path'])==record['sha256'];item=torch.load(record['path'],map_location='cpu')
        hard=item['scores']>=SPEC['cutoff']
        if not hard.any():skipped.append(record['video']);continue
        gt=torch.tensor(truth[record['video']]['segments'],dtype=torch.float32).reshape(-1,2)
        gl=torch.tensor(truth[record['video']]['labels'],dtype=torch.long)
        bg,q=targets(item['boxes'][hard],item['labels'][hard],gt,gl)
        fit.append(dict(video=record['video'],feature=item['feature'],valid=item['valid'],boxes=item['boxes'][hard],
            labels=item['labels'][hard],scores=item['scores'][hard],bg=bg,quality=q,truth=gt,truth_labels=gl))
    assert fit
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(SPEC['head_seed']);initial=HardBackgroundHead().cuda()
    heads={mode:deepcopy(initial) for mode in MODES};del initial
    initial_sha=base.tensor_digest(heads[MODES[0]].state_dict())
    assert all(base.tensor_digest(h.state_dict())==initial_sha for h in heads.values())
    opts={mode:torch.optim.AdamW(heads[mode].parameters(),lr=SPEC['lr'],weight_decay=SPEC['wd']) for mode in MODES}
    order_gen=torch.Generator().manual_seed(SPEC['seed']);jitter_gen=torch.Generator().manual_seed(911104)
    history=[];torch.cuda.reset_peak_memory_stats();first_grad={}
    for epoch in range(1,21):
        totals={m:0. for m in MODES};stats={m:dict(rank=0.,counterfactual=0.,consistency=0.,changed_labels=0) for m in MODES};start=time.time()
        for idx in torch.randperm(len(fit),generator=order_gen).tolist():
            cpu=fit[idx];item={k:v.cuda() if torch.is_tensor(v) else v for k,v in cpu.items()}
            jitter=(torch.rand((len(item['boxes']),2),generator=jitter_gen)*2-1).cuda()*SPEC['jitter']
            for mode in MODES:
                head=heads[mode];head.train();opt=opts[mode];opt.zero_grad(set_to_none=True)
                loss,observed=step_loss(head,item,mode,jitter if mode=='perturb' else None)
                assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(head.parameters(),SPEC['clip'])
                assert torch.isfinite(norm) and norm>0;opt.step();totals[mode]+=float(loss.detach())
                if mode not in first_grad:first_grad[mode]=float(norm)
                for k in stats[mode]:stats[mode][k]+=observed[k]
        row=dict(epoch=epoch,seconds=time.time()-start,loss={m:totals[m]/len(fit) for m in MODES},mechanisms=stats)
        history.append(row);save(entry/'history.json',history)
        base.checkpoint(entry/'latest_heads.pt',dict(heads={m:h.state_dict() for m,h in heads.items()},
            optimizers={m:o.state_dict() for m,o in opts.items()},order_rng=order_gen.get_state(),jitter_rng=jitter_gen.get_state(),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),
            history=history,spec=SPEC,index_sha256=sha(parent/'cache_index.json')))
        print('HEADS_EPOCH',s,epoch,row['loss'],flush=True)
    assert all(base.tensor_digest(h.state_dict())!=initial_sha for h in heads.values())
    os.link(entry/'latest_heads.pt',entry/'fixed20_heads.pt')
    for h in heads.values():h.eval()
    after_fit={m:base.tensor_digest(h.state_dict()) for m,h in heads.items()}
    source_index=index['records'];old_index=read(OLD/s/'cache_index.json')
    assert old_index['source_initial_state_digest']==index['detector_digest']
    val_index=[r for r in old_index['records'] if r['role']=='validation']
    assert not {r['video'] for r in val_index}&set(selection['source_fit'])
    for role,records in [('source_train',source_index),('source_val',val_index)]:
        rows={m:[] for m in MODES}
        for record in records:
            assert sha(record['path'])==record['sha256'];cpu=torch.load(record['path'],map_location='cpu')
            item=dict(cpu,video=record['video'],feature=cpu['feature'].cuda(),valid=cpu['valid'].cuda(),boxes=cpu['boxes'].cuda())
            for m,h in heads.items():rows[m].append(predict(h,item))
        base.checkpoint(entry/(role+'_predictions.pt'),rows)
    # 四个头共用一次原始native完整前向；捕获实际target FPN，避免另写近似检测路径。
    model=load_detector(parent);detector_before=base.tensor_digest(model.state_dict());capture=[]
    handle=model.cls_head_target.register_forward_pre_hook(lambda module,args:capture.append(args))
    ckpt=torch.load(Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')/s/'best.pt',map_location='cpu');epoch=ckpt['epoch'];del ckpt
    for role,split in [('target_train',cfg['tgt_train_split']),('target_val',cfg['tgt_val_split'])]:
        manifest=parent/(role+'_unlabeled.json');assert all('annotations' not in v for v in read(manifest)['database'].values())
        ds=base.dataset(cfg,manifest,split,False,True);rows={m:[] for m in MODES}
        for i in range(len(ds)):
            video=base.move_to_device([ds[i]],'cuda:0');assert not video[0]['segments'].numel() and not video[0]['labels'].numel()
            capture.clear()
            with torch.no_grad():result=model(video_list_src=video,video_list_tgt=video,mode='test',domain='target',curr_epoch=epoch)[1][0]
            assert len(capture)==3;f,masks=capture[1];boxes=seconds_to_grid(result['segments'],video[0]['metadata']).cuda()
            item=dict(video=video[0]['metadata']['video_id'],feature=f[0][0].detach(),valid=masks[0][0,0].bool(),boxes=boxes,
                segments=result['segments'].cpu(),scores=result['scores'].cpu(),labels=result['labels'].cpu())
            for m,h in heads.items():rows[m].append(predict(h,item))
            print('NATIVE_SHARED',s,role,i+1,len(ds),flush=True)
        assert all(sorted(r['video'] for r in rows[m])==selection[role] for m in MODES)
        base.checkpoint(entry/(role+'_predictions.pt'),rows)
    handle.remove();assert base.tensor_digest(model.state_dict())==detector_before
    assert all(after_fit[m]==base.tensor_digest(h.state_dict()) for m,h in heads.items())
    save(entry/'gpu_complete.json',dict(code=code_identity(),spec=SPEC,guard=guard,guard_sha256=sha(guard),source_videos=len(fit),
        empty_high_score_videos=skipped,source_members=selection['source_fit'],source_index_sha256=sha(parent/'cache_index.json'),
        source_truth_sha256=sha(parent/'source_truth.json'),initial_head_digest=initial_sha,final_head_digests=after_fit,
        first_gradient_norms=first_grad,head_parameters=sum(p.numel() for p in heads[MODES[0]].parameters()),detector_unchanged=True,
        optimizer_steps={m:20*len(fit) for m in MODES},peak_mib=torch.cuda.max_memory_reserved()/2**20,
        artifacts={str(p):sha(p) for p in entry.glob('*.pt')}))


def rules(frame):
    import numpy as np
    values=frame.loc[frame.score>=SPEC['cutoff'],'pbg'].to_numpy()
    return np.unique(np.r_[np.arange(0,1.001,.025),np.quantile(values,np.r_[np.linspace(0,1,51),.995,.999]),1.01]).tolist() if len(values) else [1.01]


def calibrate(gt,frame):
    import numpy as np
    from tools.foreground_oracle_ap import ClassAP
    global_rules=rules(frame);global_results=[[] for _ in global_rules];per={}
    for c in sorted(gt.label.unique()):
        f=frame[frame.label==c].reset_index(drop=True);e=ClassAP(gt[gt.label==c],f);r=rules(f)
        def mask(t):return ~((f.score.to_numpy()>=SPEC['cutoff'])&(f.pbg.to_numpy()>t))
        values=[e.evaluate(mask(t)) for t in r];idx=max(range(len(r)),key=lambda i:(values[i]['map'],values[i]['kept'],r[i]))
        per[str(int(c))]=dict(threshold=r[idx],index=idx,grid=values,rules=r)
        for i,t in enumerate(global_rules):global_results[i].append(e.evaluate(mask(t)))
    best=max(range(len(global_rules)),key=lambda i:(sum(v['map'] for v in global_results[i]),sum(v['kept'] for v in global_results[i]),global_rules[i]))
    return dict(global_threshold=global_rules[best],global_rules=global_rules,global_grid=global_results,per_class=per,
                objective='complete_train_mAP_max; ties_keep_more; no_TP_constraint')


def measure(gt,frame,cal):
    import numpy as np
    from tools.foreground_oracle_ap import ClassAP,background_mask
    bg=background_mask(gt,frame);details={};counts={k:0 for k in ['BG_removed','nonBG_removed','high_score_candidates','high_score_BG']}
    for c in sorted(gt.label.unique()):
        positions=np.flatnonzero(frame.label.to_numpy()==c);f=frame.iloc[positions].reset_index(drop=True);e=ClassAP(gt[gt.label==c],f)
        thresholds=[cal['global_threshold'],cal['per_class'].get(str(int(c)),dict(threshold=1.01))['threshold']]
        masks=[~((f.score.to_numpy()>=SPEC['cutoff'])&(f.pbg.to_numpy()>t)) for t in thresholds]
        details[str(int(c))]=dict(original=e.evaluate(np.ones(len(f),dtype=bool)),global_rule=e.evaluate(masks[0]),class_rule=e.evaluate(masks[1]))
        removed=~masks[1];b=bg[positions];counts['BG_removed']+=int((removed&b).sum());counts['nonBG_removed']+=int((removed&~b).sum())
        counts['high_score_candidates']+=int((f.score.to_numpy()>=SPEC['cutoff']).sum());counts['high_score_BG']+=int(((f.score.to_numpy()>=SPEC['cutoff'])&b).sum())
    summary={name:float(np.mean([v[name]['map'] for v in details.values()])) for name in ['original','global_rule','class_rule']}
    summary.update(global_delta_pp=100*(summary['global_rule']-summary['original']),class_delta_pp=100*(summary['class_rule']-summary['original']))
    return dict(summary=summary,classes=details,counts=counts)


def evaluate(entry):
    import torch
    from tools.foreground_oracle_ap import prediction_frame
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.background_payoff_diagnosis import distribution
    s=entry.name;parent=SOURCE/s;cfg=read(parent/'config.json');done=read(entry/'gpu_complete.json')
    for path,d in done['artifacts'].items():assert sha(path)==d
    truth={role:load_truth(path,split) for role,path,split in [
        ('source_train',cfg['dataset']['json_file'],cfg['src_train_split']),('source_val',cfg['dataset']['json_file'],cfg['src_val_split']),
        ('target_train',parent/'target_train_truth_cpu.json',cfg['tgt_train_split']),('target_val',cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])]}
    predictions={role:torch.load(entry/(role+'_predictions.pt'),map_location='cpu') for role in truth}
    for role,group in predictions.items():
        for mode in MODES:
            assert len(group[mode])==len(group[MODES[0]])
            for a,b in zip(group[mode],group[MODES[0]]):
                assert a['video']==b['video'] and all(torch.equal(a[k],b[k]) for k in ['segments','scores','labels'])
                assert (a['pbg'][a['scores']<SPEC['cutoff']]==0).all()
    results={}
    for mode in MODES:
        frames={role:prediction_frame(v[mode]) for role,v in predictions.items()}
        assert not set(frames['target_train']['video-id'])&set(frames['target_val']['video-id'])
        source_cal=calibrate(truth['source_train'],frames['source_train']);target_cal=calibrate(truth['target_train'],frames['target_train'])
        save(entry/(mode+'_calibration.json'),dict(source=source_cal,target=target_cal,parameter_selection_GT='train_only'))
        source=measure(truth['source_val'],frames['source_val'],source_cal);target=measure(truth['target_val'],frames['target_val'],target_cal)
        # AUC只在高分池统计，完整AP仍含未过滤的低分预测。
        hard=frames['target_val'][frames['target_val'].score>=SPEC['cutoff']].reset_index(drop=True)
        dist,_=distribution(hard,truth['target_val']) if len(hard) else ({},None)
        results[mode]=dict(source_val=source,target_val=target,target_high_score_diagnostics=dist)
        print('EVALUATED',s,mode,source['summary'],target['summary'],flush=True)
    save(entry/'complete.json',dict(code=code_identity(),spec=SPEC,shift=s,results=results,
        target_train_calibrated_Oracle_not_UDA=True,source_only_frozen_readout=True,
        artifacts={str(p):sha(p) for p in list(entry.glob('*.pt'))+list(entry.glob('*calibration.json'))+[entry/'gpu_complete.json',entry/'history.json']}))


def run(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    root.mkdir(parents=True,exist_ok=False);save(root/'plan.json',dict(code=code_identity(),spec=SPEC,shifts=SHIFTS,base_predictions=SOURCE.as_posix()))
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=list(SHIFTS);probe_pending=list(SHIFTS);probes=[];active={};cpu=[];done=[]
    def stop(sig,frame):
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        raise SystemExit(128+sig)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def spawn(s,stage,gpu=None):
        e=root/s;e.mkdir(exist_ok=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
        if gpu is not None:
            observed=snapshot(gpu);lease=e/(stage+'_lease.json');save(lease,dict(owner_pid=os.getpid(),index=gpu,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
        log=(e/(stage+'.log')).open('a');p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(e)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[s]=dict(p=p,stage=stage,gpu=gpu,log=log)
    try:
        while pending or probe_pending or active or cpu:
            for s,t in list(active.items()):
                if t['p'].poll() is None:continue
                t['log'].close();del active[s]
                if t['p'].returncode:raise RuntimeError(s+'/'+t['stage']+' failed')
                if t['stage']=='kernel':
                    assert read(root/s/'kernel_complete.json')['passed'];probes.append(s)
                elif t['stage']=='gpu':cpu.append(s)
                else:done.append(s)
            used={t['gpu'] for t in active.values()}
            for g in [2,0,3,1]:
                if g in used or snapshot(g)['free_mib']<12288:continue
                if probe_pending:spawn(probe_pending.pop(0),'kernel',g)
                elif len(probes)==6 and pending:spawn(pending.pop(0),'gpu',g)
            if cpu and not any(t['stage']=='evaluate' for t in active.values()):spawn(cpu.pop(0),'evaluate')
            save(root/'state.json',dict(pid=os.getpid(),pending=pending,kernel_pending=probe_pending,kernels=probes,cpu_pending=cpu,completed=done,
                active={s:dict(pid=t['p'].pid,stage=t['stage'],gpu=t['gpu']) for s,t in active.items()}))
            if pending or cpu or active:time.sleep(3)
        rows=[read(root/s/'complete.json') for s in SHIFTS]
        save(root/'final.json',dict(count=24,unique_shifts=6,results=rows,formal_UDA=False,means={m:{domain:sum(r['results'][m][domain]['summary']['class_delta_pp'] for r in rows)/6 for domain in ['source_val','target_val']} for m in MODES}))
    except Exception as e:save(root/'error.json',dict(error=repr(e)));raise
    finally:
        for t in active.values():
            if t['p'].poll() is None:os.killpg(t['p'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--worker',choices=['kernel','gpu','evaluate']);p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.run:run(a.run)
    else:
        os.chdir('/vepfs/vlms/SADA_PLUS');dict(kernel=kernel,gpu=gpu,evaluate=evaluate)[a.worker](a.entry)
