"""原Prototype冻结源读出预检；固定12 fit/4 cal视频，完整source_val，绝非联合训练。"""
import argparse
from copy import deepcopy
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
from tools.proposal_foreground import split_source
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
SHIFTS=tuple('S%d'%i for i in range(1,7))
SPEC=dict(seed=123,head_seed=311103,fit_videos=12,calibration_videos=4,validation='all_source_val_videos',
    epochs=20,lr=3e-4,weight_decay=.05,clip=1.,selection='fixed20_no_val_selection',
    frozen_detector='original_Prototype_source_selected_EMA_read_only',head_transferred_to_formal=False,
    fitting='full_source_video_original_postNMS_proposals_plus_GT_two_fixed_neighbors',
    testing='postNMS_delete_only; learned_vs_equal_count_confidence_vs_GT_background_Oracle',
    target_access=False,formal_training=False)


def pick(ids,count,role):
    return sorted(sorted(ids,key=lambda v:hashlib.sha256(('proposal_fg_pilot/'+role+'/'+v).encode()).hexdigest())[:count])


def full_candidates(boxes,gt,low,high):
    import torch
    visible=gt[(gt[:,0]>=low)&(gt[:,1]<=high)]
    duration=visible[:,1]-visible[:,0]
    result=torch.cat([boxes,visible,(visible+.2*duration[:,None]).clamp(low,high),
                      (visible-.2*duration[:,None]).clamp(low,high)])
    return result[result[:,1]>result[:,0]].detach()


def prepare(root):
    from tools import baseline_engine as base
    root.mkdir(parents=True,exist_ok=False);identity=code_identity()
    jobs={}
    for s in SHIFTS:
        cfg=read(REFERENCE/s/'config.json')
        train=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
        val=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
        roles=split_source([v['id'] for v in train.data_list])
        selection=dict(fit=pick(roles['fit'],12,'fit'),calibration=pick(roles['calibration'],4,'calibration'),
                       validation=sorted(v['id'] for v in val.data_list))
        assert not set(selection['fit'])&set(selection['calibration'])
        assert not (set(selection['fit'])|set(selection['calibration']))&set(selection['validation'])
        truth={}
        for v in train.data_list+val.data_list:
            keep=(v['labels']>=0)&(v['labels']<10)
            truth[v['id']]=dict(segments=((v['segments'][keep]*v['fps']-.5*train.num_frames)/(train.feat_stride*train.downsample_rate)).tolist(),
                                labels=v['labels'][keep].tolist())
        entry=root/s;entry.mkdir()
        save(entry/'config.json',cfg);save(entry/'source_truth.json',truth)
        save(entry/'selection.json',selection)
        # GPU清单没有annotations，源标签另在CPU准备和源头监督中使用；无target数据。
        source=read(cfg['dataset']['json_file']);selected=set(sum(selection.values(),[]))
        source['database']={k:{name:value for name,value in v.items() if name!='annotations'} for k,v in source['database'].items() if k in selected}
        save(entry/'source_unlabeled.json',source,sort_keys=False)
        jobs[s]=dict(selection=selection,config_sha256=sha(entry/'config.json'),truth_sha256=sha(entry/'source_truth.json'),
            selection_sha256=sha(entry/'selection.json'),source_unlabeled_sha256=sha(entry/'source_unlabeled.json'),
            original_complete_sha256=sha(REFERENCE/s/'complete.json'),checkpoint_sha256=read(REFERENCE/s/'complete.json')['artifacts'][str(REFERENCE/s/'best.pt')])
    save(root/'plan.json',dict(code=identity,spec=SPEC,jobs=jobs,selection_frozen_before_predictions=True))


def gpu(entry):
    import gc,numpy as np,torch
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import (ProposalForeground,seconds_to_grid,interval_labels,
        balanced_loss,calibration_threshold,true_positive_flags)
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry)
    root=entry.parent;s=entry.name;cfg=read(entry/'config.json');selection=read(entry/'selection.json')
    truth=read(entry/'source_truth.json');plan=read(root/'plan.json');frozen=plan['jobs'][s]
    assert sha(entry/'selection.json')==frozen['selection_sha256'] and sha(entry/'source_truth.json')==frozen['truth_sha256']
    assert sha(REFERENCE/s/'best.pt')==frozen['checkpoint_sha256']
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    checkpoint=torch.load(REFERENCE/s/'best.pt',map_location='cpu');model.load_state_dict(checkpoint['ema_state_dict'],strict=True);model.eval();del checkpoint
    identity=base.tensor_digest(model.state_dict());index=[];first=True
    torch.cuda.reset_peak_memory_stats()
    for role,ids in selection.items():
        subset=cfg['src_val_split'] if role=='validation' else cfg['src_train_split']
        ds=base.dataset(cfg,entry/'source_unlabeled.json',subset,False,True)
        positions={v['id']:i for i,v in enumerate(ds.data_list)}
        for vid in ids:
            video=base.move_to_device([ds[positions[vid]]],'cuda:0');assert not video[0]['segments'].numel()
            with torch.no_grad():
                x,m=model.preprocessing(video,mode='test');x=model.dropout(x)
                f,m,_=model.backbone_source(x,m);f,m=model.neck_source(f,m)
                cls,_=model.cls_head(f,m);reg,_=model.reg_head(f,m);points=model.point_generator(f)
                result=model.inference(video,points,[a[:,0] for a in m],[a.permute(0,2,1) for a in cls],
                                       [a.permute(0,2,1) for a in reg],None,None)[0]
                if first:
                    native=model(video_list_src=video,video_list_tgt=video,mode='test')[0][0]
                    assert all(torch.equal(native[k],result[k]) for k in ('segments','scores','labels'))
                    first=False
                boxes=seconds_to_grid(result['segments'],video[0]['metadata']).cpu()
                metadata=video[0]['metadata'];gt=torch.tensor(truth[vid]['segments'],dtype=torch.float32).reshape(-1,2)
                low=-.5*metadata['feat_num_frames']/metadata['feat_stride']
                high=(metadata['duration']*metadata['fps']-.5*metadata['feat_num_frames'])/metadata['feat_stride']
                fit_boxes=full_candidates(boxes,gt,low,high) if role=='fit' else boxes
                y,iou=interval_labels(fit_boxes,gt)
                item=dict(feature=f[0][0].cpu(),valid=m[0][0,0].cpu().bool(),boxes=boxes,fit_boxes=fit_boxes,
                    states=y,best_iou=iou,metadata=metadata,role=role,video=vid,
                    segments=result['segments'].cpu(),scores=result['scores'].cpu(),labels=result['labels'].cpu())
            path=entry/'cache'/(role+'_'+vid+'.pt');path.parent.mkdir(exist_ok=True)
            base.checkpoint(path,item);index.append(dict(role=role,video=vid,path=str(path),sha256=sha(path)))
            print('SOURCE_CACHE',s,role,vid,flush=True)
    assert base.tensor_digest(model.state_dict())==identity
    save(entry/'cache_index.json',dict(records=index,source_initial_state_digest=identity,frozen_detector_unchanged=True,
        direct_original_forward_arrays_exact=True,checkpoint_sha256=frozen['checkpoint_sha256']))
    del model;gc.collect();torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(SPEC['head_seed']);head=ProposalForeground(512).cuda()
    optimizer=torch.optim.AdamW(head.parameters(),lr=SPEC['lr'],weight_decay=SPEC['weight_decay'])
    # 冻结特征只在此源读出预检复用，绝不用于会更新SGP的正式联合训练。
    fit=[]
    for r in index:
        if r['role']=='fit':
            item=torch.load(r['path'],map_location='cpu')
            fit.append({k:(v.cuda() if torch.is_tensor(v) else v) for k,v in item.items()})
    generator=torch.Generator().manual_seed(123);history=[]
    for epoch in range(1,21):
        head.train();total=0.
        for idx in torch.randperm(len(fit),generator=generator).tolist():
            item=fit[idx];optimizer.zero_grad(set_to_none=True)
            logits=head(item['feature'],item['valid'],item['fit_boxes']);loss=balanced_loss(logits,item['states'])
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            total+=float(loss.detach())
        history.append(dict(epoch=epoch,train_loss=total/len(fit)));save(entry/'history.json',history)
        print('HEAD_EPOCH',s,epoch,total/len(fit),flush=True)
    base.checkpoint(entry/'head_fixed20.pt',dict(state_dict=head.state_dict(),optimizer=optimizer.state_dict(),history=history,
        generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),spec=SPEC))
    head.eval();rows=[]
    with torch.no_grad():
        for r in index:
            item=torch.load(r['path'],map_location='cpu')
            probability=head(item['feature'].cuda(),item['valid'].cuda(),item['boxes'].cuda()).softmax(-1).cpu()
            gt=torch.tensor(truth[r['video']]['segments'],dtype=torch.float32).reshape(-1,2)
            gt_labels=torch.tensor(truth[r['video']]['labels'],dtype=torch.long)
            actual_states,_=interval_labels(item['boxes'],gt)
            tp=true_positive_flags(item['boxes'],item['labels'],item['scores'],gt,gt_labels)
            row={k:item[k] for k in ('video','segments','scores','labels','boxes')}
            row.update(role=r['role'],background_probability=probability[:,0],states=actual_states,true_positive=tp)
            rows.append(row)
    calibration=calibration_threshold([r for r in rows if r['role']=='calibration'])
    save(entry/'calibration.json',calibration)
    base.checkpoint(entry/'source_predictions.pt',rows)
    save(entry/'gpu_complete.json',dict(code=code_identity(),guard=guard,guard_sha256=sha(guard),
        predictions_sha256=sha(entry/'source_predictions.pt'),head_sha256=sha(entry/'head_fixed20.pt'),
        peak_mib=torch.cuda.max_memory_reserved()/2**20,roles=selection,cold_formal_training=False))


def evaluate(entry):
    import numpy as np,torch
    from libs.evaluation import ANETdetection
    from tools.proposal_foreground_math import interval_labels,true_positive_flags
    cfg=read(entry/'config.json');done=read(entry/'gpu_complete.json');selection=read(entry/'selection.json')
    assert sha(entry/'source_predictions.pt')==done['predictions_sha256']
    assert sha(entry/'head_fixed20.pt')==done['head_sha256']
    rows=torch.load(entry/'source_predictions.pt',map_location='cpu');truth=read(entry/'source_truth.json')
    threshold=read(entry/'calibration.json')['threshold'];columns={m:{k:[] for k in ('video-id','t-start','t-end','label','score')} for m in ('original','learned','confidence_equal_count','oracle')}
    videos=[]
    for r in rows:
        assert r['video'] in selection[r['role']]
        gt=torch.tensor(truth[r['video']]['segments'],dtype=torch.float32).reshape(-1,2)
        y,_=interval_labels(r['boxes'],gt);assert torch.equal(y,r['states'])
        if r['role']!='validation':continue
        removed=r['background_probability']>threshold;confidence=torch.zeros_like(removed)
        confidence[r['scores'].argsort()[:int(removed.sum())]]=True
        masks=dict(original=torch.ones_like(removed),learned=~removed,confidence_equal_count=~confidence,oracle=y!=0)
        tps={}
        for name,mask in masks.items():
            c=columns[name];c['video-id'].extend([r['video']]*int(mask.sum()))
            for k,v in [('t-start',r['segments'][mask,0]),('t-end',r['segments'][mask,1]),('label',r['labels'][mask]),('score',r['scores'][mask])]:c[k].extend(v.tolist())
            tps[name]=int(true_positive_flags(r['boxes'][mask],r['labels'][mask],r['scores'][mask],gt,torch.tensor(truth[r['video']]['labels'],dtype=torch.long)).sum())
        bg=y==0
        videos.append(dict(video=r['video'],candidates=len(y),background=int(bg.sum()),removed=int(removed.sum()),
            learned_background_removed=int((removed&bg).sum()),confidence_background_removed=int((confidence&bg).sum()),
            source_TP_at_point3=tps))
    assert {r['video'] for r in videos}==set(selection['validation'])
    results={}
    for name,col in columns.items():
        evaluator=ANETdetection(cfg['dataset']['json_file'],cfg['src_val_split'],num_workers=1,action_subsampling='10_majority_classes')
        ap,mean,*_=evaluator.evaluate({k:np.asarray(v) for k,v in col.items()},verbose=False)
        results[name]=dict(map=float(mean),map_by_tiou=ap.tolist())
    reference=read(REFERENCE/entry.name/'evaluation/source_val.json')['map_0.1_to_0.5']
    assert abs(results['original']['map']-reference)<1e-7
    result=dict(shift=entry.name,source_only=True,full_source_val=True,results=results,videos=videos,
        delta_pp=100*(results['learned']['map']-results['original']['map']),
        equal_count_confidence_delta_pp=100*(results['confidence_equal_count']['map']-results['original']['map']),
        source_oracle_delta_pp=100*(results['oracle']['map']-results['original']['map']),
        calibration_threshold=threshold,selection=selection,source_train_fit_subset_not_all_videos=True,
        formal_joint_or_target_mAP=False,
        artifacts={str(p):sha(p) for p in (entry/'config.json',entry/'selection.json',entry/'source_truth.json',entry/'calibration.json',entry/'source_predictions.pt',entry/'head_fixed20.pt',entry/'cache_index.json',entry/'history.json',entry/'gpu_complete.json')})
    save(entry/'complete.json',result);print(json.dumps({k:result[k] for k in ('shift','delta_pp','equal_count_confidence_delta_pp','source_oracle_delta_pp')}),flush=True)


def controller(root):
    import fcntl,signal
    from tools.baseline_gpu import snapshot
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=list(SHIFTS);active={};cpu_pending=[];completed=[]
    def stop(signum,frame):
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def spawn(s,stage,gpu=None):
        entry=root/s;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='')
        if gpu is not None:
            observed=snapshot(gpu);lease=entry/'lease.json'
            save(lease,dict(owner_pid=os.getpid(),index=gpu,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
        log=(entry/(stage+'.log')).open('w')
        p=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(entry)],env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[s]=dict(process=p,stage=stage,gpu=gpu,log=log)
    try:
        while pending or active or cpu_pending:
            for s,t in list(active.items()):
                if t['process'].poll() is None:continue
                t['log'].close()
                if t['process'].returncode:raise RuntimeError('源预检失败:'+s+'/'+t['stage'])
                del active[s]
                if t['stage']=='gpu':cpu_pending.append(s)
                else:completed.append(s)
            used={t['gpu'] for t in active.values()}
            for gpu in (0,2,3,1):
                if pending and gpu not in used and snapshot(gpu)['free_mib']>=12288:spawn(pending.pop(0),'gpu',gpu)
            old=read('/vepfs/vlms/sada_negative_curriculum6_f20e133_1103/state.json')
            other_cpu=sum(t['stage'] in ('signal_cpu','evaluate_cpu') for t in old['active'].values())
            if cpu_pending and not any(t['stage']=='evaluate' for t in active.values()) and other_cpu<2:spawn(cpu_pending.pop(0),'evaluate')
            save(root/'state.json',dict(pid=os.getpid(),active={s:dict(pid=t['process'].pid,stage=t['stage'],gpu=t['gpu']) for s,t in active.items()},pending=pending,cpu_pending=cpu_pending,completed=completed))
            if pending or active or cpu_pending:time.sleep(3)
        results=[read(root/s/'complete.json') for s in SHIFTS]
        save(root/'final.json',dict(count=6,results=results,source_only=True,not_formal_UDA=True,
            mean_source_delta_pp=sum(r['delta_pp'] for r in results)/6))
    except Exception as e:
        save(root/'error.json',dict(error=str(e)));raise
    finally:
        for t in active.values():
            if t['process'].poll() is None:os.killpg(t['process'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--run',type=Path);p.add_argument('--worker',choices=['gpu','evaluate']);p.add_argument('--entry',type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.run:controller(a.run)
    elif a.worker:
        os.chdir('/vepfs/vlms/SADA_PLUS')
        (gpu if a.worker=='gpu' else evaluate)(a.entry)
