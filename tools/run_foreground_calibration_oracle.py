"""更多源视频训练小头，target_train监督校准；target_val只作冻结规则后的CPU评估。"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from tools.baseline_protocol import read,save,sha,code_identity
REFERENCE=Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029/formal/control')
OLD_PILOT=Path('/vepfs/vlms/proposal_foreground_source_pilot6_eb30065_1103')
SHIFTS=tuple('S%d'%i for i in range(1,7))
SCHEMA='prototype-proposal-background-targettrain-calibration-oracle-v1'
SPEC=dict(source_fit='all_source_train_full_videos',head_seed=311103,seed=123,head_epochs=20,
    optimizer='AdamW',lr=3e-4,weight_decay=.05,clip=1.,head_parameters=45571,
    detector='original_Prototype_source_best_EMA_frozen',
    fit_candidates='all_original_postNMS_plus_visible_GT_and_two_fixed_neighbors',
    target_training='none; target_train_GT_for_CPU_calibration_only',
    calibration='target_train_AP_max_subject_to_perclass_each_tiou_TP_retention>=.995',
    threshold_grid='0:.025:1 plus 1.01',score_protection=[.1,.3,.5,.7,1.01],
    rules=['global','per_predicted_class'],target_val='final_CPU_evaluation_only_after_calibration_frozen',
    inference='original_NMS_voting_once_then_delete_only',formal_UDA=False,
    capacity_scope='fixed_learned_head_and_target_train_calibration_not_mathematical_upper_bound')


def prepare(root):
    from tools import baseline_engine as base
    root.mkdir(parents=True,exist_ok=False); identity=code_identity();jobs={}
    assert not subprocess.check_output(['git','diff','7a6b497','--','libs','tools/baseline_engine.py',
                                       'tools/proposal_foreground_math.py'],cwd=CODE)
    for s in SHIFTS:
        entry=root/s;entry.mkdir();cfg=read(REFERENCE/s/'config.json')
        assert sha(cfg['baseline']['annotation_for_offline_evaluation'])==cfg['baseline']['annotation_sha256']
        assert sha(cfg['dataset']['json_file'])==cfg['baseline']['source_manifest_sha256']
        assert sha(cfg['prototype_capacity']['prepared_cache'])==cfg['prototype_capacity']['prepared_cache_sha256']
        annotation=read(cfg['baseline']['annotation_for_offline_evaluation'])
        source=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False)
        source_val=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_val_split'],False)
        train=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_train_split'],False,True)
        val=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,True)
        selection=dict(source_fit=sorted(v['id'] for v in source.data_list),
                       target_train=sorted(v['id'] for v in train.data_list),
                       target_val=sorted(v['id'] for v in val.data_list))
        assert not set(selection['source_fit'])&{v['id'] for v in source_val.data_list}
        for a,b in [('source_fit','target_train'),('source_fit','target_val'),('target_train','target_val')]:
            assert not set(selection[a])&set(selection[b])
        truth={}
        for v in source.data_list:
            keep=(v['labels']>=0)&(v['labels']<10)
            truth[v['id']]=dict(segments=((v['segments'][keep]*v['fps']-.5*source.num_frames)/
                (source.feat_stride*source.downsample_rate)).tolist(),labels=v['labels'][keep].tolist())
        save(entry/'source_truth.json',truth);save(entry/'selection.json',selection);save(entry/'config.json',cfg)
        for role,ids in selection.items():
            data=dict(database={v:{k:x for k,x in annotation['database'][v].items() if k!='annotations'} for v in ids})
            assert all('annotations' not in row for row in data['database'].values())
            save(entry/(role+'_unlabeled.json'),data,sort_keys=False)
        # 目标train监督只存在独立CPU校准文件；val GT不进入该文件。
        train_truth=dict(database={v:annotation['database'][v] for v in selection['target_train']})
        save(entry/'target_train_truth_cpu.json',train_truth,sort_keys=False)
        old_index=read(OLD_PILOT/s/'cache_index.json')
        reuse=[r for r in old_index['records'] if r['video'] in selection['source_fit'] and r['role']!='validation']
        assert len({r['video'] for r in reuse})==len(reuse)
        save(entry/'reuse_source_cache.json',dict(records=reuse,old_index=str(OLD_PILOT/s/'cache_index.json'),
            old_index_sha256=sha(OLD_PILOT/s/'cache_index.json')))
        cp=REFERENCE/s/'best.pt';complete=read(REFERENCE/s/'complete.json')
        jobs[s]=dict(counts={k:len(v) for k,v in selection.items()},checkpoint_sha256=complete['artifacts'][str(cp)],
            original_complete_sha256=sha(REFERENCE/s/'complete.json'),
            inputs={str(p):sha(p) for p in entry.glob('*.json')})
    (root/'tmp').mkdir()
    save(root/'plan.json',dict(schema=SCHEMA,code=identity,spec=SPEC,jobs=jobs,
                              authorized_target_train_supervision=True,target_val_selection=False))


def native_features(model,video,check=False):
    import torch
    with torch.no_grad():
        x,m=model.preprocessing(video,mode='test');x=model.dropout(x)
        f,m,_=model.backbone_source(x,m);f,m=model.neck_source(f,m)
        cls,_=model.cls_head(f,m);reg,_=model.reg_head(f,m);points=model.point_generator(f)
        result=model.inference(video,points,[a[:,0] for a in m],[a.permute(0,2,1) for a in cls],
                               [a.permute(0,2,1) for a in reg],None,None)[0]
        if check:
            native=model(video_list_src=video,video_list_tgt=video,mode='test')[0][0]
            assert all(torch.equal(native[k],result[k]) for k in ('segments','scores','labels'))
    return f[0][0].detach(),m[0][0,0].bool().detach(),result


def load_detector(entry):
    import torch
    from libs.models.SADA import make_meta_arch
    cfg=read(entry/'config.json');model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    ckpt=torch.load(REFERENCE/entry.name/'best.pt',map_location='cpu')
    model.load_state_dict(ckpt['ema_state_dict'],strict=True);model.eval()
    return model


def validate_inputs(entry):
    frozen=read(entry.parent/'plan.json')['jobs'][entry.name]
    for p,digest in frozen['inputs'].items():assert sha(p)==digest
    assert sha(REFERENCE/entry.name/'best.pt')==frozen['checkpoint_sha256']
    return frozen


def kernel(entry):
    """真实视频/磁盘/梯度/AdamW完整恢复验收，不把两步接线当效果。"""
    import io
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import ProposalForeground,interval_labels,balanced_loss,seconds_to_grid
    from tools.proposal_foreground_pilot import full_candidates
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);validate_inputs(entry);cfg=read(entry/'config.json')
    model=load_detector(entry);initial=base.tensor_digest(model.state_dict())
    ds=base.dataset(cfg,entry/'source_fit_unlabeled.json',cfg['src_train_split'],False,True)
    video=base.move_to_device([ds[0]],'cuda:0');feature,valid,result=native_features(model,video,True)
    vid=video[0]['metadata']['video_id'];meta=video[0]['metadata']
    gt=torch.tensor(read(entry/'source_truth.json')[vid]['segments'],dtype=torch.float32,device='cuda').reshape(-1,2)
    low=-.5*meta['feat_num_frames']/meta['feat_stride'];high=(meta['duration']*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
    # 原NMS/voting把最终segments返回CPU；小头在GPU，显式映射到feature设备。
    boxes=full_candidates(seconds_to_grid(result['segments'],meta).to(feature.device),gt,low,high)
    labels,_=interval_labels(boxes,gt)
    torch.manual_seed(SPEC['head_seed']);head=ProposalForeground(512).cuda()
    opt=torch.optim.AdamW(head.parameters(),lr=SPEC['lr'],weight_decay=SPEC['weight_decay'])
    before=base.tensor_digest(head.state_dict());grads=[]
    for _ in range(2):
        opt.zero_grad(set_to_none=True);loss=balanced_loss(head(feature,valid,boxes),labels);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(head.parameters(),SPEC['clip']);assert torch.isfinite(norm) and norm>0
        opt.step();grads.append(float(norm))
    assert before!=base.tensor_digest(head.state_dict())
    assert base.tensor_digest(model.state_dict())==initial and all(p.grad is None for p in model.parameters())
    payload=dict(head=head.state_dict(),optimizer=opt.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0))
    path=entry/'kernel'/'state.pt';path.parent.mkdir(exist_ok=True);base.checkpoint(path,payload)
    restored=torch.load(path,map_location='cpu');second=ProposalForeground(512).cuda();second.load_state_dict(restored['head'])
    second_opt=torch.optim.AdamW(second.parameters(),lr=SPEC['lr'],weight_decay=SPEC['weight_decay']);second_opt.load_state_dict(restored['optimizer'])
    assert base.tensor_digest(restored['head'])==base.tensor_digest(head.state_dict())
    assert base.tensor_digest(second_opt.state_dict())==base.tensor_digest(opt.state_dict())
    with torch.no_grad():assert torch.equal(second(feature,valid,boxes),head(feature,valid,boxes))
    torch.set_rng_state(restored['cpu_rng']);torch.cuda.set_rng_state(restored['cuda_rng'],0)
    assert torch.equal(torch.get_rng_state(),payload['cpu_rng']) and torch.equal(torch.cuda.get_rng_state(0),payload['cuda_rng'])
    save(entry/'kernel_complete.json',dict(guard=guard,guard_sha256=sha(guard),passed=True,source_video=vid,
        source_candidates=len(boxes),grad_norms=grads,original_forward_arrays_exact=True,
        frozen_detector_unchanged=True,source_head_optimizer_disk_restore=True,state_sha256=sha(path),
        peak_mib=torch.cuda.max_memory_reserved()/2**20,code=code_identity()))


def fit_source(entry):
    import gc,random
    import numpy as np
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import ProposalForeground,seconds_to_grid,interval_labels,balanced_loss
    from tools.proposal_foreground_pilot import full_candidates
    from tools.run_proposal_foreground import first_cuda_guard
    from libs.utils import fix_random_seed
    guard=first_cuda_guard(entry);frozen=validate_inputs(entry);cfg=read(entry/'config.json')
    selection=read(entry/'selection.json');truth=read(entry/'source_truth.json')
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123);torch.cuda.reset_peak_memory_stats()
    model=load_detector(entry);initial=base.tensor_digest(model.state_dict());records=[];first=True
    reuse=read(entry/'reuse_source_cache.json');assert sha(reuse['old_index'])==reuse['old_index_sha256']
    lookup={r['video']:r for r in reuse['records']}
    ds=base.dataset(cfg,entry/'source_fit_unlabeled.json',cfg['src_train_split'],False,True)
    positions={v['id']:i for i,v in enumerate(ds.data_list)}
    for vid in selection['source_fit']:
        if vid in lookup:
            old=lookup[vid];assert sha(old['path'])==old['sha256']
            # 旧GPU缓存属于相同原Prototype EMA；绑定其原完整index和checkpoint SHA。
            old_index=read(OLD_PILOT/entry.name/'cache_index.json')
            assert old_index['checkpoint_sha256']==frozen['checkpoint_sha256']
            assert old_index['source_initial_state_digest']==initial
            path=Path(old['path']);digest=old['sha256']
        else:
            video=base.move_to_device([ds[positions[vid]]],'cuda:0')
            assert video[0]['segments'].numel()==0
            feature,valid,result=native_features(model,video,first);first=False
            metadata=video[0]['metadata'];boxes=seconds_to_grid(result['segments'],metadata).cpu()
            item=dict(feature=feature.cpu(),valid=valid.cpu(),boxes=boxes,metadata=metadata,video=vid,
                      segments=result['segments'].cpu(),scores=result['scores'].cpu(),labels=result['labels'].cpu())
            path=entry/'cache'/('source_'+vid+'.pt');path.parent.mkdir(exist_ok=True)
            base.checkpoint(path,item);digest=sha(path)
        records.append(dict(video=vid,path=str(path),sha256=digest,reused=vid in lookup))
        print('SOURCE_CACHE',entry.name,vid,'reused' if vid in lookup else 'new',flush=True)
    assert base.tensor_digest(model.state_dict())==initial
    save(entry/'cache_index.json',dict(records=records,detector_digest=initial,detector_unchanged=True,
                                      original_forward_arrays_exact=not first))
    del model;gc.collect();torch.cuda.empty_cache()
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(SPEC['head_seed']);head=ProposalForeground(512).cuda()
    assert sum(p.numel() for p in head.parameters())==SPEC['head_parameters']
    optimizer=torch.optim.AdamW(head.parameters(),lr=SPEC['lr'],weight_decay=SPEC['weight_decay'])
    # 缓存CPU内存只读：每视频搬到GPU一次/step；绝不把全部长视频常驻GPU。
    fit=[]
    for r in records:
        item=torch.load(r['path'],map_location='cpu');vid=r['video'];gt=torch.tensor(truth[vid]['segments'],dtype=torch.float32).reshape(-1,2)
        meta=item['metadata'];low=-.5*meta['feat_num_frames']/meta['feat_stride']
        high=(meta['duration']*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
        boxes=full_candidates(item['boxes'],gt,low,high);states,_=interval_labels(boxes,gt)
        fit.append(dict(feature=item['feature'],valid=item['valid'],boxes=boxes,states=states))
    generator=torch.Generator().manual_seed(123);history=[];start=base.tensor_digest(head.state_dict())
    for epoch in range(1,SPEC['head_epochs']+1):
        head.train();total=0.;begin=time.time()
        for idx in torch.randperm(len(fit),generator=generator).tolist():
            item=fit[idx];optimizer.zero_grad(set_to_none=True)
            loss=balanced_loss(head(item['feature'].cuda(),item['valid'].cuda(),item['boxes'].cuda()),item['states'].cuda())
            assert torch.isfinite(loss);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(head.parameters(),SPEC['clip']);assert torch.isfinite(norm)
            optimizer.step();total+=float(loss.detach())
        history.append(dict(epoch=epoch,train_loss=total/len(fit),seconds=time.time()-begin))
        payload=dict(state_dict=head.state_dict(),optimizer=optimizer.state_dict(),history=history,
            generator=generator.get_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),spec=SPEC,
            index_sha256=sha(entry/'cache_index.json'),initial_head_digest=start)
        base.checkpoint(entry/'head_latest.pt',payload);save(entry/'history.json',history)
        print('HEAD_EPOCH',entry.name,epoch,history[-1],flush=True)
    assert base.tensor_digest(head.state_dict())!=start
    os.link(entry/'head_latest.pt',entry/'head_fixed20.pt')
    save(entry/'fit_complete.json',dict(guard=guard,guard_sha256=sha(guard),code=code_identity(),
        spec=SPEC,source_videos=len(fit),epochs=len(history),head_updated=True,initial_head_digest=start,
        detector_digest=initial,peak_mib=torch.cuda.max_memory_reserved()/2**20,
        artifacts={str(p):sha(p) for p in (entry/'head_fixed20.pt',entry/'history.json',entry/'cache_index.json')}))


def predict_target(entry,role):
    import torch
    from tools import baseline_engine as base
    from tools.proposal_foreground_math import ProposalForeground,seconds_to_grid
    from tools.run_proposal_foreground import first_cuda_guard
    guard=first_cuda_guard(entry);validate_inputs(entry);cfg=read(entry/'config.json')
    done=read(entry/'fit_complete.json')
    assert sha(entry/'head_fixed20.pt')==done['artifacts'][str(entry/'head_fixed20.pt')]
    model=load_detector(entry);initial=base.tensor_digest(model.state_dict())
    head=ProposalForeground(512).cuda();head.load_state_dict(torch.load(entry/'head_fixed20.pt',map_location='cpu')['state_dict']);head.eval()
    # 该函数从不打开target_train_truth_cpu.json或离线annotation；无GT清单唯一数据入口。
    splits=cfg['tgt_train_split'] if role=='target_train' else cfg['tgt_val_split']
    manifest=entry/(role+'_unlabeled.json');assert all('annotations' not in r for r in read(manifest)['database'].values())
    ds=base.dataset(cfg,manifest,splits,False,True);selection=read(entry/'selection.json')[role]
    assert sorted(v['id'] for v in ds.data_list)==selection
    rows=[];torch.cuda.reset_peak_memory_stats()
    for i in range(len(ds)):
        video=base.move_to_device([ds[i]],'cuda:0');assert not video[0]['segments'].numel()
        feature,valid,result=native_features(model,video,i==0)
        with torch.no_grad():
            boxes=seconds_to_grid(result['segments'],video[0]['metadata']).to(feature.device)
            probability=head(feature,valid,boxes).softmax(-1).cpu()
        rows.append(dict(video=video[0]['metadata']['video_id'],segments=result['segments'].cpu(),scores=result['scores'].cpu(),
                         labels=result['labels'].cpu(),pbg=probability[:,0]))
        print('PREDICT',entry.name,role,i+1,len(ds),flush=True)
    assert base.tensor_digest(model.state_dict())==initial
    assert sorted(r['video'] for r in rows)==selection
    path=entry/(role+'_predictions.pt');base.checkpoint(path,rows)
    save(entry/(role+'_gpu_complete.json'),dict(guard=guard,guard_sha256=sha(guard),manifest=str(manifest),
        manifest_sha256=sha(manifest),predictions_sha256=sha(path),head_sha256=sha(entry/'head_fixed20.pt'),
        detector_unchanged=True,original_forward_arrays_exact=True,peak_mib=torch.cuda.max_memory_reserved()/2**20))


def load_truth(path,splits):
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    gt=load_gt_seg_from_json(path,splits,action_subsampling='10_majority_classes')
    return gt[gt['t-start']!=gt['t-end']].reset_index(drop=True)


def calibrate_cpu(entry):
    import torch
    from tools.foreground_oracle_ap import calibrate,prediction_frame
    role='target_train';done=read(entry/(role+'_gpu_complete.json'));path=entry/(role+'_predictions.pt')
    assert sha(path)==done['predictions_sha256']
    selection=read(entry/'selection.json');truth=entry/'target_train_truth_cpu.json'
    assert set(read(truth)['database'])==set(selection[role]) and not set(read(truth)['database'])&set(selection['target_val'])
    frame=prediction_frame(torch.load(path,map_location='cpu'));cfg=read(entry/'config.json')
    gt=load_truth(truth,cfg['tgt_train_split']);cal=calibrate(gt,frame)
    cal.update(predictions_sha256=sha(path),target_train_truth_sha256=sha(truth),code=code_identity())
    save(entry/'calibration_frozen.json',cal)
    print('CALIBRATION_FROZEN',entry.name,cal['global_rule'],flush=True)


def evaluate_cpu(entry):
    import numpy as np
    import torch
    from tools.foreground_oracle_ap import (prediction_frame,apply_rules,background_mask,ClassAP)
    from libs.evaluation import ANETdetection
    cfg=read(entry/'config.json');cal_path=entry/'calibration_frozen.json';cal=read(cal_path)
    assert sha(entry/'target_train_predictions.pt')==cal['predictions_sha256']
    assert sha(entry/'target_train_truth_cpu.json')==cal['target_train_truth_sha256']
    frozen_sha=sha(cal_path);done=read(entry/'target_val_gpu_complete.json');path=entry/'target_val_predictions.pt'
    assert sha(path)==done['predictions_sha256']
    rows=torch.load(path,map_location='cpu');frame=prediction_frame(rows)
    gt=load_truth(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'])
    assert set(frame['video-id'])<=set(read(entry/'selection.json')['target_val'])
    background=background_mask(gt,frame)
    masks=dict(original=np.ones(len(frame),dtype=bool),global_threshold=apply_rules(frame,cal),
               class_threshold=apply_rules(frame,cal,True),perfect_background_oracle=~background)
    results={};perclass={};classes=sorted(int(c) for c in gt.label.unique())
    for c in classes:
        positions=np.flatnonzero(frame.label.to_numpy()==c);sub=frame.iloc[positions].reset_index(drop=True)
        engine=ClassAP(gt[gt.label==c],sub)
        perclass[str(c)]={name:engine.evaluate(mask[positions]) for name,mask in masks.items()}
    for name,mask in masks.items():
        ap=np.array([perclass[str(c)][name]['ap'] for c in classes])
        tp=np.array([perclass[str(c)][name]['tp'] for c in classes])
        results[name]=dict(map=float(ap.mean()),map_by_tiou=ap.mean(0).tolist(),kept=int(mask.sum()),
            tp_by_tiou=tp.sum(0).tolist(),background_removed=int((~mask&background).sum()),
            nonbackground_removed=int((~mask&~background).sum()))
    # 独立官方评估器核全部四组真实预测；不是只用自写快速AP给自己验收。
    official={}
    for name,mask in masks.items():
        evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],
                                num_workers=1,action_subsampling='10_majority_classes')
        values=evaluator.evaluate(frame.loc[mask,['video-id','t-start','t-end','label','score']].copy(),verbose=False)
        assert abs(float(values[1])-results[name]['map'])<1e-7
        np.testing.assert_allclose(values[0],results[name]['map_by_tiou'],atol=1e-7,rtol=0)
        official[name]=float(values[1])
    ref=read(REFERENCE/entry.name/'evaluation/target_val.json')['map_0.1_to_0.5']
    assert abs(results['original']['map']-ref)<1e-7
    assert sha(cal_path)==frozen_sha
    save(entry/'evaluation.json',dict(results=results,classes=perclass,official_AP=official,
        global_delta_pp=100*(results['global_threshold']['map']-results['original']['map']),
        class_delta_pp=100*(results['class_threshold']['map']-results['original']['map']),
        oracle_delta_pp=100*(results['perfect_background_oracle']['map']-results['original']['map']),
        target_train_supervised_Oracle=True,formal_UDA=False,calibration_frozen_sha256=frozen_sha))
    artifacts={str(p):sha(p) for p in entry.glob('*.json') if p.name not in ('complete.json',)}
    artifacts.update({str(entry/x):sha(entry/x) for x in ('target_train_predictions.pt','target_val_predictions.pt','head_fixed20.pt')})
    save(entry/'complete.json',dict(schema=SCHEMA,shift=entry.name,code=code_identity(),spec=SPEC,
        artifacts=artifacts,evaluation=read(entry/'evaluation.json'),target_val_used_for_selection=False))


def audit(entry):
    done=read(entry/'complete.json');assert done['schema']==SCHEMA and done['spec']==SPEC
    for path,digest in done['artifacts'].items():assert sha(path)==digest,path
    selection=read(entry/'selection.json');assert len(selection['source_fit'])==read(entry.parent/'plan.json')['jobs'][entry.name]['counts']['source_fit']
    assert set(read(entry/'source_truth.json'))==set(selection['source_fit'])
    assert set(read(entry/'target_train_truth_cpu.json')['database'])==set(selection['target_train'])
    assert not set(selection['target_train'])&set(selection['target_val'])
    history=read(entry/'history.json');assert [r['epoch'] for r in history]==list(range(1,21))
    for role in ('source_fit','target_train','target_val'):
        assert all('annotations' not in r for r in read(entry/(role+'_unlabeled.json'))['database'].values())
    for name in ('fit_complete','target_train_gpu_complete','target_val_gpu_complete'):
        record=read(entry/(name+'.json'));assert sha(record['guard'])==record['guard_sha256']
        guard=read(record['guard']);assert sha(guard['lease'])==guard['lease_sha256']
        lease=read(guard['lease']);assert guard['owner_pid']==lease['owner_pid']
        assert guard['observed']['uuid']==lease['uuid'] and guard['observed']['free_mib']>=12288
    cal=read(entry/'calibration_frozen.json')
    for d in cal['per_class'].values():
        selected=d['grid'][d['selected_index']]
        assert all(a>=.995*b for a,b in zip(selected['tp'],d['baseline']['tp']))
    return dict(passed=True,complete_sha256=sha(entry/'complete.json'),evaluation=done['evaluation'])


def controller(root):
    import fcntl
    from tools.baseline_gpu import snapshot
    lock=(root/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pending=list(SHIFTS);kernel_pending=list(SHIFTS);kernels=[];active={};next_gpu=[];cpu_pending=[];completed=[]
    def stop(signum,frame):
        for task in active.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        raise SystemExit(128+signum)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def spawn(s,stage,gpu=None):
        entry=root/s;env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CODE),
            OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',CUDA_VISIBLE_DEVICES='',TMPDIR=str(root/'tmp'))
        if gpu is not None:
            observed=snapshot(gpu);lease=entry/(stage+'_lease.json')
            save(lease,dict(owner_pid=os.getpid(),index=gpu,uuid=observed['uuid'],minimum_free_mib=12288,commit=code_identity()['commit']))
            env.update(CUDA_VISIBLE_DEVICES=observed['uuid'],PROPOSAL_FOREGROUND_LEASE=str(lease),PROPOSAL_FOREGROUND_LEASE_SHA256=sha(lease))
        log=(entry/(stage+'.log')).open('a')
        process=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__)),'--worker',stage,'--entry',str(entry)],
            env=env,cwd=CODE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        active[s]=dict(process=process,stage=stage,gpu=gpu,log=log)
    try:
        while pending or kernel_pending or active or next_gpu or cpu_pending:
            for s,t in list(active.items()):
                if t['process'].poll() is None:continue
                t['log'].close();del active[s]
                if t['process'].returncode:raise RuntimeError(s+'/'+t['stage']+' failed')
                if t['stage']=='kernel':
                    assert read(root/s/'kernel_complete.json')['passed'];kernels.append(s)
                elif t['stage']=='fit_source':next_gpu.append((s,'target_train'))
                elif t['stage']=='target_train':cpu_pending.append((s,'calibrate_cpu'))
                elif t['stage']=='calibrate_cpu':next_gpu.append((s,'target_val'))
                elif t['stage']=='target_val':cpu_pending.append((s,'evaluate_cpu'))
                else:
                    report=audit(root/s);save(root/'controller_audits'/(s+'.json'),report);completed.append(s)
            used={t['gpu'] for t in active.values()}
            for gpu in (2,0,3,1):
                if gpu in used or snapshot(gpu)['free_mib']<12288:continue
                # 已完成头的目标校准优先，尽早提供完整方向结果。
                if kernel_pending:spawn(kernel_pending.pop(0),'kernel',gpu)
                elif len(kernels)<6:continue
                elif next_gpu:s,stage=next_gpu.pop(0);spawn(s,stage,gpu)
                elif pending:spawn(pending.pop(0),'fit_source',gpu)
            # 本CPU队列最多1路；其它旧两队列都已退出/无CPU才放行，合计不超过2路。
            other_cpu=0
            for path in (OLD_PILOT/'state.json',Path('/vepfs/vlms/sada_negative_curriculum6_f20e133_1103/state.json')):
                if not path.exists():continue
                state=read(path)
                for t in state.get('active',{}).values():
                    if t.get('stage') in ('evaluate','evaluate_cpu','signal_cpu') and Path('/proc/%s'%t['pid']).exists():other_cpu+=1
            if cpu_pending and other_cpu<2 and not any(t['gpu'] is None for t in active.values()):
                s,stage=cpu_pending.pop(0);spawn(s,stage)
            save(root/'state.json',dict(pid=os.getpid(),pending=pending,kernel_pending=kernel_pending,kernels=kernels,next_gpu=next_gpu,cpu_pending=cpu_pending,
                completed=completed,active={s:dict(pid=t['process'].pid,stage=t['stage'],gpu=t['gpu']) for s,t in active.items()}))
            if pending or active or next_gpu or cpu_pending:time.sleep(3)
        results=[read(root/s/'complete.json') for s in SHIFTS]
        save(root/'final.json',dict(schema=SCHEMA,count=6,results=results,formal_UDA=False,
            mean_global_delta_pp=sum(r['evaluation']['global_delta_pp'] for r in results)/6,
            mean_class_delta_pp=sum(r['evaluation']['class_delta_pp'] for r in results)/6,
            mean_perfect_background_delta_pp=sum(r['evaluation']['oracle_delta_pp'] for r in results)/6))
    except Exception as e:
        save(root/'error.json',dict(error=repr(e)));raise
    finally:
        for task in active.values():
            if task['process'].poll() is None:os.killpg(task['process'].pid,signal.SIGTERM)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--worker',choices=['kernel','fit_source','target_train','target_val','calibrate_cpu','evaluate_cpu']);p.add_argument('--entry',type=Path)
    a=p.parse_args()
    if a.prepare:prepare(a.prepare)
    elif a.run:controller(a.run)
    elif a.worker:
        os.chdir('/vepfs/vlms/SADA_PLUS')
        if a.worker=='kernel':kernel(a.entry)
        elif a.worker=='fit_source':fit_source(a.entry)
        elif a.worker in ('target_train','target_val'):predict_target(a.entry,a.worker)
        elif a.worker=='calibrate_cpu':calibrate_cpu(a.entry)
        else:evaluate_cpu(a.entry)
