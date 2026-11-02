"""只读复用源FPN；背景头拟合/源视频校准/源val测试三者分开。"""
from copy import deepcopy
from pathlib import Path
import hashlib,time,random
import numpy as np
import torch
from torch.nn import functional as F
from tools.baseline_protocol import read,save,sha
from tools.background_risk import BackgroundRisk,SPEC,video_split,fit_calibration,balanced_negative_BCE
from tools.background_risk_signals import weighted_metrics,aggregate


def result_record(metrics,calibration,**fields):
    # 校准集质量与校准器参数是两种不同工件，不能共用calibration键。
    return dict(status='source_evaluated',fit=metrics['fit'],calibration_quality=metrics['calibration'],
        validation=metrics['validation'],calibration=calibration,**fields)


def restore_source(head,optimizer,generator,saved):
    head.load_state_dict(saved['state_dict'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    generator.set_state(saved['generator'].cpu());torch.set_rng_state(saved['cpu_rng'].cpu())
    torch.cuda.set_rng_state(saved['cuda_rng'].cpu(),device=0)
    random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])


def raw_prediction(head,cached,device='cuda:0'):
    return torch.cat(head([x.to(device) for x in cached['features']],[x.to(device) for x in cached['logits']],
                         [x.to(device) for x in cached['masks']]),1)[0,:,0]


def run(cfg,entry):
    origin=Path(cfg['background_risk']['source_cache_entry'])
    old=read(origin/'independent_source_audit.json');assert old['result_sha256']==sha(origin/'result.json')
    previous=read(origin/'result.json');assert previous['index_sha256']==sha(origin/'cache_index.json')
    old_index=read(origin/'cache_index.json')
    reference=Path(cfg['background_risk']['reference'])/'best.pt'
    assert Path(old_index['reference'])==reference and old_index['reference_sha256']==sha(reference)
    index=old_index['index'];role=video_split([x['video'] for x in index['source_train']])
    manifest=read(cfg['dataset']['json_file'])['database']
    assert {x['video'] for x in index['source_train']}=={k for k,v in manifest.items() if v['subset'] in cfg['src_train_split']}
    assert {x['video'] for x in index['source_val']}=={k for k,v in manifest.items() if v['subset'] in cfg['src_val_split']}
    fit=[x for x in index['source_train'] if x['video'] in role['fit']]
    calibration=[x for x in index['source_train'] if x['video'] in role['calibration']]
    # 每份旧cache只核一次SHA，不复制大FPN，也不从GPU重新提取。
    for item in index['source_train']+index['source_val']:assert sha(item['path'])==item['sha256']
    save(entry/'cache_reuse.json',dict(origin=str(origin),index_sha256=sha(origin/'cache_index.json'),
        original_source_audit_sha256=sha(origin/'independent_source_audit.json'),roles=role,index=index,readonly=True))
    sample=torch.load(fit[0]['path'],map_location='cpu')
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(198731);head=BackgroundRisk(sample['features'][0].shape[1]).cuda()
    del sample
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.05)
    generator=torch.Generator().manual_seed(198731);history=[];start_epoch=0
    if (entry/'source_resume.json').exists():
        receipt=read(entry/'source_resume.json');assert sha(entry/'latest_risk.pt')==receipt['checkpoint_sha256']
        saved=torch.load(entry/'latest_risk.pt',map_location='cpu')
        history=deepcopy(saved['history']);start_epoch=saved['epoch']
        assert history==read(entry/'history.json') and [x['epoch'] for x in history]==list(range(1,start_epoch+1))
        restore_source(head,optimizer,generator,saved)
        from tools.baseline_engine import tensor_digest
        restored=dict(epoch=start_epoch,state_dict=head.state_dict(),optimizer=optimizer.state_dict(),history=history,
            generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),python_rng=random.getstate(),numpy_rng=np.random.get_state())
        assert tensor_digest(restored)==tensor_digest(saved)
        save(entry/'source_restore_proof.json',dict(epoch=start_epoch,all_state_digest=tensor_digest(restored),
            original_state_digest=tensor_digest(saved),checkpoint_sha256=receipt['checkpoint_sha256'],completed_epochs_replayed=0))
    for epoch in range(start_epoch+1,11):
        started=time.monotonic();values=[];head.train()
        for j in torch.randperm(len(fit),generator=generator).tolist():
            c=torch.load(fit[j]['path'],map_location='cpu');raw=raw_prediction(head,c)
            candidate=(c['valid']&(c['probability'].max(-1).values<SPEC['candidate_max_class_score'])).cuda()
            target=torch.from_numpy(~c['classes'].any(-1)).cuda()
            loss=balanced_negative_BCE(raw,target,candidate)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
            values.append(float(loss.detach()));del c
        history.append(dict(epoch=epoch,loss=float(np.mean(values)),seconds=time.monotonic()-started))
        from tools.baseline_engine import checkpoint
        checkpoint(entry/'latest_risk.pt',dict(epoch=epoch,state_dict=head.state_dict(),optimizer=optimizer.state_dict(),history=deepcopy(history),
            generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),python_rng=random.getstate(),numpy_rng=np.random.get_state()))
        save(entry/'history.json',history);print('BACKGROUND_FIT',history[-1],flush=True)
    head.eval();calrows=[]
    with torch.no_grad():
        for item in calibration:
            c=torch.load(item['path'],map_location='cpu');raw=raw_prediction(head,c).cpu()
            keep=c['valid']&(c['probability'].max(-1).values<SPEC['candidate_max_class_score'])
            negative=torch.from_numpy(~c['classes'].any(-1)).float()
            calrows.append(dict(video=item['video'],logits=raw[keep],background=negative[keep]))
    cal=fit_calibration(calrows,epoch=10);save(entry/'calibration.json',cal)
    result={}
    with torch.no_grad():
        for name,items in [('fit',fit),('calibration',calibration),('validation',index['source_val'])]:
            rows=[]
            for item in items:
                c=torch.load(item['path'],map_location='cpu');raw=raw_prediction(head,c).cpu().numpy()
                row=weighted_metrics(c['probability'].numpy(),raw,c['valid'].numpy(),~c['classes'].any(-1),
                    c['occupancy'],c['classes'],cal)
                rows.append(dict(row,video=item['video']))
            result[name]=aggregate(rows);save(entry/(name+'_quality.json'),dict(macro=result[name],videos=rows))
    from tools.baseline_engine import checkpoint
    if (entry/'best_risk.pt').exists():
        previous=torch.load(entry/'best_risk.pt',map_location='cpu')
        assert previous['epoch']==10 and all(torch.equal(v,head.state_dict()[k].cpu()) for k,v in previous['state_dict'].items())
        assert previous['calibration']==cal
    else:checkpoint(entry/'best_risk.pt',dict(epoch=10,state_dict=head.state_dict(),calibration=cal,selection='fixed_epoch10'))
    save(entry/'result.json',result_record(result,cal,
        roles=role,history_sha256=sha(entry/'history.json'),cache_reuse_sha256=sha(entry/'cache_reuse.json'),
        best_sha256=sha(entry/'best_risk.pt'),target_GT_used=False,head_not_transferred_to_formal=True))


def audit(cfg,entry):
    result=read(entry/'result.json');reuse=read(entry/'cache_reuse.json');roles=reuse['roles']
    assert not set(roles['fit'])&set(roles['calibration'])
    assert set(roles['fit'])|set(roles['calibration'])=={x['video'] for x in reuse['index']['source_train']}
    assert not (set(roles['fit'])|set(roles['calibration']))&{x['video'] for x in reuse['index']['source_val']}
    assert roles==video_split([x['video'] for x in reuse['index']['source_train']])
    assert result['best_sha256']==sha(entry/'best_risk.pt') and result['cache_reuse_sha256']==sha(entry/'cache_reuse.json')
    assert [x['epoch'] for x in read(entry/'history.json')]==list(range(1,11))
    saved=torch.load(entry/'best_risk.pt',map_location='cpu');assert saved['epoch']==10 and saved['calibration']==result['calibration']
    assert set(result['calibration']['videos'])<=set(roles['calibration'])
    report=dict(passed=True,result_sha256=sha(entry/'result.json'),cache_reuse_sha256=sha(entry/'cache_reuse.json'),
        roles=roles,head_epoch=10,source_val_not_used_for_fit_calibration_or_head_selection=True,target_GT_used=False)
    save(entry/'independent_source_audit.json',report);return report


def recalibrate(ctx,cfg,entry,epoch):
    """SGP已更新时重算真实EMA源calibration特征，绝不复用冻结FPN。"""
    from tools import baseline_engine as base
    from tools.supervision_risk import teacher_predictions
    from tools.supervision_risk_signals import stable_crop
    ids=set(ctx.background_roles['calibration']);ds=base.dataset(cfg,cfg['dataset']['json_file'],cfg['src_train_split'],False,True)
    rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(0),python=random.getstate(),numpy=np.random.get_state())
    teacher=ctx.model_ema.module;was_training=teacher.training;rows=[]
    try:
        with torch.no_grad():
            for i,item in enumerate(ds.data_list):
                if item['id'] not in ids:continue
                v=stable_crop(ds[i],teacher.max_seq_len);video=base.move_to_device([v],'cuda:0')[0]
                assert not video['segments'].numel() and not video['labels'].numel()
                pred=teacher_predictions(teacher,[video]);valid=pred['valid'][0].cpu()
                score=pred['cls'][0].sigmoid().max(-1).values.cpu();raw=pred['risk'][0,:,0].cpu()
                points=torch.cat(pred['points'])[:,0].cpu()
                full=torch.tensor(ctx.model.risk_source_truth[item['id']]['segments'],dtype=torch.float32).reshape(-1,2)-v['metadata']['crop_start']
                labels=np.asarray(ctx.model.risk_source_truth[item['id']]['labels'],int)
                length=v['feats'].shape[-1];intersection=(full[:,1]>0)&(full[:,0]<length)
                from tools.supervision_risk_signals import truth_arrays
                classes,_,_=truth_arrays(cfg,torch.cat(pred['points']).cpu().numpy(),full[intersection].clamp(0,float(length)).numpy(),
                    labels[intersection.numpy()],valid.numpy())
                negative=torch.from_numpy(~classes.any(-1))
                keep=valid&(score<SPEC['candidate_max_class_score'])
                rows.append(dict(video=item['id'],logits=raw[keep],background=negative[keep].float(),
                    crop_start=v['metadata']['crop_start'],length=v['feats'].shape[-1]))
        assert {r['video'] for r in rows}==ids
        cal=fit_calibration(rows,epoch=epoch)
        for model in [ctx.model,ctx.model_ema.module]:model.risk_state['calibration']=deepcopy(cal)
        folder=entry/'calibration';folder.mkdir(exist_ok=True)
        p=folder/('e%03d_rows.pt'%epoch);torch.save(rows,p)
        save(folder/('e%03d.json'%epoch),dict(calibration=cal,rows=str(p),rows_sha256=sha(p),source_train_only=True,GT_on_CPU_only=True))
        return cal
    finally:
        teacher.train(was_training)
        torch.set_rng_state(rng['cpu']);torch.cuda.set_rng_state(rng['cuda'],device=0)
        random.setstate(rng['python']);np.random.set_state(rng['numpy'])
