"""冻结成熟A的源域可学性检查。此小头不转移到正式冷启动训练。"""
from copy import deepcopy
import math
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from tools.baseline_protocol import read,save,sha
from tools.supervision_risk import attach,SPEC,teacher_predictions
from tools.supervision_risk_signals import stable_crop,truth_arrays,ranking_metrics,aggregate,readiness


def loss_on_cache(head,cache,device):
    features=[x.to(device) for x in cache['features']];logits=[x.to(device) for x in cache['logits']]
    masks=[x.to(device) for x in cache['masks']]
    result=torch.cat(head(features,logits,masks),1)[0]
    valid=cache['valid'].to(device);candidate=valid&(cache['probability'].max(-1).values.to(device)>=SPEC['candidate_floor'])
    correct=cache['correct'].to(device);bg=cache['background'].to(device)
    a=F.binary_cross_entropy_with_logits(result[candidate,0],correct[candidate]) if candidate.any() else result[:,0].sum()*0
    b=F.binary_cross_entropy_with_logits(result[valid,1],bg[valid])
    return a+b,result


def run(cfg,entry):
    from tools import baseline_engine as base
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    fix_random_seed(123,include_cuda=True);torch.cuda.manual_seed(123)
    model=make_meta_arch('sada',deepcopy(cfg),device='cuda:0').cuda()
    reference=Path(cfg['supervision_risk']['reference'])/'best.pt'
    state=torch.load(reference,map_location='cpu');model.load_state_dict(state['ema_state_dict'],strict=True);del state
    for p in model.parameters():p.requires_grad=False
    attach(model);model.eval()
    index={};folder=entry/'cache';folder.mkdir()
    for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
        ds=base.dataset(cfg,cfg['dataset']['json_file'],splits,False)
        paths=[]
        for i in range(len(ds)):
            video=stable_crop(ds[i],model.max_seq_len)
            # 源标注仅CPU构造标签，缓存抽取GPU只读去标注视频。
            segments=video['segments'].numpy();labels=video['labels'].numpy()
            unlabeled=dict(video,segments=torch.empty((0,2)),labels=torch.empty(0,dtype=torch.long))
            with torch.no_grad():pred=teacher_predictions(model,base.move_to_device([unlabeled],'cuda:0'))
            points=torch.cat(pred['points']).cpu().numpy();valid=pred['valid'][0].cpu().numpy()
            classes,bg,occupancy=truth_arrays(cfg,points,segments,labels,valid)
            probability=pred['cls'][0].sigmoid().cpu();label=probability.argmax(-1).numpy()
            cached=dict(features=tuple(x.cpu() for x in pred['features']),logits=tuple(x.cpu() for x in pred['raw_cls']),
                masks=tuple(x.cpu() for x in pred['masks']),probability=probability,valid=torch.from_numpy(valid),
                correct=torch.from_numpy((classes[np.arange(len(label)),label]>0).astype(np.float32)),
                background=torch.from_numpy(bg.astype(np.float32)),classes=classes,occupancy=occupancy,
                points=points,video=video['metadata']['video_id'],metadata=video['metadata'])
            path=folder/(role+'_'+video['metadata']['video_id']+'.pt');tmp=path.with_suffix('.tmp')
            torch.save(cached,tmp);tmp.replace(path)
            paths.append(dict(path=str(path),sha256=sha(path),video=cached['video'],subset=role))
            print('RISK_SOURCE_CACHE',role,i+1,len(ds),flush=True)
            del pred,cached
        index[role]=paths
    assert not {x['video'] for x in index['source_train']}&{x['video'] for x in index['source_val']}
    save(entry/'cache_index.json',dict(index=index,reference=str(reference),reference_sha256=sha(reference),
        protocol=SPEC['source_feasibility'],target_GT_used=False))
    head=model.supervision_risk
    optimizer=torch.optim.AdamW(head.parameters(),lr=3e-4,weight_decay=.05)
    generator=torch.Generator().manual_seed(198731);history=[];best=float('inf');best_epoch=None
    for epoch in range(11):
        started=time.monotonic();train_losses=[]
        if epoch:
            head.train()
            for j in torch.randperm(len(index['source_train']),generator=generator).tolist():
                cached=torch.load(index['source_train'][j]['path'],map_location='cpu')
                optimizer.zero_grad();loss,_=loss_on_cache(head,cached,'cuda:0');assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);optimizer.step()
                train_losses.append(float(loss.detach()));del cached
        head.eval();rows=[];val_losses=[]
        # val标签不进入GPU；GPU只产生risk输出，CPU计算BCE与全部质量指标。
        with torch.no_grad():
            for item in index['source_val']:
                cached=torch.load(item['path'],map_location='cpu')
                raw=torch.cat(head([x.cuda() for x in cached['features']],[x.cuda() for x in cached['logits']],
                                   [x.cuda() for x in cached['masks']]),1)[0].cpu()
                p=cached['probability'];valid=cached['valid'];candidate=valid&(p.max(-1).values>=SPEC['candidate_floor'])
                a=F.binary_cross_entropy_with_logits(raw[candidate,0],cached['correct'][candidate]) if candidate.any() else raw[:,0].sum()*0
                b=F.binary_cross_entropy_with_logits(raw[valid,1],cached['background'][valid])
                val_losses.append(float(a+b))
                row=ranking_metrics(p.numpy(),raw.sigmoid().numpy(),valid.numpy(),cached['classes'],cached['background'].numpy().astype(bool),cached['occupancy'])
                rows.append(dict(row,video=cached['video']))
        value=float(np.mean(val_losses));macro=aggregate(rows)
        history.append(dict(epoch=epoch,train_loss=float(np.mean(train_losses)) if train_losses else None,
            validation_loss=value,macro_by_video=macro,seconds=time.monotonic()-started))
        if value<best:
            best=value;best_epoch=epoch
            base.checkpoint(entry/'best_risk.pt',dict(epoch=epoch,state_dict=head.state_dict(),optimizer=optimizer.state_dict(),
                history=deepcopy(history),generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0)))
            save(entry/'best_quality.json',dict(epoch=epoch,macro_by_video=macro,videos=rows))
        save(entry/'history.json',history)
        base.checkpoint(entry/'latest_risk.pt',dict(epoch=epoch,state_dict=head.state_dict(),optimizer=optimizer.state_dict(),
            history=deepcopy(history),generator=generator.get_state(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(0),
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),best_epoch=best_epoch,best_loss=best))
        print('RISK_SOURCE_FIT',json.dumps(history[-1],allow_nan=False),flush=True)
    result=read(entry/'best_quality.json')
    save(entry/'result.json',dict(status='ready' if readiness(result['macro_by_video']) else 'source_mechanism_unready',
        best_epoch=best_epoch,macro_by_video=result['macro_by_video'],history_sha256=sha(entry/'history.json'),
        best_sha256=sha(entry/'best_risk.pt'),index_sha256=sha(entry/'cache_index.json'),head_not_transferred_to_formal=True,
        source_fit_only=True,target_GT_used=False))


def audit(cfg,entry):
    """源阶段结束后只读全审：独立从原manifest重算GT、坐标与每份标签。"""
    manifest=read(cfg['dataset']['json_file'])['database'];index=read(entry/'cache_index.json')['index']
    checked={}
    for role,splits in [('source_train',cfg['src_train_split']),('source_val',cfg['src_val_split'])]:
        expected={k for k,v in manifest.items() if v['subset'] in splits}
        assert {x['video'] for x in index[role]}==expected
        assert len(index[role])==len(expected)
        for item in index[role]:
            assert sha(item['path'])==item['sha256']
            c=torch.load(item['path'],map_location='cpu');m=c['metadata'];row=manifest[item['video']]
            gt=[x for x in row.get('annotations',[]) if 0<=int(x['label_id'])<10]
            segments=np.asarray([x['segment'] for x in gt],np.float32).reshape(-1,2)
            segments=(segments*m['fps']-.5*m['feat_num_frames'])/m['feat_stride']-m['crop_start']
            length=m['crop_feat_len'];keep=(segments[:,1]>0)&(segments[:,0]<length)
            labels=np.asarray([int(x['label_id']) for x in gt])[keep];segments=segments[keep].clip(0,length)
            classes,bg,occupancy=truth_arrays(cfg,c['points'],segments,labels,c['valid'].numpy())
            assert np.array_equal(classes,c['classes']) and np.array_equal(bg,c['background'].numpy().astype(bool))
            assert np.array_equal(occupancy,c['occupancy'])
            label=c['probability'].argmax(-1).numpy()
            assert np.array_equal(classes[np.arange(len(label)),label]>0,c['correct'].numpy().astype(bool))
            points=np.concatenate([np.arange(x.shape[-1],dtype=np.float32)*float(2**i) for i,x in enumerate(c['features'])])
            assert np.array_equal(points,c['points'][:,0])
        checked[role]=len(expected)
    assert not {x['video'] for x in index['source_train']}&{x['video'] for x in index['source_val']}
    history=read(entry/'history.json');assert [x['epoch'] for x in history]==list(range(11))
    selected=min(history,key=lambda x:(x['validation_loss'],x['epoch']))
    best=torch.load(entry/'best_risk.pt',map_location='cpu');latest=torch.load(entry/'latest_risk.pt',map_location='cpu')
    assert best['epoch']==selected['epoch'] and latest['epoch']==10 and latest['history']==history
    assert len(latest['optimizer']['param_groups'])==1 and latest['optimizer']['state']
    quality=read(entry/'best_quality.json');result=read(entry/'result.json')
    assert quality['epoch']==selected['epoch'] and quality['macro_by_video']==selected['macro_by_video']
    assert result['status']==('ready' if readiness(quality['macro_by_video']) else 'source_mechanism_unready')
    assert result['best_sha256']==sha(entry/'best_risk.pt') and result['index_sha256']==sha(entry/'cache_index.json')
    report=dict(passed=True,checked_members=checked,all_cache_SHA_and_GT_labels_recomputed=True,best_epoch=selected['epoch'],
        result_sha256=sha(entry/'result.json'),latest_sha256=sha(entry/'latest_risk.pt'),head_not_transferred_to_formal=True)
    save(entry/'independent_source_audit.json',report);return report
