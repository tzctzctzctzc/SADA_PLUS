"""端点通道容量训练Oracle；仅专用实验读目标train真值，不是正式UDA。"""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

MODES=('control','gate','grid','gt')
SCHEMA='prototype-endpoint-training-oracle-v1'
SPEC=dict(schema=SCHEMA,modes=list(MODES),epochs=20,seed=123,batch=2,onset=5,ramp=5,weight=.1,
    teacher_score=.5,teacher_nms=.6,teacher_cap=32,grid=17,radius=.25,
    initialization='same mature Prototype EMA extra budget',inference='original detector unchanged',
    primary='fixed epoch20 EMA target AP',secondary='source cls+reg selected EMA incl epoch0; source AP all snapshots',
    scope='headless endpoint supervision capacity; no latent phase ground truth or extra source phase loss',
    target_training_GT='explicitly authorized diagnostic only; train subset only; oracle models never enter formal UDA')


def crop_truth(record,metadata,length):
    assert record['fps']==metadata['fps'] and record['feat_stride']==metadata['feat_stride']
    assert record['feat_num_frames']==metadata['feat_num_frames']
    start=metadata.get('crop_start',0)
    boxes=np.asarray(record['segments'],dtype=np.float32).reshape(-1,2)-start
    labels=np.asarray(record['labels'],dtype=np.int64)
    ids=(boxes[:,1]>0)&(boxes[:,0]<length)&(labels<10)
    return boxes[ids],labels[ids]


def choose_truth(truth,labels,label,point):
    ids=np.flatnonzero((labels==int(label))&(truth[:,0]<=point)&(truth[:,1]>=point))
    return int(ids[np.argmin(truth[ids,1]-truth[ids,0])]) if len(ids) else None


def select_endpoint(reference,gt,side,mode,length):
    if mode=='gate':return float(reference[side])
    if mode=='gt':return float(gt[side])
    assert mode=='grid'
    offsets=np.r_[np.float32(0),np.arange(-8,9,dtype=np.float32)[np.arange(-8,9)!=0]/np.float32(32)]
    coordinates=np.float32(reference[side])+offsets*np.float32(reference[1]-reference[0])
    valid=(coordinates>=0)&(coordinates<length);coordinates=coordinates[valid]
    return float(coordinates[np.argmin(np.abs(coordinates-np.float32(gt[side])))])


def oracle_mask(teacher,student,gt,point,length,view):
    from tools.phase_boundary import map_coordinates
    mapped=map_coordinates(teacher,view);mapped_gt=map_coordinates(gt,view)
    pred_back=map_coordinates(student.detach(),view,True)
    representable=torch.stack([mapped[0]<=point,mapped[1]>=point])
    visible=(gt>0)&(gt<length)&(mapped_gt>=0)&(mapped_gt<view['length'])&(mapped>=0)&(mapped<view['length'])
    return ((teacher-gt).abs()<(pred_back-gt).abs())&representable&visible


def oracle_loss(model,weak,strong,views,videos,truth,mode,observations):
    from tools.phase_boundary import select_proposals,map_coordinates
    sf,sm,sc,sr=strong;tf,tm,tc,tr=weak
    zero=sf[0].sum()*0;losses=[]
    stats=dict(proposals=0,matched=0,left=0,right=0,source_GT_used=False,target_train_GT_used=mode!='control')
    for vi,(v,view) in enumerate(zip(videos,views)):
        proposals=select_proposals(tf,tm,tc,tr,model,vi,view['original_length'])
        stats['proposals']+=len(proposals)
        if mode=='control':continue
        boxes,labels=crop_truth(truth[v['metadata']['video_id']],v['metadata'],view['original_length'])
        for box,c,level,index,confidence in proposals:
            stride=model.fpn_strides[level];point=index*stride
            gi=choose_truth(boxes,labels,c,point)
            if gi is None:continue
            stats['matched']+=1
            if box[0]<view['lo'] or box[1]>view['lo']+view['width']-1:continue
            k=int((map_coordinates(box.new_tensor(point),view)/stride).round())
            if not 0<=k<sf[level].shape[-1] or not sm[level][vi,0,k]:continue
            pred=torch.stack([k*stride-sr[level][vi,0,k]*stride,k*stride+sr[level][vi,1,k]*stride])
            raw=box.detach().cpu().numpy();target=box.new_tensor([select_endpoint(raw,boxes[gi],side,mode,view['original_length']) for side in [0,1]])
            gt=box.new_tensor(boxes[gi]);mask=oracle_mask(target,pred,gt,k*stride,view['original_length'],view)
            if mask.any():
                mapped=map_coordinates(target,view);duration=map_coordinates(box,view).diff()[0].clamp_min(1e-6)
                loss=F.smooth_l1_loss((pred-mapped.detach())/duration,torch.zeros_like(pred),reduction='none')[mask].mean()
                losses.append(loss);stats['left']+=int(mask[0]);stats['right']+=int(mask[1])
            observations.append(dict(video=v['metadata']['video_id'],label=int(c),instance=int(gi),level=level,point=point,
                crop_start=v['metadata'].get('crop_start',0),reference=raw.tolist(),target=target.cpu().tolist(),
                student=map_coordinates(pred.detach(),view,True).cpu().tolist(),mask=mask.cpu().tolist(),view=view))
    return (torch.stack(losses).mean() if losses else zero),stats


def install(ctx,cfg,truth):
    from tools import baseline_engine as base
    model=ctx.model;ema=ctx.model_ema.module
    saved=torch.load(Path(cfg['training_oracle']['reference'])/'best.pt',map_location='cpu')
    for m in [model,ema]:
        m.load_state_dict(saved['ema_state_dict'],strict=True)
        for k,v in saved['ema_controls'].items():
            if torch.is_tensor(v):getattr(m,k).data.copy_(v)
            elif isinstance(v,list):
                for a,b in zip(getattr(m,k),v):a.data.copy_(b)
            else:setattr(m,k,deepcopy(v))
        m.oracle_training_state=dict(step=0,events=0,digest='0'*64)
    original=model.forward;ctx.oracle_original_forward=original;ctx.oracle_gradient_epoch=None;ctx.oracle_ledger=None;ctx.oracle_last={}
    def forward(*args,**kwargs):
        from tools.phase_boundary import direct,strong_view
        if kwargs.get('mode')!='train':return original(*args,**kwargs)
        state=model.oracle_training_state;state['step']+=1;out=original(*args,**kwargs)
        epoch=kwargs['curr_epoch'];mode=cfg['training_oracle']['mode'];stats=dict(proposals=0,matched=0,left=0,right=0,target_train_GT_used=False,aux_loss=0.)
        observations=[]
        if epoch>=SPEC['onset']:
            videos=kwargs['video_list_tgt'];strong=[];views=[]
            for i,v in enumerate(videos):
                x,view=strong_view(v,state['step'],i);view['original_length']=v['feats'].shape[-1];strong.append(x);views.append(view)
            ema.eval()
            with torch.no_grad():weak=direct(ema,videos)
            transformed=direct(model,strong)
            loss,stats=oracle_loss(model,weak,transformed,views,videos,truth,mode,observations)
            weight=SPEC['weight']*min(1.,(epoch-SPEC['onset']+1)/SPEC['ramp'])
            assert torch.isfinite(loss)
            out[0]['downstream_loss']=out[0]['downstream_loss']+weight*loss
            stats['aux_loss']=float(loss.detach());stats['weight']=weight
            state['events']+=int(stats['left']+stats['right']>0)
            if stats['left']+stats['right']>0 and ctx.oracle_gradient_epoch!=epoch:
                for name,params in [('target_sgp_gradient',model.backbone_source.parameters()),('target_regression_gradient',model.reg_head.parameters())]:
                    gs=torch.autograd.grad(loss,tuple(params),retain_graph=True,allow_unused=True)
                    stats[name]=sum(float(g.detach().square().sum()) for g in gs if g is not None)**.5
                    assert math.isfinite(stats[name]) and stats[name]>0
                ctx.oracle_gradient_epoch=epoch
        ctx.oracle_last=stats
        for k,v in stats.items():out[0]['oracle_'+k]=out[0]['downstream_loss'].new_tensor(float(v)).detach()
        event=dict(step=state['step'],epoch=epoch,mode=mode,target_videos=[v['metadata']['video_id'] for v in kwargs['video_list_tgt']],stats=stats,observations=observations)
        state['digest']=hashlib.sha256((state['digest']+json.dumps(event,sort_keys=True)).encode()).hexdigest()
        if ctx.oracle_ledger:
            with Path(ctx.oracle_ledger).open('a') as f:f.write(json.dumps(dict(event,digest=state['digest']))+'\n')
        return out
    model.forward=forward


def enable_controls():
    from tools import baseline_engine as base
    if getattr(base,'oracle_controls_enabled',False):return
    previous=base.controls
    def controls(model):
        d=previous(model)
        if hasattr(model,'oracle_training_state'):d['oracle_training_state']=deepcopy(model.oracle_training_state)
        return d
    base.controls=controls;base.oracle_controls_enabled=True
