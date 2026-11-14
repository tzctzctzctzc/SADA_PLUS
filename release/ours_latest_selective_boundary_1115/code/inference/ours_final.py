"""Ours 六向共享读出。原生候选、类别和端点固定，仅重新计算候选分数。"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

POLICIES = ('global','two_edges','global_and_edges','global_odds')


def decay(boxes, scores):
    """同视频同类、稳定分数排序；自身已遭抑制的框对后续框影响较小。"""
    order = np.argsort(-scores, kind='stable')
    start, end = boxes[order].T
    width = end - start
    inter = np.maximum(0., np.minimum(end[:, None], end) - np.maximum(start[:, None], start))
    union = width[:, None] + width - inter
    overlap = np.triu(np.divide(inter, union, out=np.zeros_like(inter), where=union > 0), 1)
    compensation = overlap.max(axis=0)
    strongest = np.maximum(0., (overlap**2 - compensation[:, None]**2).max(axis=0))
    result = scores.copy()
    result[order] *= np.exp(-2. * strongest)
    return result


def all_rival_factor(cos, labels, ready):
    output = torch.ones(len(cos),dtype=cos.dtype)
    n = int(ready.sum())
    if n < 2:
        return output
    own = cos.gather(1,labels[:,None])[:,0]
    rivals = cos.masked_fill(~ready,-torch.inf)
    rivals.scatter_(1,labels[:,None],-torch.inf)
    other = rivals.logsumexp(1) - math.log(n-1)
    valid = ready[labels]
    output[valid] = 2 * (own[valid]-other[valid]).sigmoid()
    return output


def contrast(q,boxes,labels):
    """在有效网格内精确积分分段线性类别响应，与已有四分之一框宽外带定义相同。"""
    n=q.shape[1]
    if n<2:
        return np.ones((2,len(boxes)))
    q=q.astype(float)
    slope=np.diff(q,axis=1)
    prefix=np.pad(np.cumsum(.5*(q[:,1:]+q[:,:-1]),axis=1),((0,0),(1,0)))
    a,b=boxes.T
    margin=(b-a)/4
    l,a,x,y,b,r=np.clip([a-margin,a,a+margin,b-margin,b,b+margin],0,n-1)
    def integral(t):
        j=np.minimum(t.astype(int),n-2)
        u=t-j
        return prefix[labels,j]+u*q[labels,j]+.5*u*u*slope[labels,j]
    il,ia,ix,iy,ib,ir=map(integral,(l,a,x,y,b,r))
    def difference(inside,wi,outside,wo):
        ok=(wi>1e-8)&(wo>1e-8)
        values=np.ones(len(boxes))
        values[ok]=1+(inside[ok]/wi[ok]).clip(0,1)-(outside[ok]/wo[ok]).clip(0,1)
        return values,ok
    g,_=difference(ib-ia,b-a,(ia-il)+(ir-ib),(a-l)+(r-b))
    left,vl=difference(ix-ia,x-a,ia-il,a-l)
    right,vr=difference(ib-iy,b-y,ir-ib,r-b)
    edge=np.where(vl&vr,np.sqrt(left*right),np.where(vl,left,right))
    return np.stack((g,edge))


def responses(cache, boxes, n_classes):
    fps, offset, stride = cache['transform']
    grid = (boxes.astype(float)*fps-offset)/stride
    radius = np.maximum(np.diff(grid, axis=1).ravel(), 1.)/2
    levels = np.full(len(radius), len(cache['strides'])-1)
    for level, (lo, hi) in reversed(list(enumerate(cache['ranges']))):
        levels[(lo<=radius)&(radius<=hi)] = level
    result = np.empty((len(grid), n_classes))
    for level in np.unique(levels):
        ix = np.flatnonzero(levels == level)
        q = cache['q'+str(level)]
        assert len(q) == n_classes
        coords = grid[ix]/cache['strides'][level]
        for c in range(n_classes):
            result[ix, c] = contrast(q, coords, np.full(len(ix), c))[0]
    return result


def profiles(cache, boxes, labels):
    fps,offset,stride=cache['transform']
    grid = (boxes.astype(float)*fps-offset)/stride
    width = grid[:,1] - grid[:,0]
    radius = np.maximum(width,1.)/2
    levels = np.full(len(boxes),len(cache['ranges'])-1)
    for i,(low,high) in reversed(list(enumerate(cache['ranges']))):
        levels[(low<=radius)&(radius<=high)] = i
    result = np.ones((len(boxes),len(POLICIES)))
    for level in np.unique(levels):
        ix = np.flatnonzero(levels==level)
        q = cache['q'+str(level)].astype(float)
        n = q.shape[1]
        if n<2:
            continue
        diff = np.diff(q,axis=1)
        prefix = np.pad(np.cumsum(.5*(q[:,1:]+q[:,:-1]),axis=1),((0,0),(1,0)))
        classes = labels[ix].astype(int)
        start,end = (grid[ix]/cache['strides'][level]).T
        quarter = (end-start)/4
        l,a,x,y,b,r = np.clip([start-quarter,start,start+quarter,end-quarter,end,end+quarter],0,n-1)
        def integral(t):
            j = np.minimum(t.astype(int),n-2)
            u = t-j
            return prefix[classes,j]+u*q[classes,j]+.5*(u*u)*diff[classes,j]
        il,ia,ixx,iy,ib,ir = (integral(t) for t in (l,a,x,y,b,r))
        ok = ((b-a)>1e-8)&(((a-l)+(r-b))>1e-8)
        inner = np.zeros(len(ix))
        outer = np.zeros(len(ix))
        inner[ok] = ((ib-ia)[ok]/(b-a)[ok]).clip(0,1)
        outer[ok] = (((ia-il)+(ir-ib))[ok]/((a-l)+(r-b))[ok]).clip(0,1)
        g = 1+inner-outer

        def side(inside_mass, inside_width, outside_mass, outside_width):
            valid = (inside_width>1e-8)&(outside_width>1e-8)
            f = np.ones(len(ix))
            f[valid] += inside_mass[valid]/inside_width[valid] - outside_mass[valid]/outside_width[valid]
            return f.clip(0,2),valid
        left,vl = side(ixx-ia,x-a,ia-il,a-l)
        right,vr = side(ib-iy,b-y,ir-ib,r-b)
        # 一侧越出视频时只用可见侧；双侧均不可用时中性。
        e = np.where(vl&vr,np.sqrt(left*right),np.where(vl,left,right))
        pos, neg = inner*(1-outer), outer*(1-inner)
        odds = np.divide(2*pos,pos+neg,out=np.ones(len(ix)),where=(pos+neg)>0)
        result[ix] = np.stack((g,e,np.sqrt(g*e),odds),axis=1)
    assert np.isfinite(result).all() and result.min()>=0 and result.max()<=2+1e-10
    return result


def final_scores(boxes, raw, labels, cos, ready, temporal, all_contrasts):
    if not len(raw):
        return raw.copy()
    row = np.arange(len(labels))
    classes = torch.as_tensor(labels).long()
    valid = torch.as_tensor(ready,dtype=torch.bool)
    similarity = torch.from_numpy(cos)
    rival = all_rival_factor(similarity,classes,valid).numpy()
    boost = np.ones(len(raw),dtype=cos.dtype)
    if valid.any():
        boost = torch.where(valid[classes],2*similarity[:,valid].amax(1).sigmoid(),torch.ones_like(similarity[:,0])).numpy()
    branches = []
    for weight in (temporal[:,0],temporal[:,3]):
        initial = raw.astype(float)*weight*rival
        ranked = initial.copy()
        for c in np.unique(labels):
            ix = np.flatnonzero(labels==c)
            ranked[ix] = decay(boxes[ix],initial[ix])
        branches.append(ranked)
    base = (branches[0]+.5*(branches[1]-branches[0]))*boost
    own = all_contrasts[row,labels]
    competitor = all_contrasts.argmax(1)
    top = all_contrasts[row,competitor]
    mask = (competitor!=labels)&(top>1.)&((all_contrasts==top[:,None]).sum(1)==1)
    discount = np.ones(len(raw))
    discount[mask] = 2*own[mask]/(own[mask]+top[mask])
    return base*discount


def video_scores(model,video,maps,boxes,raw,labels):
    if not len(labels):
        return raw.copy()
    meta = video['metadata']
    cfg_ranges = model.rq3_capacity.regression_ranges
    tr = [meta['fps'],.5*meta['feat_num_frames'],meta['feat_stride']] if meta['use_temporal_grid_feats'] else [1.,0.,1.]
    cache = dict(transform=tr,ranges=cfg_ranges,strides=model.fpn_strides)
    for i,(mask,z) in enumerate(zip(maps[1],maps[2])):
        cache['q'+str(i)] = z[0,:,:int(mask[0].sum())].sigmoid().numpy()
    b = torch.as_tensor(boxes,dtype=torch.float32)
    b = (b*tr[0]-tr[1])/tr[2]
    radius = (b[:,1]-b[:,0]).clamp_min(1)/2
    levels = torch.full((len(b),),len(cfg_ranges)-1,dtype=torch.long)
    for i,(lo,hi) in reversed(list(enumerate(cfg_ranges))):
        levels[(lo<=radius)&(radius<=hi)] = i
    branch = model.rq3_capacity.branch
    q = torch.cat([branch.encode(maps[0],[{'segments':b[j:j+128],'levels':levels[j:j+128]}],model.fpn_strides)['embedding'] for j in range(0,len(b),128)])
    ready = branch.source_prototype_ready
    bank = F.normalize(branch.source_prototypes,dim=-1)
    if ready.any():
        axis = F.normalize(bank[ready].mean(0),dim=0)
        bank = F.normalize(bank-(bank@axis)[:,None]*axis,dim=-1)
        q = F.normalize(q-(q@axis)[:,None]*axis,dim=-1)
    cos = (q@bank.T).numpy()
    temporal = profiles(cache,boxes,labels)
    evidence = responses(cache,boxes,len(bank))
    np.testing.assert_allclose(temporal[:,0],evidence[np.arange(len(labels)),labels],rtol=0,atol=1e-7)
    return final_scores(boxes,raw,labels,cos,ready.numpy(),temporal,evidence)


@torch.inference_mode()
def evaluate_entry(entry,out,plan):
    from tools.baseline_protocol import read,save,sha
    from tools.generalization_models import DetectorWithAdaptation,native_videos
    from tools import baseline_engine as engine
    from tools.full_readout import score
    from tools.native_ap_fast import install
    from tools.baseline_gpu import first_cuda_guard
    from libs.trainer.da_trainer import move_to_device
    first_cuda_guard()
    install()
    torch.set_num_threads(2)
    cfg=read(entry/'config.json')
    audit=read(entry/'audit.json')
    assert audit['passed'] and audit['source_selected'] and audit['target_GT_training'] is False and audit['seed']==123
    assert cfg['test_cfg']==plan['test_cfg']
    checkpoint=sha(entry/'best.pt')
    assert checkpoint==audit['checkpoint_sha256']==plan['hashes'][audit['detector']+'/'+audit['shift']]
    engine.fix_random_seed(123,include_cuda=True)
    saved=torch.load(entry/'best.pt',map_location='cpu')
    assert saved['epoch']==audit['selected_epoch']
    gpu=DetectorWithAdaptation(cfg,'cuda:0').to('cuda:0').eval()
    cpu=DetectorWithAdaptation(cfg,'cpu').eval()
    for model in (gpu,cpu):
        model.load_state_dict(saved['ema_state_dict'],strict=True)
    del saved
    ds=engine.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,unlabeled=True)
    rows=[]
    for i in range(len(ds)):
        video=ds[i]
        assert not video['labels'].numel() and not video['segments'].numel()
        dev=move_to_device([dict(video)],'cuda:0')
        pred=gpu.native(native_videos(dev))[0]
        boxes=pred['segments'].cpu().numpy().astype(float)
        raw=pred['scores'].cpu().numpy().astype(float)
        labels=pred['labels'].cpu().numpy().astype(int)
        pyramid,masks,_,logits=cpu.dense([video])
        final=video_scores(cpu,video,(pyramid,masks,logits),boxes,raw,labels)
        rows.append(pd.DataFrame({'video-id':pred['video_id'],'t-start':boxes[:,0],'t-end':boxes[:,1],'label':labels,'raw':raw,'score':final}))
    df=pd.concat(rows,ignore_index=True)
    assert len(df)==audit['results']['target']['prediction_count'] and np.isfinite(df.score).all()
    fields=df[['video-id','t-start','t-end','label']].to_dict('list')
    raw_metric=score(cfg,dict(fields,score=df.raw.to_numpy()))
    np.testing.assert_allclose(raw_metric['by_tiou'],100*np.asarray(audit['results']['target']['map_by_tiou']),rtol=0,atol=1e-5)
    result=score(cfg,dict(fields,score=df.score.to_numpy()))
    out.mkdir(parents=True,exist_ok=True)
    torch.save(dict(fields,score=df.score.to_numpy()),out/'predictions.pt')
    save(out/'result.json',dict(detector=audit['detector'],shift=audit['shift'],checkpoint_sha256=checkpoint,
         selected_epoch=audit['selected_epoch'],native=raw_metric,ours=result,predictions=len(df),videos=len(ds),
         predictions_sha256=sha(out/'predictions.pt'),rule='dense_discount',seed=123))


if __name__=='__main__':
    evaluate_entry(Path(sys.argv[1]),Path(sys.argv[2]),json.loads(Path(sys.argv[3]).read_text()))
