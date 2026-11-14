"""已有定位头的中心回归支持；不等于后NMS候选投票，不移动候选边界。"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,allow_nan=False));temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def support(boxes, labels, cache, fields, radius):
    """沿用训练中心半径和回归范围；类分数仅作为固定权重，缺失支持中性。"""
    fps,offset,stride=cache['transform']
    grid=(boxes.astype(float)*fps-offset)/stride
    half=np.maximum(grid[:,1]-grid[:,0],1)/2
    levels=np.full(len(boxes),len(cache['strides'])-1)
    for level,(lo,hi) in reversed(list(enumerate(cache['ranges']))):
        levels[(half>=lo)&(half<=hi)]=level
    agreement=np.ones(len(boxes));counts=np.zeros(len(boxes),dtype=int)
    for level in np.unique(levels):
        take=np.flatnonzero(levels==level);b=grid[take];center=b.mean(1)
        points=fields['points'+str(level)];prediction=fields['segments'+str(level)]
        stride_level=float(cache['strides'][level]);reach=int(np.ceil(radius))
        origin=points[0]
        base=np.floor((center-origin)/stride_level).astype(int)
        indices=base[:,None]+np.arange(-reach,reach+1)[None]
        valid=(indices>=0)&(indices<len(points));safe=np.clip(indices,0,len(points)-1)
        x=points[safe]
        left=np.maximum(b[:,0],center-radius*stride_level)
        right=np.minimum(b[:,1],center+radius*stride_level)
        valid &= (x>left[:,None])&(x<right[:,None])
        maximum=np.maximum(x-b[:,:1],b[:,1:]-x)
        lo,hi=cache['ranges'][level]
        valid &= (maximum>=lo)&(maximum<=hi)
        pred=prediction[safe]
        valid &= pred[:,:,1]>pred[:,:,0]
        # 同一温度2与既有共享时间证据一致；没有再选择一个置信阈值。
        q=cache['q'+str(level)].astype(float)
        root=np.sqrt(q);other=np.sqrt(1-q);q=root/(root+other)
        weight=q[labels[take,None],safe]*valid
        inter=np.maximum(0,np.minimum(pred[:,:,1],b[:,1:])-np.maximum(pred[:,:,0],b[:,:1]))
        union=pred[:,:,1]-pred[:,:,0]+b[:,1:]-b[:,:1]-inter
        overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
        mass=weight.sum(1);value=(weight*overlap).sum(1)
        agreement[take]=np.divide(value,mass,out=np.ones_like(value),where=mass>0)
        counts[take]=valid.sum(1)
    assert np.isfinite(agreement).all() and agreement.min()>=0 and agreement.max()<=1+1e-10
    return np.clip(agreement,0,1),counts


def score(cfg,data,values):
    from libs.evaluation import ANETdetection
    evaluator=ANETdetection(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],
                          num_workers=2,action_subsampling='10_majority_classes')
    columns={'video-id':data['names'][data['idx']],'t-start':data['boxes'][:,0],
             't-end':data['boxes'][:,1],'label':data['labels'],'score':values}
    ap,mean,*_=evaluator.evaluate(columns,verbose=False)
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from tools.audit_background_ap_cost import IndependentAP
    truth=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],action_subsampling='10_majority_classes')
    truth=truth[truth['t-start']<truth['t-end']].reset_index(drop=True);frame=pd.DataFrame(columns)
    independent=np.array([IndependentAP(truth[truth.label==c].reset_index(drop=True),frame[frame.label==c].reset_index(drop=True)).ap(
        np.ones(int((frame.label==c).sum()),dtype=bool)) for c in sorted(truth.label.unique())]).mean(0)
    np.testing.assert_allclose(ap,independent,atol=1e-7,rtol=0)
    return dict(mAP=float(mean)*100,by_tiou=(100*np.asarray(ap)).tolist(),independent_AP_passed=True)


@torch.inference_mode()
def run(out,shift):
    from tools import baseline_engine as base
    from tools.native_ap_fast import install
    install();torch.set_num_threads(2);torch.manual_seed(123)
    spec=read(out/'spec.json');archive=Path(spec['archive']);plan=read(archive/'plan.json')
    summary=read(archive/'summary.json');entry=Path(plan['base']['inputs']).parents[2]/shift
    cfg,audit=read(entry/'config.json'),read(entry/'state_audit.json')
    assert audit['status']=='passed' and not cfg['model']['use_trident_head']
    checkpoint=sha(entry/'best.pt');assert checkpoint==audit['best_sha256']
    assert cfg['model']['center_sample']=='radius'
    radius=float(cfg['model']['center_sample_radius'])
    with np.load(Path(plan['base']['inputs'])/('native_'+shift+'_input.npz')) as f:data=dict(f)
    chosen=[i for i,r in enumerate(plan['factor_rules']) if all(r[k]==summary['common_rule'][k] for k in ['temp','sigma','blend','proto_power'])]
    assert len(chosen)==1 and summary['common_rule']['width']==1 and summary['common_rule']['duration_power']==0
    with np.load(archive/(shift+'_factors.npz')) as f:
        old=f['scores'][chosen[0]]
        assert str(f['input_sha256'])==sha(Path(plan['base']['inputs'])/('native_'+shift+'_input.npz'))
    control=score(cfg,data,old)
    np.testing.assert_allclose(control['by_tiou'],summary['per_shift'][shift]['by_tiou'],atol=1e-5,rtol=0)
    model=base.make_meta_arch('sada',deepcopy(cfg),device='cpu').cpu().eval().requires_grad_(False)
    state=torch.load(entry/'best.pt',map_location='cpu');model.load_state_dict(state['ema_state_dict'],strict=True)
    del state;gc.collect();before=base.tensor_digest(model.state_dict())
    ds=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,unlabeled=True)
    lookup={str(row['id']):i for i,row in enumerate(ds.data_list)}
    destination=out/shift;destination.mkdir(exist_ok=True);cache_out=destination/'fields';cache_out.mkdir(exist_ok=True)
    values=np.ones(len(old));counts=np.zeros(len(old),dtype=int);deltas=[];start=time.time()
    for j,name in enumerate(data['names']):
        take=np.flatnonzero(data['idx']==j);filename=hashlib.sha256(str(name).encode()).hexdigest()+'.npz'
        old_cache=Path(plan['base']['dense'])/'dense'/'old'/shift/filename
        with np.load(old_cache) as f:cache=dict(f)
        file=cache_out/filename
        if not file.exists():
            video=ds[lookup[str(name)]]
            assert not video['labels'].numel() and not video['segments'].numel()
            x,mask=model.preprocessing([video],mode='test')
            f,mask,_=model.backbone_target(model.dropout(x),mask);f,mask=model.neck_target(f,mask)
            logits,_=model.cls_head_target(f,mask);offsets,_=model.reg_head_target(f,mask);points=model.point_generator(f)
            fields={}
            for level in range(len(f)):
                valid=int(mask[level][0].sum());q=logits[level][0,:,:valid].sigmoid().numpy()
                expected=cache['q'+str(level)];assert q.shape==expected.shape
                delta=float(np.max(np.abs(q-expected)));np.testing.assert_allclose(q,expected,atol=2e-5,rtol=1e-5)
                deltas.append(delta)
                pts=points[level][:valid].numpy();reg=offsets[level][0,:,:valid].T.numpy()
                seg=np.column_stack([pts[:,0]-reg[:,0]*pts[:,3],pts[:,0]+reg[:,1]*pts[:,3]])
                fps,offset,stride=cache['transform'];limit=video['metadata']['duration']
                seg=np.clip((seg*stride+offset)/fps,0,limit);seg=(seg*fps-offset)/stride
                fields['points'+str(level)]=pts[:,0];fields['segments'+str(level)]=seg
            with file.with_suffix('.tmp').open('wb') as stream:np.savez_compressed(stream,**fields)
            file.with_suffix('.tmp').replace(file)
        with np.load(file) as f:fields=dict(f)
        values[take],counts[take]=support(data['boxes'][take],data['labels'][take],cache,fields,radius)
        save(out/(shift+'_progress.json'),dict(stage='regression_field',videos=j+1,total=len(data['names']),seconds=time.time()-start))
    assert before==base.tensor_digest(model.state_dict())
    del model;gc.collect()
    # 两个预定且六向共用的强度；无逐方向选参，框/类别/候选数保持原样。
    results={'control':control,'field_support':score(cfg,data,old*values),
             'half_field_support':score(cfg,data,old*np.sqrt(values))}
    with (destination/'support.npz').open('wb') as f:np.savez_compressed(f,agreement=values,counts=counts)
    save(out/(shift+'.json'),dict(status='passed',shift=shift,checkpoint_sha256=checkpoint,
        results=results,center_radius_from_original_training=radius,
        missing_support=int((counts==0).sum()),single_support=int((counts==1).sum()),
        mean_support=float(values.mean()),candidates=len(old),videos=len(data['names']),
        max_dense_response_difference=max(deltas) if deltas else None,frozen_model_unchanged=True,
        full_training_runs=0,target_train_GT=False,target_parameter_fit=False,seconds=time.time()-start))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--shift',required=True);a=parser.parse_args()
    if not (a.out/(a.shift+'.json')).exists():run(a.out,a.shift)


if __name__=='__main__':main()
