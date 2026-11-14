"""冻结候选池内，用模块B给原生 Soft-NMS 提供优先级；候选及特征只来自无标注推理。"""
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch


def coordinates(grid, metadata):
    if metadata['use_temporal_grid_feats']:
        return ((grid * metadata['feat_stride'] + .5*metadata['feat_num_frames']) /
                metadata['fps']).clamp(0,metadata['duration']).numpy()
    return grid.clamp(0,metadata['video_length']).numpy()


def gt_coverage(truth, name, boxes, labels, scores, limit):
    gt = truth[truth['video-id']==name]
    order = np.argsort(-scores,kind='stable')[:limit]
    boxes,labels = boxes[order],labels[order]
    result = np.zeros((len(gt),2),bool)
    for i,(l,a,b) in enumerate(gt[['label','t-start','t-end']].itertuples(index=False,name=None)):
        x=boxes[labels==l]
        if len(x):
            inter=np.maximum(0,np.minimum(x[:,1],b)-np.maximum(x[:,0],a))
            result[i] = (inter/np.maximum(x[:,1]-x[:,0]+b-a-inter,1e-12)).max() >= np.array([.3,.5])
    return result


@torch.inference_mode()
def evaluate(primary,shift,out):
    from tools import baseline_engine as engine
    from tools.native_ap_fast import install
    from tools.baseline_protocol import read,sha
    from libs.utils import batched_nms
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from dense_boundary_support_1109 import score
    from paired_witness_inference import load_source_bank, _prototype_cosines
    from paired_witness_core_1109 import describe,witness_scores
    from ours_final import profiles,responses
    from uniform_tricks import transform
    from candidate_evidence_1109 import score_candidates
    torch.set_num_threads(2)
    install()
    spec=read(primary/'spec.json')
    entry=Path(spec['training_root'])/shift
    cfg=read(entry/'config.json')
    test=cfg['test_cfg']
    cache=entry.parent/'diagnostics'/'target_sigma_49f0bb250c9f'/shift
    assert sha(cache/'pre_nms.pt')==read(cache/'grid.json')['raw_sha']
    pool=torch.load(cache/'pre_nms.pt',map_location='cpu')
    checkpoint=sha(entry/'best.pt')
    assert checkpoint==pool['checkpoint']==read(entry/'state_audit.json')['best_sha256']
    assert pool['config']==sha(entry/'config.json') and test['multiclass_nms'] and test['nms_method']=='soft'
    model=engine.make_meta_arch('sada',deepcopy(cfg),device='cpu').eval().requires_grad_(False)
    state=torch.load(entry/'best.pt',map_location='cpu')
    model.load_state_dict(state['ema_state_dict'],strict=True)
    del state
    bank=load_source_bank(primary,shift,checkpoint)
    plan=read(Path(spec['archive'])/'plan.json')['base']
    inputs=Path(plan['inputs'])/('native_'+shift+'_input.npz')
    assert sha(inputs)==read(Path(spec['evidence'])/(shift+'.json'))['input_sha256']
    with np.load(inputs) as f: old=dict(f)
    ds=engine.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,unlabeled=True)
    indices={str(v['id']):i for i,v in enumerate(ds.data_list)}
    truth=load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],action_subsampling='10_majority_classes')
    truth=truth[truth['t-end']>truth['t-start']]
    methods=('control','early_priority','early_and_late')
    parts={k:[] for k in methods}
    coverage={k:{j:np.zeros((2,2),int) for j in ('all','100')} for k in methods}
    changes=dict(pre=0,baseline=0,early=0,new_geometries=0,removed_geometries=0,gt=0)
    checked=set()
    for v in pool['domains']['target']:
        name=v['video_id']
        assert name not in checked
        checked.add(name)
        vid=ds[indices[name]]
        assert not vid['labels'].numel() and not vid['segments'].numel()
        x,m=model.preprocessing([vid],mode='test')
        f,m,_=model.backbone_target(model.dropout(x),m)
        f,m=model.neck_target(f,m)
        z,_=model.cls_head_target(f,m)
        meta=vid['metadata']
        grid,raw,lab=[v[k] for k in ('segments','scores','labels')]
        boxes=coordinates(grid,v)
        keys,inverse=np.unique(np.column_stack([boxes,lab.numpy()]),axis=0,return_inverse=True)
        lookup={tuple(key):i for i,key in enumerate(keys)}
        desc,levels=describe(model,f,m,z,meta,keys[:,:2],keys[:,2].astype(int))
        w=witness_scores(desc,keys[:,2].astype(int),levels,bank)[0]
        factor=torch.from_numpy(w[inverse]).to(raw)
        def select(priority):
            g,s,c=batched_nms(grid.clone(),priority.clone(),lab.clone(),test['iou_threshold'],
                test['min_score'],test['max_seg_num'],use_soft_nms=True,multiclass=True,sigma=test['nms_sigma'],voting_thresh=test['voting_thresh'])
            b=coordinates(g,v)
            take=np.array([lookup[tuple(row)] for row in np.column_stack([b,c.numpy()])],dtype=int)
            return b.astype(float),s.numpy().astype(float),c.numpy(),take
        b0,s0,c0,k0=select(raw)
        b1,s1,c1,k1=select(raw*factor)
        idx=np.flatnonzero(old['names'][old['idx']]==name)
        for value,ref in [(b0,old['boxes'][idx]),(s0,old['scores'][idx]),(c0,old['labels'][idx])]:
            np.testing.assert_array_equal(value,ref)
        changes['pre']+=len(grid)
        changes['baseline']+=len(k0)
        changes['early']+=len(k1)
        changes['new_geometries']+=len(set(k1)-set(k0))
        changes['removed_geometries']+=len(set(k0)-set(k1))
        changes['gt']+=int((truth['video-id']==name).sum())
        ids=np.union1d(k0,k1)
        tr=(meta['fps'],.5*meta['feat_num_frames'],meta['feat_stride']) if meta['use_temporal_grid_feats'] else (1.,0.,1.)
        seg=torch.tensor(keys[ids,:2].copy(),dtype=torch.float32)
        cos,ready=_prototype_cosines(model,f,(seg*tr[0]-tr[1])/tr[2],torch.as_tensor(levels[ids]))
        dense=dict(transform=tr,ranges=model.rq3_capacity.regression_ranges,strides=model.fpn_strides)
        for l,(mask,logit) in enumerate(zip(m,z)):
            dense['q'+str(l)]=logit[0,:,:int(mask[0].sum())].sigmoid().numpy()
        center=transform(dense,'temperature2')
        control_cov={}
        for method,b,sc,c,k in [('control',b0,s0,c0,k0),('early_priority',b1,s1/w[k1],c1,k1),('early_and_late',b1,s1,c1,k1)]:
            p=profiles(center,b,c)
            p[:,3]=w[k]
            val=score_candidates(b,sc,c,cos[np.searchsorted(ids,k)],ready,p,responses(center,b,len(ready)))
            parts[method].append((np.full(len(b),name),b,c,val))
            for j,limit in [('all',len(b)),('100',100)]:
                hits=gt_coverage(truth,name,b,c,val,limit)
                if method=='control':control_cov[j]=hits
                coverage[method][j]+=np.stack([((~control_cov[j])&hits).sum(0),(control_cov[j]&(~hits)).sum(0)])
        print(name,len(grid),len(k1),flush=True)
    assert checked==set(old['names'])
    result={}
    for method,rows in parts.items():
        videos,b,c,s=(np.concatenate(col) for col in zip(*rows))
        names,idx=np.unique(videos,return_inverse=True)
        result[method]=score(cfg,dict(names=names,idx=idx,boxes=b,labels=c),s)
    np.testing.assert_allclose(result['control']['by_tiou'],read(primary/(shift+'.json'))['real']['by_tiou'],atol=1e-5,rtol=0)
    return dict(metrics=result,counts=changes,checkpoint=checkpoint,shift=shift,
        coverage={k:{j:v.tolist() for j,v in row.items()} for k,row in coverage.items()})
