"""同一预先固定的竞争规则在两组 EMA 上验证；源 TRAIN 拟合只读取其本方向源标注。"""
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from witness_keep_order import competing_pairs, fit_strength

P = Path('/vepfs/vlms/class_path_formal6_e0246d8_1107/diagnostics/paired_witness_bff50e3b15b7')
OLD = P.parents[1]
NEW = Path('/vepfs/vlms/witness_pair_train6_1109_gatefix')


def load_bank(directory, info, shift, checkpoint):
    binding = info.get('binding', info.get('reference_binding'))
    assert binding['checkpoint']==checkpoint and binding['shift']==shift
    key = hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()
    groups=[]
    for f in sorted((directory/shift/key/'train').glob('*.npz')):
        with np.load(f,allow_pickle=False) as part:
            groups.append({k:part[k] for k in ('pos','neg','labels','levels','ids')})
    groups.sort(key=lambda g: str(g['ids'][0]))
    bank={k:np.concatenate([g[k] for g in groups]) for k in groups[0]}
    assert len(bank['ids'])==info.get('train_pairs',info.get('train_reference_pairs'))
    return bank


def maps_for(model,v,domain):
    x,m=model.preprocessing([v],mode='test')
    f,m,_=getattr(model,'backbone_'+domain)(model.dropout(x),m)
    f,m=getattr(model,'neck_'+domain)(f,m)
    head=model.cls_head if domain=='source' else model.cls_head_target
    z,_=head(f,m)
    return f,m,z


def fit_source(model,cfg,bank):
    from tools import baseline_engine as base
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from paired_witness_core_1109 import describe,witness_scores
    from witness_early import coordinates
    path=cfg['dataset']['json_file']
    ds=base.dataset(cfg,path,cfg['src_train_split'],False,unlabeled=True)
    gt=load_gt_seg_from_json(path,cfg['src_train_split'],action_subsampling='10_majority_classes')
    gv={str(n):g for n,g in gt[gt['t-end']>gt['t-start']].groupby('video-id')}
    margins=[]
    for i,record in enumerate(ds.data_list):
        name=str(record['id'])
        if name not in gv:
            continue
        v=ds[i]
        f,m,z=maps_for(model,v,'source')
        offsets,_=model.reg_head(f,m)
        none=[None]*len(f)
        r=model.inference_single_video(model.point_generator(f),[x[0,0] for x in m],
            [x[0].T for x in z],[x[0].T for x in offsets],none,none)
        boxes=coordinates(r['segments'],v['metadata'])
        labels=r['labels'].numpy()
        raw=r['scores'].numpy().astype(float)
        g=gv[name]
        pairs=competing_pairs(boxes,labels,raw,g[['t-start','t-end']].to_numpy(),g.label.to_numpy())
        if not len(pairs):
            continue
        ix,inv=np.unique(pairs,return_inverse=True)
        desc,lvl=describe(model,f,m,z,v['metadata'],boxes[ix],labels[ix])
        keep=bank['ids']!=name
        refs={k:bank[k][keep] for k in ('pos','neg','labels','levels')}
        b,_,covered=witness_scores(desc,labels[ix],lvl,refs)
        hi,lo=inv.reshape(-1,2).T
        valid=covered[hi]&covered[lo]
        if valid.any():
            hi,lo=hi[valid],lo[valid]
            margins.append(np.column_stack([np.log(raw[ix[hi]]/raw[ix[lo]]),np.log(b[hi]/b[lo])]))
    return fit_strength(margins)


@torch.inference_mode()
def run(line,shift,where):
    from tools import baseline_engine as base
    from tools.baseline_protocol import read,save,sha
    from tools.native_ap_fast import install
    from dense_boundary_support_1109 import score
    from witness_priority_readout import pre_nms_evidence,readout
    install()
    torch.set_num_threads(2)
    entry=(OLD if line=='old' else NEW)/shift
    cfg=read(entry/'config.json')
    digest=sha(entry/'best.pt')
    assert read(entry/'state_audit.json')['best_sha256']==digest
    if line=='old':
        info=read(P/(shift+'.json'))['source_bank']
        cache=OLD/'diagnostics/target_sigma_49f0bb250c9f'/shift
        assert read(cache/'grid.json')['raw_sha']==sha(cache/'pre_nms.pt')
        bank=load_bank(P/'banks',info,shift,digest)
    else:
        info=read(entry/'readout/result.json')
        cache=where/'exports'/shift
        bank=load_bank(entry/'readout/banks',info,shift,digest)
    model=base.make_meta_arch('sada',deepcopy(cfg),device='cpu').eval().requires_grad_(False)
    ck=torch.load(entry/'best.pt',map_location='cpu')
    model.load_state_dict(ck['ema_state_dict'],strict=True)
    del ck
    fit_path=where/(line+'_'+shift+'_source.json')
    if fit_path.is_file():
        fit=read(fit_path)
        assert fit['checkpoint']==digest
    else:
        fit=fit_source(model,cfg,bank)
        fit['checkpoint']=digest
        save(fit_path,fit)
    pool=torch.load(cache/'pre_nms.pt',map_location='cpu')
    assert pool['checkpoint']==digest and pool['config']==sha(entry/'config.json')
    dataset=base.dataset(cfg,cfg['baseline']['annotation_for_offline_evaluation'],cfg['tgt_val_split'],False,unlabeled=True)
    ids={str(row['id']):i for i,row in enumerate(dataset.data_list)}
    rows={'fixed':[],'fitted':[]}
    assert set(ids)=={r['video_id'] for r in pool['domains']['target']}
    for r in pool['domains']['target']:
        name=r['video_id']
        v=dataset[ids[name]]
        assert not v['segments'].numel() and not v['labels'].numel()
        maps=maps_for(model,v,'target')
        w,_=pre_nms_evidence(model,v,maps,r,bank)
        for method,priority in [('fixed',w),('fitted',w**fit['strength'])]:
            if method=='fitted' and fit['strength']==1.:
                rows[method].append(rows['fixed'][-1])
                continue
            b,c,s=readout(model,v,maps,r,bank,priority=priority)
            rows[method].append((np.full(len(c),name),b,c,s))
    result={}
    for method,parts in rows.items():
        vids,b,c,s=(np.concatenate(x) for x in zip(*parts))
        names,idx=np.unique(vids,return_inverse=True)
        result[method]=score(cfg,dict(names=names,idx=idx,boxes=b,labels=c),s)
    if line=='old':
        expected=read(P/'priority_d57f031d57ae'/(shift+'.json'))['metrics']['real']
        np.testing.assert_allclose(result['fixed']['by_tiou'],expected['by_tiou'],atol=1e-5,rtol=0)
    return dict(checkpoint=digest,fit=fit,metrics=result,shift=shift,line=line)


if __name__=='__main__':
    from tools.baseline_protocol import save
    line,shift=sys.argv[1:]
    where=Path(__file__).resolve().parent
    save(where/(line+'_'+shift+'.json'),run(line,shift,where))
