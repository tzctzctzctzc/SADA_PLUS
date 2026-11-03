"""相同高分候选的层信号、时长选层和非视觉条件诊断。"""
import hashlib
import numpy as np
import torch
from tools.background_linear_temporal import region_samples

STAGES=('raw_RP512','embedding','SGP','FPN0','duration_scale')
MODES=tuple(m for s in STAGES for m in (s,s+'_centered'))+('metadata',)
FAMILIES={'early':tuple(m for s in STAGES[:2] for m in (s,s+'_centered')),
          'temporal':tuple(m for s in STAGES[2:] for m in (s,s+'_centered'))+('metadata',)}


def raw_projection():
    g=torch.Generator().manual_seed(311106)
    return torch.randn(2304,512,generator=g)/np.sqrt(512)


def duration_levels(boxes,ranges):
    """按原预测半时长落入回归范围选层；不是追溯NMS后的真实来源层。"""
    half=(boxes[:,1]-boxes[:,0])/2
    distance=torch.stack([torch.relu(a-half)+torch.relu(half-b) for a,b in ranges],1)
    # 范围边界多层同时有效时固定取较细一层，不看GT或结果。
    return distance.argmin(1)


def pool_signals(feature,valid,boxes,stride=1):
    values,masks=region_samples(feature,valid,boxes/stride)
    def mean(v,m):return (v*m[...,None]).sum(1)/m.sum(1,keepdim=True).clamp_min(1)
    left,inside,right=[mean(v,m) for v,m in zip(values,masks)]
    inner=values[1];ok=masks[1];pair=ok[:,:-1]&ok[:,1:]
    changes=(inner[:,1:]-inner[:,:-1]).norm(dim=-1)
    change=(changes*pair).sum(1)/pair.sum(1).clamp_min(1)
    scale=(inner.norm(dim=-1)*ok).sum(1)/ok.sum(1).clamp_min(1)
    contrast=(inside-(left+right)/2).norm(dim=1)
    statistics=torch.stack((inside.norm(dim=1),change/scale.clamp_min(1e-8),contrast/inside.norm(dim=1).clamp_min(1e-8),ok.float().mean(1)),1)
    global_mean=feature[:,valid].mean(1) if valid.any() else feature.new_zeros(feature.shape[0])
    return inside,global_mean,statistics


def conditional_auc(bg,scores,classes,original_scores,lengths,videos,edges):
    """同类/同原分数档/同时长档、不同视频的BG-vs-nonBG条件排序。"""
    bins=[]
    for c,s,l in zip(classes,original_scores,lengths):
        e=edges[str(int(c))]
        bins.append((int(c),int(np.searchsorted(e['score'],s,side='right')),int(np.searchsorted(e['length'],l,side='right'))))
    values={};anchors=0;pairs=0
    for key in sorted(set(bins)):
        ids=np.array([i for i,k in enumerate(bins) if k==key],dtype=int)
        for i in ids[np.asarray(bg)[ids]]:
            neg=ids[(~np.asarray(bg)[ids])&(np.asarray(videos)[ids]!=videos[i])]
            if not len(neg):continue
            value=float(((scores[i]>scores[neg])+.5*(scores[i]==scores[neg])).mean())
            values.setdefault(str(videos[i]),[]).append(value);anchors+=1;pairs+=len(neg)
    macros=[float(np.mean(x)) for x in values.values()]
    return dict(video_macro_conditional_AUC=float(np.mean(macros)) if macros else None,valid_videos=len(macros),BG_anchors=anchors,pairs=pairs,
                caveat='pairs share videos/anchors; not independent samples')


def video_geometry(features,classes,original_scores,lengths,videos,edges,boxes):
    """无GT选查询；同条件最近邻的视频聚集只能支持捷径线索，不能证明因果。"""
    z=np.asarray(features,dtype=np.float64);z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
    bins=[]
    for c,s,l in zip(classes,original_scores,lengths):
        e=edges[str(int(c))];bins.append((int(c),np.searchsorted(e['score'],s,side='right'),np.searchsorted(e['length'],l,side='right')))
    query={}
    for i,(v,c) in enumerate(zip(videos,classes)):
        digest=hashlib.sha256(('layer-signal-query-v1|'+str(v)+'|'+str(int(c))+'|'+str(i)).encode()).hexdigest()
        key=(str(v),int(c))
        if key not in query or digest<query[key][0]:query[key]=(digest,i)
    hit=[];chance=[]
    for _,i in query.values():
        candidates=np.array([j for j,k in enumerate(bins) if k==bins[i] and j!=i],dtype=int)
        if len(candidates):
            intersection=np.maximum(0,np.minimum(boxes[candidates,1],boxes[i,1])-np.maximum(boxes[candidates,0],boxes[i,0]))
            union=(boxes[candidates,1]-boxes[candidates,0])+(boxes[i,1]-boxes[i,0])-intersection
            # 排除同视频重叠框，避免最近邻仅仅是同一时刻的重复proposal。
            candidates=candidates[(np.asarray(videos)[candidates]!=videos[i])|(intersection/np.maximum(union,1e-12)<.1)]
        if not len(candidates):continue
        similarity=z[candidates]@z[i];j=int(candidates[np.lexsort((candidates,-similarity))[0]])
        hit.append(float(videos[j]==videos[i]));chance.append(float(np.mean(np.asarray(videos)[candidates]==videos[i])))
    overall=z.mean(0);total=float(((z-overall)**2).sum());between=0.
    for v in sorted(set(videos)):
        subset=z[np.asarray(videos)==v];between+=len(subset)*float(((subset.mean(0)-overall)**2).sum())
    return dict(queries=len(hit),same_video_nearest_fraction=float(np.mean(hit)) if hit else None,matched_cell_same_video_chance=float(np.mean(chance)) if chance else None,
                between_video_variance_fraction=between/total if total else None,
                same_video_overlap_excluded_at_IoU=.1,
                limitation='nonoverlapping neighbors may still share temporal context/action distribution; not proof of scene-only shortcut')
