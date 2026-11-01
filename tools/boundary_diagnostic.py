"""只在固定源留出集检验定位稳定性；候选和跨视图统计均不使用GT。"""
import math
import numpy as np
import torch

SPEC = dict(schema='source-boundary-stability-diagnostic-v1', seed=123,
    source_videos_per_shift=8, crop_tokens=32, maximum_candidates=64,
    confidence_floor=.05, same_class_nms_iou=.5, minimum_views=3,
    gt_match_minimum_iou=.1, confidence_bins=4,
    gate_minimum_matched=100, gate_minimum_matched_per_shift=10,
    gate_spearman=.1, gate_positive_shifts=4, gate_confidence_matched_iou_gain=.02,
    training=False, target_data=False, target_GT=False,
    origin='SoftTeacher localization reliability principle; temporal context variation replaces 2D RoI jitter',
    reference_code='https://github.com/microsoft/SoftTeacher/blob/main/ssod/models/soft_teacher.py')


def ranges(length):
    # 步长32保持所有FPN层的物理点对齐；SGP所需160倍padding由原preprocessing处理。
    k=SPEC['crop_tokens']
    return [(0,length)] if length<=2*k else [(0,length),(0,length-k),(k,length),(k,length-k)]


def decode(points, offsets, begin=0):
    return torch.stack((points[:,0]-offsets[:,0]*points[:,3],
                        points[:,0]+offsets[:,1]*points[:,3]),-1)+begin


def iou(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=np.maximum(0,np.minimum(a[...,1],b[...,1])-np.maximum(a[...,0],b[...,0]))
    union=a[...,1]-a[...,0]+b[...,1]-b[...,0]-inter
    return inter/np.maximum(union,1e-12)


def candidates(output,length):
    rows=[]
    for level,(logit,off,pts,mask) in enumerate(zip(output['logits'],output['offsets'],output['points'],output['masks'])):
        score,label=logit[0,:,:10].sigmoid().max(-1);bounds=decode(pts,off[0])
        valid=mask[0] & (score>=SPEC['confidence_floor']) & (bounds[:,0]>=0) & (bounds[:,1]<=length) & (bounds[:,1]>bounds[:,0])
        indices=valid.nonzero().flatten()
        indices=indices[score[indices].argsort(descending=True)[:SPEC['maximum_candidates']*4]]
        for j in indices.tolist():
            rows.append(dict(level=level,point=float(pts[j,0]),stride=float(pts[j,3]),
                label=int(label[j]),score=float(score[j]),segment=bounds[j].cpu().tolist()))
    rows.sort(key=lambda x:-x['score']);kept=[]
    for row in rows:
        if any(row['label']==x['label'] and iou(row['segment'],x['segment'])>SPEC['same_class_nms_iou'] for x in kept):continue
        kept.append(row)
        if len(kept)==SPEC['maximum_candidates']:break
    return kept


def observe(row,output,begin,end):
    level=row['level'];pts=output['points'][level]
    physical=pts[:,0]+begin;indices=(physical-row['point']).abs().lt(1e-4).nonzero().flatten()
    if len(indices)!=1:return None
    j=int(indices[0]);mask=output['masks'][level][0]
    if not bool(mask[j]):return None
    prob=output['logits'][level][0,j,:10].sigmoid()
    # 不能把不同类别的点当同一个动作的定位观测。
    if int(prob.argmax())!=row['label']:return None
    bound=decode(pts[j:j+1],output['offsets'][level][0,j:j+1],begin)[0].cpu().tolist()
    if not (begin<=bound[0]<bound[1]<=end):return None
    return bound


def attach_stability(rows,outputs,windows):
    result=[]
    for row in rows:
        observations=[]
        for output,(begin,end) in zip(outputs,windows):
            bound=observe(row,output,begin,end)
            if bound is not None:observations.append(bound)
        item=dict(row,support=len(observations))
        if len(observations)>=SPEC['minimum_views']:
            values=np.array(observations);center=values.mean(0)
            item.update(mean_segment=center.tolist(),relative_std=float(values.std(0).mean()/max(center[1]-center[0],1.)))
        result.append(item)
    return result


def match_source(rows,gt_segments,gt_labels):
    """GT只在无GT候选与稳定性记录落盘后使用，评估定位误差，不修改候选。"""
    segments=np.asarray(gt_segments).reshape(-1,2);labels=np.asarray(gt_labels);matched=[]
    for row in rows:
        if 'relative_std' not in row:continue
        same=segments[labels==row['label']]
        if not len(same):
            matched.append(dict(row,base_iou=0.,mean_iou=0.,boundary_error=None,gt_supported=False));continue
        overlaps=iou(np.array(row['segment']),same);j=int(overlaps.argmax())
        target=same[j];length=max(float(target[1]-target[0]),1.)
        matched.append(dict(row,base_iou=float(overlaps[j]),mean_iou=float(iou(row['mean_segment'],target)),
            boundary_error=float(np.abs(np.array(row['segment'])-target).mean()/length),
            gt_supported=bool(overlaps[j]>=SPEC['gt_match_minimum_iou'])))
    return matched


def rank(x):
    x=np.asarray(x);order=np.argsort(x,kind='stable');out=np.empty(len(x),float)
    start=0
    while start<len(x):
        end=start+1
        while end<len(x) and x[order[end]]==x[order[start]]:end+=1
        out[order[start:end]]=(start+end-1)/2;start=end
    return out


def correlation(a,b):
    if len(a)<3:return None
    a=rank(a);b=rank(b)
    if a.std()==0 or b.std()==0:return None
    return float(np.corrcoef(a,b)[0,1])


def statistics(rows):
    n=len(rows)
    if not n:return dict(matched=0,eligible=0,spearman=None,confidence_matched_iou_gain=None)
    # 各置信度分位箱内等数量比较，避免把高分类置信误当定位稳定性。
    gains=[];ordered=sorted(rows,key=lambda x:x['score'])
    for group in np.array_split(np.arange(n),SPEC['confidence_bins']):
        items=sorted((ordered[int(i)] for i in group),key=lambda x:x['relative_std']);half=len(items)//2
        if half<2:continue
        gains.append(np.mean([x['base_iou'] for x in items[:half]])-np.mean([x['base_iou'] for x in items[-half:]]))
    supported=[x for x in rows if x['gt_supported']]
    return dict(matched=len(supported),eligible=n,spearman=correlation([x['relative_std'] for x in supported],[x['boundary_error'] for x in supported]),
        confidence_error_spearman=correlation([-x['score'] for x in supported],[x['boundary_error'] for x in supported]),
        confidence_matched_iou_gain=float(np.mean(gains)) if gains else None,
        mean_base_iou=float(np.mean([x['base_iou'] for x in rows])),
        mean_consensus_iou=float(np.mean([x['mean_iou'] for x in rows])))


def decide(groups):
    stats=[g['statistics'] for g in groups];rows=[x for g in groups for x in g['scored_rows']];pooled=statistics(rows)
    positive=sum((x['spearman'] or 0)>0 for x in stats)
    gains=[x['confidence_matched_iou_gain'] for x in stats if x['confidence_matched_iou_gain'] is not None]
    gain=float(np.mean(gains)) if len(gains)==6 else None
    passed=len(groups)==6 and pooled['matched']>=SPEC['gate_minimum_matched'] and all(x['matched']>=SPEC['gate_minimum_matched_per_shift'] for x in stats) and (pooled['spearman'] or 0)>=SPEC['gate_spearman'] and positive>=SPEC['gate_positive_shifts'] and gain is not None and gain>=SPEC['gate_confidence_matched_iou_gain']
    return dict(status='passed',mechanism_gate_passed=bool(passed),pooled=pooled,
        positive_spearman_shifts=positive,macro_confidence_matched_iou_gain=gain,
        interpretation='Gate only tests source localization reliability; it does not establish target mAP gain or novelty.')
