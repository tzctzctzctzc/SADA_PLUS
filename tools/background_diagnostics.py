"""只读诊断定义。错误计数非可加mAP分解；GT oracle不能用于方法筛选/阈值拟合。"""
from collections import Counter
import numpy as np

SCHEMA='prototype-background-diagnostics-v1'
SPEC=dict(roles=['student','ema'],shifts=['S1','S2','S3','S4','S5','S6'],
    primary_model='original-source-best-EMA',tiou=[.1,.2,.3,.4,.5],background_iou=.01,
    oracle='same dense predictions, mask original GT-unassigned points before topk/NMS',
    occupancy_oracle='mask points outside annotated task action spans before topk/NMS',
    target_gt='post-prediction exploratory diagnosis only; never routing parameters or training',
    new_training=False,threshold_search=False)


def iou(segment,segments):
    segments=np.asarray(segments,dtype=float).reshape(-1,2)
    if not len(segments):return np.empty(0)
    overlap=np.maximum(0,np.minimum(segment[1],segments[:,1])-np.maximum(segment[0],segments[:,0]))
    union=np.maximum(segment[1]-segment[0],0)+np.maximum(segments[:,1]-segments[:,0],0)-overlap
    return overlap/np.maximum(union,1e-12)


def error_counts(predictions,ground_truth,threshold,epsilon=.01):
    """按类/分数贪心匹配，互斥预测错误类型；漏检另计，不与预测计数相加。"""
    classes={int(x['label']) for rows in ground_truth.values() for x in rows}
    predictions=[p for p in predictions if p['label'] in classes]
    used={v:set() for v in ground_truth};counts=Counter();per_gt=sum(map(len,ground_truth.values()))
    for pred in sorted(predictions,key=lambda x:-x['score']):
        video=pred['video_id'];gt=ground_truth.get(video,[])
        overlaps=iou(pred['segment'],[g['segment'] for g in gt]);same=np.array([g['label']==pred['label'] for g in gt],dtype=bool)
        candidates=[int(j) for j in np.argsort(-overlaps) if same[j] and overlaps[j]>=threshold]
        fresh=[j for j in candidates if j not in used.get(video,set())]
        if fresh:
            used[video].add(fresh[0]);category='true_positive'
        elif candidates:category='duplicate'
        elif any(overlaps[~same]>=threshold):category='classification'
        elif any(overlaps[same]>=epsilon):category='localization'
        elif any(overlaps>=epsilon):category='classification_and_localization'
        else:category='background'
        counts[category]+=1
    counts.update(predictions=len(predictions),gt=per_gt,missed_gt=per_gt-sum(map(len,used.values())))
    return dict(counts)


def occupancy(points,segments):
    points=np.asarray(points);segments=np.asarray(segments).reshape(-1,2)
    if not len(segments):return np.zeros(len(points),dtype=bool)
    return ((points[:,None]>=segments[:,0]) & (points[:,None]<=segments[:,1])).any(1)


def routing_counts(probabilities,valid,assigned,inside,threshold):
    valid=np.asarray(valid,dtype=bool);assigned=np.asarray(assigned,dtype=bool);inside=np.asarray(inside,dtype=bool)
    bg=np.asarray(probabilities).max(-1)<threshold
    n=int(valid.sum());b=int((valid&bg).sum());p=int((valid&assigned).sum())
    return dict(valid=n,routed_background=b,assigned_positive=p,
        background_but_assigned_positive=int((valid&bg&assigned).sum()),
        background_inside_annotated_action=int((valid&bg&inside).sum()),
        background_inside_action_but_unassigned=int((valid&bg&inside&~assigned).sum()),
        background_outside_annotated_task_actions=int((valid&bg&~inside).sum()))


def sum_counts(rows):
    result=Counter()
    for row in rows:result.update(row)
    return dict(result)
