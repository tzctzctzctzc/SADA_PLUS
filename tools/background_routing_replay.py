"""四路离线重放，不声称重训mAP：source拟定模糊分数带，target GT仅用于事后统计。"""
import hashlib
from pathlib import Path
import numpy as np
from tools import baseline_protocol as p
from tools.background_diagnostics import sum_counts


def masks(confidence,category,valid,threshold,lower,seed):
    original=np.asarray(valid,dtype=bool)&(confidence<threshold)
    ambiguous=original&(confidence>=lower) if lower is not None else np.zeros_like(original)
    removed=np.zeros_like(original);rng=np.random.RandomState(seed)
    for cls in np.unique(category[original]):
        ids=np.flatnonzero(original&(category==cls));count=int((ambiguous&(category==cls)).sum())
        removed[rng.permutation(ids)[:count]]=True
    return dict(original=original,protect_ambiguous=original&~ambiguous,
        matched_random=original&~removed,disable_background=np.zeros_like(original))


def weighted_lower(groups,quantile=.05):
    if len(groups)<2:return None
    values=np.concatenate(groups);weights=np.concatenate([np.full(len(x),1./len(x)/len(groups)) for x in groups])
    order=np.argsort(values);index=np.searchsorted(np.cumsum(weights[order]),quantile)
    return float(values[order[min(index,len(values)-1)]])


def replay(root,plan):
    root=Path(root);index=p.read(root/'point_index.json');roles=['student','ema']
    source=sorted({x['video_id'] for x in index if x['domain']=='source'},key=lambda v:hashlib.sha256(('123:'+v).encode()).hexdigest())
    fit=source[::2];audit=source[1::2];thresholds={};counts={}
    levels=sorted({x['level'] for x in p.read(root/'routing.json') if x['active_alignment_level']})
    for role in roles:
        thresholds[role]={}
        for level in levels:
            groups=[]
            for entry in index:
                if entry['role']!=role or entry['domain']!='source' or entry['video_id'] not in fit:continue
                with np.load(entry['path'],allow_pickle=False) as data:
                    select=data['l%d_valid'%level]&data['l%d_assigned'%level]
                    if select.any():groups.append(data['l%d_confidence'%level][select])
            thresholds[role][str(level)]=weighted_lower(groups)
    for entry in index:
        if p.sha(entry['path'])!=entry['sha256']:raise ValueError('点记录改变')
        with np.load(entry['path'],allow_pickle=False) as data:
            for level in levels:
                prefix='l%d_'%level;confidence=data[prefix+'confidence'];assigned=data[prefix+'assigned'];inside=data[prefix+'inside']
                seed=int(hashlib.sha256((entry['video_id']+entry['role']+str(level)).encode()).hexdigest()[:8],16)
                branches=masks(confidence,data[prefix+'category'],data[prefix+'valid'],plan['classification_threshold'],thresholds[entry['role']][str(level)],seed)
                # 两种保护策略只改target背景分组，关闭背景对齐的对照则同时关闭source/target。
                if entry['domain']=='source':
                    branches['protect_ambiguous']=branches['original'];branches['matched_random']=branches['original']
                if int(branches['protect_ambiguous'].sum())!=int(branches['matched_random'].sum()):raise ValueError('随机对照覆盖率不配对')
                for arm,mask in branches.items():
                    key=(entry['domain'],entry['role'],arm)
                    row=dict(kept_background=int(mask.sum()),kept_assigned_positive=int((mask&assigned).sum()),
                        kept_inside_annotated_action=int((mask&inside).sum()),kept_outside_task_actions=int((mask&~inside).sum()))
                    counts.setdefault(key,[]).append(row)
    report=dict(scope='offline routing replay; no retrained mAP',source_calibration_videos=fit,source_audit_videos=audit,
        lower_score_thresholds=thresholds,upper_score_threshold=plan['classification_threshold'],quantile=.05,
        score_proxy='source positive-score lower quantile; exploratory ambiguity band, not a calibrated foreground/uncertainty head',
        target_gt_used_for_threshold=False,random_matching='same video/FPN level/predicted class, exact removed count',
        counts={'/'.join(key):sum_counts(rows) for key,rows in counts.items()},training_performed=False,
        conclusion_limit='routing contamination changes do not establish mAP improvement')
    p.save(root/'routing_replay.json',report);return report
