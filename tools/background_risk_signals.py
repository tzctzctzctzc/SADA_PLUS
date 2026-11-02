"""背景连续权重的源域误伤、校准与等权重分布对照。"""
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.supervision_risk_signals import stable_crop,truth_arrays
from tools.supervision_risk import teacher_predictions
from tools.background_risk import SPEC,calibrated


def weighted_metrics(probability,raw,valid,background,occupancy,classes,calibration):
    score=probability.max(-1);candidate=valid&(score<SPEC['candidate_max_class_score'])
    ids=np.flatnonzero(candidate)
    if not len(ids):return dict(candidates=0,weight_mass=0.,effective_fraction=0.)
    with torch.no_grad():q=calibrated(torch.as_tensor(raw[ids]),calibration).numpy()
    w=q**2;mass=float(w.sum());total=len(w)
    # 完全相同的权重集合，只按原低置信度重新排列；隔离整体loss缩小效应。
    order=np.argsort(score[ids],kind='stable');matched=np.empty_like(w)
    matched[order]=np.sort(w)[::-1]
    physical=occupancy[ids].astype(float);detection=classes[ids].any(-1).astype(float)
    def mean(v,weight):return float((v*weight).sum()/max(float(weight.sum()),1e-12))
    uniform=float(physical.mean());learned=mean(physical,w);confidence=mean(physical,matched)
    rawq=1/(1+np.exp(-np.clip(raw[ids],-80,80)))
    bg=background[ids].astype(float)
    return dict(candidates=total,weight_mass=mass/total,effective_fraction=mass**2/(max(float((w*w).sum()),1e-12)*total),
        weighted_physical_contamination=learned,uniform_physical_contamination=uniform,
        same_weight_confidence_physical_contamination=confidence,
        gain_vs_uniform=uniform-learned,gain_vs_same_weight_confidence=confidence-learned,
        weighted_detection_positive_contamination=mean(detection,w),
        raw_brier=float(((rawq-bg)**2).mean()),calibrated_brier=float(((q-bg)**2).mean()),
        calibrated_mean_background=float(q.mean()),actual_background=float(bg.mean()))


def aggregate(rows):
    usable=[r for r in rows if r['candidates']]
    if not usable:return dict(videos=len(rows),candidate_videos=0)
    keys=[k for k in usable[0] if k not in ('video','candidates')]
    return dict({k:float(np.mean([r[k] for r in usable])) for k in keys},videos=len(rows),candidate_videos=len(usable),
        candidates=sum(r['candidates'] for r in rows))


def group_readiness(reports):
    if len(reports)!=6:return False
    m=[r['validation'] for r in reports]
    if any(not r.get('candidate_videos') or r['weight_mass']<.1 or r['effective_fraction']<.25 for r in m):return False
    return np.mean([r['gain_vs_uniform'] for r in m])>0 and np.mean([r['gain_vs_same_weight_confidence'] for r in m])>0


def diagnostic_arrays(model,video):
    cropped=stable_crop(video,model.max_seq_len)
    with torch.no_grad():p=teacher_predictions(model,[cropped])
    a=dict(probability=p['cls'][0].sigmoid(),raw=p['risk'][0,:,0],valid=p['valid'][0],points=torch.cat(p['points']))
    return {k:v.cpu().numpy() for k,v in a.items()},dict(video=video['metadata']['video_id'],metadata=cropped['metadata'],
        calibration=model.risk_state['calibration'],GT_loaded=False,view='fixed_source_crop')


def evaluate_diagnostics(cfg,entry,epoch):
    d=entry/('signals/e%03d'%epoch);manifest=read(d/'diagnostics.json')
    path=cfg['baseline']['annotation_for_offline_evaluation'];assert sha(path)==cfg['baseline']['annotation_sha256']
    db=read(path)['database'];rows=[]
    for row in manifest['videos']:
        assert db[row['video']]['subset'] in cfg['src_val_split'] and sha(row['arrays'])==row['arrays_sha256']
        meta=row['metadata'];gt=[g for g in db[row['video']].get('annotations',[]) if 0<=int(g['label_id'])<10]
        segments=np.asarray([g['segment'] for g in gt],np.float32).reshape(-1,2)
        segments=(segments*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']-meta['crop_start']
        length=meta['crop_feat_len'];keep=(segments[:,1]>0)&(segments[:,0]<length)
        segments=segments[keep].clip(0,length);labels=np.asarray([int(g['label_id']) for g in gt])[keep]
        with np.load(row['arrays'],allow_pickle=False) as a:
            classes,bg,occ=truth_arrays(cfg,a['points'],segments,labels,a['valid'])
            r=weighted_metrics(a['probability'],a['raw'],a['valid'],bg,occ,classes,row['calibration'])
        rows.append(dict(r,video=row['video']))
    result=dict(epoch=epoch,macro_by_video=aggregate(rows),videos=rows,target_GT_used=False,
        scope='source_val_only_fixed_crop_not_target_AP',diagnostics_sha256=sha(d/'diagnostics.json'))
    save(d/'mechanism_quality.json',result);return result
