"""源域固定crop风险验证；检测负点与物理背景分别计算，视频等权。"""
import hashlib
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha
from tools.dense_assignment_signals import assign_cpu
from tools.supervision_risk import SPEC,teacher_predictions,select


def stable_crop(video,maximum):
    n=video['feats'].shape[-1];length=min(n,maximum)
    lo=int(hashlib.sha256(('supervision-risk-v1/'+video['metadata']['video_id']).encode()).hexdigest()[:16],16)%(n-length+1)
    result=dict(video);result['metadata']=dict(video['metadata'],crop_start=lo,crop_feat_len=length)
    result['feats']=video['feats'][:,lo:lo+length]
    if video.get('segments') is not None:
        s=video['segments']-lo;keep=(s[:,1]>0)&(s[:,0]<length)
        result['segments']=s[keep].clamp(0,float(length));result['labels']=video['labels'][keep]
    return result


def truth_arrays(cfg,points,segments,labels,valid):
    classes=assign_cpu(points,segments,labels,cfg['model']['center_sample_radius'],cfg['model'].get('reg_distance',True))
    occupancy=((points[:,0,None]>=segments[None,:,0])&(points[:,0,None]<=segments[None,:,1])).any(-1) if len(segments) else np.zeros(len(points),bool)
    return classes,valid&~occupancy&~classes.any(-1),occupancy


def ranking_metrics(p,r,valid,classes,safe_background,occupancy):
    score=p.max(-1);label=p.argmax(-1);candidate=valid&(score>=SPEC['candidate_floor'])
    correct=classes[np.arange(len(p)),label]>0
    result=dict(valid=int(valid.sum()),candidates=int(candidate.sum()))
    def precision(mask,order,k,truth):
        ids=np.flatnonzero(mask);ids=ids[np.argsort(-order[ids],kind='stable')[:k]]
        return float(truth[ids].mean()) if len(ids) else None
    class_rows=[];bg_rows=[]
    for f in [.05,.1,.2,.4]:
        k=max(1,int(candidate.sum()*f))
        if candidate.any():class_rows.append(dict(fraction=f,count=min(k,int(candidate.sum())),
            learned=precision(candidate,r[:,0],k,correct),confidence=precision(candidate,score,k,correct)))
    bgmask=valid&(score<SPEC['candidate_floor'])
    for f in [.1,.25,.5]:
        k=max(1,int(bgmask.sum()*f))
        if bgmask.any():bg_rows.append(dict(fraction=f,count=min(k,int(bgmask.sum())),
            learned=precision(bgmask,r[:,1],k,safe_background),confidence=precision(bgmask,1-score,k,safe_background)))
    with torch.no_grad():
        item=select(torch.from_numpy(p),torch.from_numpy(r),torch.from_numpy(valid))
    pos=item['positive'].numpy();neg=item['negative'].numpy()
    result.update(class_equal_count=class_rows,background_equal_count=bg_rows,
        positive=int(pos.sum()),negative=int(neg.sum()),ignored=int(item['ignored'].sum()),
        positive_accuracy=float(correct[pos].mean()) if pos.any() else None,
        negative_safe_precision=float(safe_background[neg].mean()) if neg.any() else None,
        negative_detection_positive_contamination=float(classes[neg].any(-1).mean()) if neg.any() else None,
        negative_physical_action_contamination=float(occupancy[neg].mean()) if neg.any() else None,
        class_brier=float(((r[candidate,0]-correct[candidate])**2).mean()) if candidate.any() else None,
        background_brier=float(((r[valid,1]-safe_background[valid])**2).mean()),
        class_ranking_gain=float(np.mean([x['learned']-x['confidence'] for x in class_rows])) if class_rows else None,
        background_ranking_gain=float(np.mean([x['learned']-x['confidence'] for x in bg_rows])) if bg_rows else None)
    return result


def aggregate(rows):
    keys=['class_ranking_gain','background_ranking_gain','positive_accuracy','negative_safe_precision',
          'negative_detection_positive_contamination','negative_physical_action_contamination','class_brier','background_brier']
    macro={k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) if any(r.get(k) is not None for r in rows) else None for k in keys}
    macro.update(positive=sum(r['positive'] for r in rows),negative=sum(r['negative'] for r in rows),
        candidates=sum(r['candidates'] for r in rows),valid=sum(r['valid'] for r in rows),videos=len(rows))
    return macro


def readiness(m):
    a=m['class_ranking_gain'];b=m['background_ranking_gain']
    return (a is not None and b is not None and a>=0 and b>=0 and (a>0 or b>0)
        and m['positive']>=max(1,.01*m['candidates']) and m['negative']>=max(1,.01*m['valid'])
        and m['positive_accuracy'] is not None and m['positive_accuracy']>=.8
        and m['negative_safe_precision'] is not None and m['negative_safe_precision']>=.95)


def diagnostic_arrays(model,video):
    cropped=stable_crop(video,model.max_seq_len)
    with torch.no_grad():pred=teacher_predictions(model,[cropped])
    arrays=dict(probability=pred['cls'][0].sigmoid(),reliability=pred['risk'][0].sigmoid(),
                valid=pred['valid'][0],points=torch.cat(pred['points']))
    return {k:v.detach().cpu().numpy() for k,v in arrays.items()},dict(video=video['metadata']['video_id'],
        metadata=cropped['metadata'],GT_loaded=False,view='fixed_crop_same_across_all_epochs')


def evaluate_diagnostics(cfg,entry,epoch):
    folder=entry/('signals/e%03d'%epoch);manifest=read(folder/'diagnostics.json')
    annotation=cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation)==cfg['baseline']['annotation_sha256']
    db=read(annotation)['database'];rows=[]
    for row in manifest['videos']:
        video=row['video'];assert db[video]['subset'] in cfg['src_val_split']
        assert sha(row['arrays'])==row['arrays_sha256']
        with np.load(row['arrays'],allow_pickle=False) as a:
            meta=row['metadata'];gt=[g for g in db[video].get('annotations',[]) if 0<=int(g['label_id'])<10]
            segments=np.asarray([g['segment'] for g in gt],np.float32).reshape(-1,2)
            segments=(segments*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']-meta['crop_start']
            length=meta['crop_feat_len'];keep=(segments[:,1]>0)&(segments[:,0]<length)
            segments=segments[keep].clip(0,length);labels=np.asarray([int(g['label_id']) for g in gt])[keep]
            classes,bg,occupancy=truth_arrays(cfg,a['points'],segments,labels,a['valid'])
            r=ranking_metrics(a['probability'],a['reliability'],a['valid'],classes,bg,occupancy)
            rows.append(dict(r,video=video))
    result=dict(epoch=epoch,videos=rows,macro_by_video=aggregate(rows),target_GT_used=False,
                scope='source_val_fixed_crop_video_macro_not_target_AP',manifest_sha256=sha(folder/'diagnostics.json'))
    save(folder/'mechanism_quality.json',result);return result
