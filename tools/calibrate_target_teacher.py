"""经用户授权，仅用targetTRAIN标签校准教师门限；不读targetVAL。

六个预先固定候选；按视频宏F0.5选择可靠伪动作输入，最终收益仍由完整验证AP决定。
这不是严格UDA，也不把伪标签质量指标叫作AP提升。
"""
from pathlib import Path
import numpy as np
from tools.baseline_protocol import read,save,sha

THRESHOLDS=(.3,.4,.5,.6,.7,.8)


def counts(confidence,correct,threshold):
    selected=confidence>=threshold
    tp=int((selected&correct).sum());fp=int((selected&~correct).sum());fn=int((~selected&correct).sum())
    return dict(tp=tp,fp=fp,fn=fn,selected=int(selected.sum()),f_half=1.25*tp/max(1e-12,1.25*tp+.25*fn+fp))


def choose(videos):
    trials=[]
    for threshold in THRESHOLDS:
        stats=[counts(c,g,threshold) for c,g in videos]
        trials.append(dict(threshold=threshold,macro_f_half=float(np.mean([s['f_half'] for s in stats])),
            selected=sum(s['selected'] for s in stats),tp=sum(s['tp'] for s in stats),fp=sum(s['fp'] for s in stats),per_video=stats))
    usable=[r for r in trials if r['selected']>0]
    assert usable,'no selected target anchors across the preregistered candidates'
    assert max(r['tp'] for r in usable)>0,'teacher has no class-correct localized target actions'
    return max(usable,key=lambda r:(r['macro_f_half'],r['threshold'])),trials


def calibrate(origin,root):
    import torch
    from tools.run_foreground_calibration_oracle import load_truth
    from tools.run_native_joint import SOURCE
    reports={}
    for s in ('S1','S2'):
        cfg=read(SOURCE/s/'config.json');truthfile=SOURCE/s/'target_train_truth_cpu.json'
        datafile=origin/'inputs'/(s+'_unlabeled.pt');data=torch.load(datafile,map_location='cpu')
        rows=data['target_train'];ids={r['video'] for r in rows}
        assert set(read(truthfile)['database'])==ids
        assert not ids&{r['video'] for r in data['evaluation']['target_val']}
        truth=load_truth(truthfile,cfg['tgt_train_split'])
        path=origin/(s+'_teacher')/'semantic_teacher_logits.pt'
        gpu=read(path.parent/'gpu_complete.json');assert sha(path)==gpu['semantic_logits_sha256']
        cache=torch.load(path,map_location='cpu');assert set(cache)==ids
        videos=[];identity=[]
        for row in rows:
            item=cache[row['video']];meta=row['video_metadata'];assert meta['use_temporal_grid_feats']
            frame=truth[truth['video-id']==row['video']]
            boxes=frame[['t-start','t-end']].to_numpy();labels=frame['label'].to_numpy()
            conf=[];correct=[]
            for level,(c,r,mask) in enumerate(zip(item['cls'],item['reg'],item['masks'])):
                prob=c.sigmoid().numpy();confidence=prob.max(-1);predicted=prob.argmax(-1)
                stride=cfg['model']['scale_factor']**level;center=np.arange(len(c))*stride;offset=r.numpy()*stride
                segments=np.stack((center-offset[:,0],center+offset[:,1]),1)
                segments=(segments*meta['feat_stride']+.5*meta['feat_num_frames'])/meta['fps']
                segments=np.clip(segments,0,meta['duration']);best=np.zeros(len(c))
                # 一维原网格解码；只核teacher预测类别与同类GT，不把任意类覆盖算正确伪类别。
                for label in range(10):
                    keep=np.flatnonzero(predicted==label);gt=boxes[labels==label]
                    if not len(keep) or not len(gt):continue
                    a=segments[keep];inter=np.maximum(0,np.minimum(a[:,None,1],gt[None,:,1])-np.maximum(a[:,None,0],gt[None,:,0]))
                    union=(a[:,1]-a[:,0])[:,None]+(gt[:,1]-gt[:,0])[None]-inter
                    best[keep]=(inter/np.maximum(union,1e-8)).max(1)
                valid=mask.numpy().astype(bool);conf.append(confidence[valid]);correct.append(best[valid]>=.5)
            c=np.concatenate(conf);g=np.concatenate(correct);videos.append((c,g));identity.append(dict(video=row['video'],anchors=len(c),teacher_correct_at_iou05=int(g.sum()),max_confidence=float(c.max())))
        selected,trials=choose(videos)
        reports[s]=dict(selected_threshold=selected['threshold'],trials=trials,videos=identity,
            target_train_truth_sha256=sha(truthfile),teacher_cache_sha256=sha(path),unlabeled_input_sha256=sha(datafile),
            target_val_read=False,criterion='video-macro F0.5 of class-correct teacher boxes at IoU .5; NOT AP',
            original_sada_threshold=cfg['model']['pl_threshold'])
    save(root/'target_calibration.json',dict(passed=True,shifts=reports,target_train_labels_used=True,
        strict_UDA=False,target_val_read=False,threshold_candidates=list(THRESHOLDS),selection_frozen_before_training=True))
