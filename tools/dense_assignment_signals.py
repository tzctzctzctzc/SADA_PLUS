"""源域AP旁路与实际三态/TSA监督质量；真值仅在CPU读取。"""
import hashlib
import numpy as np
import torch
from tools.baseline_protocol import read,save,sha


def frozen_crop(video, maximum):
    length=video['feats'].shape[-1];width=min(length,maximum)
    value=int(hashlib.sha256(('dense-assignment-source-v1/'+video['metadata']['video_id']).encode()).hexdigest()[:16],16)
    lo=value % (length-width+1)
    result=dict(video);result['metadata']=dict(video['metadata'],crop_start=lo)
    result['feats']=video['feats'][:,lo:lo+width]
    return result


def diagnostic_arrays(student,teacher,states,video,step,mode):
    from tools.dense_assignment import native_predictions,target_losses
    from tools.dense_assignment_math import feature_view,decoded
    cropped=frozen_crop(video,student.max_seq_len)
    strong,view=feature_view(cropped,step,0)
    with torch.no_grad():
        teacher.eval();student.train()
        weak=native_predictions(teacher,[cropped],states[1])
        strong_pred=native_predictions(student,[strong],states[0])
        _,stats,obs,items=target_losses(student,weak,strong_pred,[cropped],[view],mode,update=False)
        item=items[0];a=item['alignment'];points=torch.cat(strong_pred['points'])
        sb=decoded(points,torch.cat([r[0] for r in strong_pred['reg']]))
    arrays={k:item[k] for k in ['classes','active','positive','negative','ignored','soft','regression','mined','conflict','boxes']}
    arrays.update(points=points,teacher_probability=a['prob'],teacher_quality=a['quality'],
                  student_boxes=sb,aligned_valid=a['valid'])
    return {k:v.detach().cpu().numpy() for k,v in arrays.items()},dict(video=video['metadata']['video_id'],
        crop_start=cropped['metadata']['crop_start'],weak_length=cropped['feats'].shape[-1],
        view=view,stats=stats,observations=obs)


def assign_cpu(points,segments,labels,radius,use_range=True):
    classes=np.zeros((len(points),10),np.float32)
    if not len(segments):return classes
    left=points[:,0,None]-segments[None,:,0];right=segments[None,:,1]-points[:,0,None]
    center=segments.mean(-1)
    lo=np.maximum(center[None]-points[:,3,None]*radius,segments[None,:,0])
    hi=np.minimum(center[None]+points[:,3,None]*radius,segments[None,:,1])
    inside=np.minimum(points[:,0,None]-lo,hi-points[:,0,None])>0
    if use_range:
        dist=np.maximum(left,right);inside &= (dist>=points[:,1,None])&(dist<=points[:,2,None])
    lengths=np.broadcast_to(segments[:,1]-segments[:,0],inside.shape).copy();lengths[~inside]=np.inf
    best=lengths.min(-1)
    # 原检测器同长度实例允许多类标签，不强行变成单一类别。
    with np.errstate(invalid='ignore'):
        tied=(np.abs(lengths-best[:,None])<=1e-3)&np.isfinite(lengths)
    for c in range(10):classes[:,c]=tied[:,labels==c].any(-1)
    return classes


def evaluate_diagnostics(cfg,entry,epoch):
    d=entry/('signals/e%03d'%epoch);manifest=read(d/'diagnostics.json')
    annotation=cfg['baseline']['annotation_for_offline_evaluation']
    assert sha(annotation)==cfg['baseline']['annotation_sha256']
    db=read(annotation)['database'];rows=[]
    for row in manifest['videos']:
        assert row['video'] in db and db[row['video']]['subset'] in cfg['src_val_split']
        assert sha(row['arrays'])==row['arrays_sha256']
        with np.load(row['arrays'],allow_pickle=False) as source:a={k:source[k] for k in source.files}
        meta=row['metadata'];gt=[g for g in db[row['video']].get('annotations',[]) if 0<=int(g['label_id'])<10]
        segments=np.array([g['segment'] for g in gt],np.float32).reshape(-1,2)
        labels=np.array([int(g['label_id']) for g in gt],int)
        if meta['use_temporal_grid_feats']:
            segments=(segments*meta['fps']-.5*meta['feat_num_frames'])/meta['feat_stride']
        segments-=row['crop_start']
        view=row['view'];segments=(segments-view['lo']+.5)*view['ratio']-.5
        keep=(segments[:,1]>0)&(segments[:,0]<view['length'])
        segments=np.clip(segments[keep],0,view['length']);labels=labels[keep]
        classes=assign_cpu(a['points'],segments,labels,cfg['model']['center_sample_radius'],cfg['model'].get('reg_distance',True))
        p=a['teacher_probability'];pred_label=a['classes'].argmax(-1);raw_label=p.argmax(-1)
        if len(segments):
            inter=np.maximum(0,np.minimum(a['boxes'][:,None,1],segments[None,:,1])-np.maximum(a['boxes'][:,None,0],segments[None,:,0]))
            iou=inter/np.maximum(1e-8,(a['boxes'][:,1]-a['boxes'][:,0])[:,None]+(segments[:,1]-segments[:,0])[None]-inter)
            same=(raw_label[:,None]==labels[None]);correct_boxes=(iou>=.5)&same
            occupancy=((a['points'][:,0,None]>segments[None,:,0])&(a['points'][:,0,None]<segments[None,:,1])).any(-1)
        else:
            correct_boxes=np.zeros((len(p),0),bool);iou=np.zeros((len(p),0));occupancy=np.zeros(len(p),bool)
        pos=a['positive'];neg=a['negative'];valid=a['aligned_valid']
        correct=classes[np.arange(len(p)),pred_label]>0
        k=int(pos.sum());conf=p.max(-1);joint=conf*a['teacher_quality']
        candidates=np.flatnonzero(valid)
        def quality_of_order(score):
            ids=candidates[np.argsort(-score[candidates],kind='stable')[:k]]
            return float(correct_boxes[ids].any(-1).mean()) if len(ids) else None
        row_result=dict(video=row['video'],valid=int(valid.sum()),positive=k,negative=int(neg.sum()),
            soft=int(a['soft'].sum()),ignored=int(a['ignored'].sum()),regression=int(a['regression'].sum()),
            positive_detection_class_accuracy=float(correct[pos].mean()) if k else None,
            negative_detection_positive_contamination=float((classes.sum(-1)[neg]>0).mean()) if neg.any() else None,
            negative_physical_action_contamination=float(occupancy[neg].mean()) if neg.any() else None,
            confidence_equal_count_joint_precision=quality_of_order(conf),
            joint_equal_count_joint_precision=quality_of_order(joint),
            positive_instances_covered=int(correct_boxes[pos].any(0).sum()),source_instances=len(segments))
        rows.append(row_result)
    keys=['positive_detection_class_accuracy','negative_detection_positive_contamination',
          'negative_physical_action_contamination','confidence_equal_count_joint_precision','joint_equal_count_joint_precision']
    macro={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
    result=dict(epoch=epoch,macro_by_video=macro,videos=rows,
        scope='one_frozen_source_val_crop_per_video; source_selected_set_not_independent_target; dense_counts_not_AP',
        target_GT_used=False,diagnostics_manifest_sha256=sha(d/'diagnostics.json'))
    save(d/'mechanism_quality.json',result)
    return result
