"""与发布版TP/FP匹配等价：去掉逐预测pandas构造，不改变排序/阈值/AP积分。"""
import numpy as np


def interpolated_prec_rec(prec, rec):
    # 发布版原公式，保留求和次序。
    mprec=np.hstack([[0],prec,[0]]);mrec=np.hstack([[0],rec,[1]])
    for i in range(len(mprec)-1)[::-1]:mprec[i]=max(mprec[i],mprec[i+1])
    idx=np.where(mrec[1::]!=mrec[0:-1])[0]+1
    return np.sum((mrec[idx]-mrec[idx-1])*mprec[idx])


def compute_average_precision_detection(ground_truth,prediction,tiou_thresholds=np.linspace(.1,.5,5)):
    thresholds=np.asarray(tiou_thresholds);ap=np.zeros(len(thresholds))
    if prediction.empty:return ap
    order=prediction['score'].values.argsort()[::-1]
    prediction=prediction.loc[order].reset_index(drop=True)
    # 官方调用入口已重置每类GT索引，锁仍以该完整索引寻址。
    assert np.array_equal(ground_truth.index.to_numpy(),np.arange(len(ground_truth)))
    npos=float(len(ground_truth));lock=np.full((len(thresholds),len(ground_truth)),-1.)
    tp=np.zeros((len(thresholds),len(prediction)));fp=np.ones_like(tp)
    truth_groups=ground_truth.groupby('video-id',sort=False).indices
    truth_boxes=ground_truth[['t-start','t-end']].to_numpy()
    pred_boxes=prediction[['t-start','t-end']].to_numpy()
    minimum=float(thresholds.min())
    for video,positions in prediction.groupby('video-id',sort=False).indices.items():
        ids=truth_groups.get(video)
        if ids is None:continue
        boxes=truth_boxes[ids]
        for offset in range(0,len(positions),256):
            chunk=positions[offset:offset+256];pred=pred_boxes[chunk]
            if boxes.dtype==np.float64:
                left=np.maximum(pred[:,None,0],boxes[None,:,0])
                right=np.minimum(pred[:,None,1],boxes[None,:,1])
                intersection=(right-left).clip(0)
                union=(boxes[:,1]-boxes[:,0])[None,:]+(pred[:,1]-pred[:,0])[:,None]-intersection
                matrix=intersection.astype(float)/union
            else:
                # 保持numpy对标量与低精度GT数组的原有类型提升规则。
                rows=[]
                for start,end in pred:
                    left=np.maximum(float(start),boxes[:,0]);right=np.minimum(float(end),boxes[:,1])
                    intersection=(right-left).clip(0)
                    union=(boxes[:,1]-boxes[:,0])+(float(end)-float(start))-intersection
                    rows.append(intersection.astype(float)/union)
                matrix=np.asarray(rows)
            for rank,row in zip(chunk,matrix):
                candidates=row.argsort()[::-1]
                if np.isfinite(row).all():candidates=candidates[row[candidates]>=minimum]
                for ti,threshold in enumerate(thresholds):
                    for local in candidates:
                        if row[local]<threshold:break
                        gid=ids[local]
                        if lock[ti,gid]>=0:continue
                        tp[ti,rank]=1.;fp[ti,rank]=0.;lock[ti,gid]=rank;break
    tp_cumsum=np.cumsum(tp,axis=1).astype(float);fp_cumsum=np.cumsum(fp,axis=1).astype(float)
    recall=tp_cumsum/npos;precision=tp_cumsum/(tp_cumsum+fp_cumsum)
    for i in range(len(thresholds)):ap[i]=interpolated_prec_rec(precision[i],recall[i])
    tp_total=np.sum(tp,axis=1);fp_total=np.sum(fp,axis=1)
    fn_total=np.full((len(thresholds),),len(ground_truth))-tp_total
    return ap,tp_total,fp_total,fn_total


def install():
    from libs.evaluation import evaluator_anet
    evaluator_anet.compute_average_precision_detection=compute_average_precision_detection
