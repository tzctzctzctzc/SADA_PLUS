"""VLM语义可靠性门控，复用已缓存的Router输出，不调用VLM。"""
import torch


def _rows(cache, video_id):
    return list(cache.get(str(video_id), ())) if cache is not None else []


def class_reliability(cache, video_id, class_id, threshold=.8):
    rows=_rows(cache, video_id)
    values=[]
    for row in rows:
        q=row.get('class_probabilities', ())
        if len(q)!=10: continue
        cls=max(range(len(q)), key=lambda i: q[i])
        if int(cls)==int(class_id) and bool(row.get('participation_selected',False)):
            values.append(float(row.get('routing_score',0.)))
    # 没有该类别的VLM证据时保持中性，不把缺失证据误当成低可靠性。
    if not values: return 1.
    return float(max(0.,min(1.,sum(values)/len(values)))) if max(values)>=threshold else .25


def batch_reliability(cache, videos, class_id, device, threshold=.8):
    values=[class_reliability(cache,v['metadata']['video_id'],class_id,threshold) for v in videos]
    return torch.as_tensor(values,device=device,dtype=torch.float32)


def gate_alignment(loss, cache, videos, class_ids, device, threshold=.8):
    """按视频/类别平均门控目标无标签对齐，源监督不被削弱。"""
    if loss.ndim==0 and not videos: return loss,loss.new_ones(())
    gates=[]
    for video,c in zip(videos,class_ids):
        gates.append(class_reliability(cache,video['metadata']['video_id'],int(c),threshold))
    gate=torch.as_tensor(gates,device=device,dtype=loss.dtype).mean() if gates else loss.new_ones(())
    return loss*gate.clamp_min(.25),gate
