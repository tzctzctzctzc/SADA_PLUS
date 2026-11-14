"""推理侧VLM语义门控；只读取训练阶段缓存，不调用VLM。"""
import numpy as np


def candidate_gate(cache, video_id, labels, threshold=.8):
    rows=list(cache.get(str(video_id),())) if cache is not None else []
    table={}
    for row in rows:
        if not row.get('participation_selected',False): continue
        q=row.get('class_probabilities',())
        if len(q)!=10: continue
        cls=int(np.argmax(np.asarray(q)))
        table.setdefault(cls,[]).append(float(row.get('calibrated_routing_score',row.get('routing_score',0.))))
    out=np.ones(len(labels),dtype=np.float64)
    for i,c in enumerate(np.asarray(labels,dtype=np.int64)):
        values=table.get(int(c),())
        if values and max(values)>=threshold:
            out[i]=float(np.clip(np.mean(values),.25,1.))
        elif values:
            out[i]=.25
    return out
