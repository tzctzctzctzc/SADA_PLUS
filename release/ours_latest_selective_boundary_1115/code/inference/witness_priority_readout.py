"""推理时让现有源时频参照参与原生 Soft-NMS；一次处理一个视频。"""
import numpy as np
import torch
from paired_witness_core_1109 import describe, witness_scores
from paired_witness_inference import video_scores


@torch.inference_mode()
def pre_nms_evidence(model, video, maps, decoded, bank):
    """decoded 为 inference_single_video 解码后的网格坐标。返回真实证据及审计用对内互换证据。"""
    if model.training or decoded['video_id'] != video['metadata']['video_id']:
        raise ValueError('推理状态或视频不匹配')
    g = decoded['segments'].detach().cpu()
    if decoded['use_temporal_grid_feats']:
        seconds = (g * decoded['feat_stride'] + .5 * decoded['feat_num_frames']) / decoded['fps']
        limit = decoded['duration']
    else:
        seconds, limit = g, decoded['video_length']
    b = seconds.clamp(0,limit).numpy()
    c = decoded['labels'].detach().cpu().numpy()
    keys, inv = np.unique(np.column_stack([b,c]),axis=0,return_inverse=True)
    desc, levels = describe(model,*maps,video['metadata'],keys[:,:2],keys[:,2].astype(int))
    real, swap, _ = witness_scores(desc,keys[:,2].astype(int),levels,bank)
    route_cfg=getattr(model,'cfg',{}).get('uda_route',{}) if hasattr(getattr(model,'cfg',{}),'get') else {}
    if route_cfg.get('b_selective_boundary',False) and hasattr(getattr(model,'adapt_route',None),'inference_gate'):
        gate=model.adapt_route.inference_gate(model,video,desc,keys[:,2].astype(int),real,swap)
        real=1.0+gate*(real-1.0)
    return real[inv], swap[inv]


@torch.inference_mode()
def readout(model, video, maps, decoded, bank, *, priority=None):
    """返回秒坐标、原类别与最终分数。默认用真实 B 证据，priority 允许复用本视频的计算。"""
    if priority is None:
        priority, _ = pre_nms_evidence(model,video,maps,decoded,bank)
    sc = decoded['scores']
    if np.shape(priority) != tuple(sc.shape) or not np.isfinite(priority).all():
        raise ValueError('去重优先级与候选不匹配')
    # 原生后处理可能就地裁剪网格坐标，保留调用者输入供配对复核。
    row = dict(decoded,segments=decoded['segments'].clone(),
               scores=sc*torch.as_tensor(priority,dtype=sc.dtype,device=sc.device))
    pred = model.postprocessing([row])[0]
    b = pred['segments'].detach().cpu().numpy().astype(float)
    c = pred['labels'].detach().cpu().numpy()
    raw = pred['scores'].detach().cpu().numpy().astype(float)
    values = video_scores(model,video,maps,b,raw,c,bank)
    return b,c,values
