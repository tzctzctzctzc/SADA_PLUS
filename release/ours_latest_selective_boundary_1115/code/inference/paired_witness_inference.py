"""Ours 的单视频目标域读出。输入为原生候选（秒）与同一 EMA 的目标域 FPN。

bank = load_source_bank(primary_root, shift, checkpoint_sha256)
scores = video_scores(model, video, (fpn, masks, logits), boxes, raw, labels, bank)

每个源参照库与训练方向和检查点绑定；六个方向使用同一个评分函数。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from candidate_evidence_1109 import score_candidates
from ours_final import responses
from paired_witness_core_1109 import describe, witness_scores
from uniform_tricks import transform


def load_source_bank(primary_root, shift, checkpoint_sha256):
    """从已核验的复放目录装入该检查点的源 TRAIN 正/错位配对描述。"""
    root = Path(primary_root)
    row = json.loads((root / (shift + '.json')).read_text(encoding='utf-8'))
    summary = row['source_bank']
    binding = summary['binding']
    if (row['status'] != 'passed' or binding['shift'] != shift or
            binding['checkpoint'] != checkpoint_sha256 or row['checkpoint_sha256'] != checkpoint_sha256):
        raise ValueError('源参照与实验方向或检查点不匹配')
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    directory = root / 'banks' / shift / digest / 'train'
    groups = {}
    for path in directory.glob('*.npz'):
        with np.load(path, allow_pickle=False) as f:
            name = str(f['ids'][0])
            if name in groups or not (f['ids'] == name).all():
                raise ValueError('源参照视频记录重复或混合')
            groups[name] = {key: f[key] for key in ('pos', 'neg', 'labels', 'levels')}
    if not groups or len(groups) != summary['train_videos']:
        raise ValueError('源 TRAIN 参照缺少视频')
    bank = {key: np.concatenate([groups[name][key] for name in sorted(groups)])
            for key in ('pos', 'neg', 'labels', 'levels')}
    if len(bank['labels']) != summary['train_pairs']:
        raise ValueError('源 TRAIN 配对数量与核验记录不一致')
    return bank


def _prototype_cosines(model, features, segments, levels):
    """复用 A 的区间编码器，在去除共有方向后读出与各源类原型的余弦。"""
    branch = model.rq3_capacity.branch
    prototypes = F.normalize(branch.source_prototypes, dim=-1)
    ready = branch.source_prototype_ready
    ready_np = ready.cpu().numpy()
    if not ready.any():
        return np.zeros((len(segments), len(prototypes)), dtype=np.float32), ready_np
    embeddings = []
    for seg, lvl in zip(segments.split(128), levels.split(128)):
        proposal = {'segments': seg, 'levels': lvl}
        embeddings.append(branch.encode(features, [proposal], model.fpn_strides)['embedding'])
    query = torch.cat(embeddings)
    axis = F.normalize(prototypes[ready].mean(0), dim=0)
    query = F.normalize(query - (query @ axis)[:, None] * axis, dim=-1)
    prototypes = F.normalize(prototypes - (prototypes @ axis)[:, None] * axis, dim=-1)
    return (query @ prototypes.T).cpu().numpy(), ready_np


@torch.inference_mode()
def video_scores(model, video, maps, boxes, raw, labels, bank):
    """按原输入行序返回分数。maps 为一段视频的特征金字塔、掩码和密集分类 logits。"""
    if model.training:
        raise ValueError('推理入口要求 eval 模式')
    boxes, raw, labels = np.asarray(boxes), np.asarray(raw), np.asarray(labels, dtype=np.int64)
    if len(raw) != len(labels) or boxes.shape != (len(labels), 2):
        raise ValueError('候选区间、分数与标签的行数不一致')
    if not len(raw):
        return raw.copy()

    features, masks, logits = maps
    meta = video['metadata']
    describes, levels = describe(model, features, masks, logits, meta, boxes, labels)
    location_support, shuffled_support, _covered = witness_scores(describes, labels, levels, bank)
    route_cfg=getattr(model,'cfg',{}).get('uda_route',{}) if hasattr(getattr(model,'cfg',{}),'get') else {}
    if route_cfg.get('b_selective_boundary',False) and hasattr(getattr(model, 'adapt_route', None), 'inference_gate'):
        gate=model.adapt_route.inference_gate(model, video, describes, labels,
                                              location_support, shuffled_support)
        location_support=1.0+gate*(location_support-1.0)
    elif route_cfg.get('b_vlm_gate',False) and hasattr(getattr(model, 'rq3_capacity', None), '_moe_target_cache'):
        from vlm_gated_boundary import candidate_gate
        gate=candidate_gate(model.rq3_capacity._moe_target_cache, video['metadata']['video_id'], labels)
        location_support=1.0+gate*(location_support-1.0)

    tr = (meta['fps'], .5 * meta['feat_num_frames'], meta['feat_stride']) if meta['use_temporal_grid_feats'] else (1., 0., 1.)
    segments = torch.as_tensor(boxes, dtype=torch.float32, device=features[0].device)
    segments = (segments * tr[0] - tr[1]) / tr[2]
    cosines, ready = _prototype_cosines(model, features, segments, torch.as_tensor(levels, device=segments.device))
    dense = dict(transform=tr, ranges=model.rq3_capacity.regression_ranges, strides=model.fpn_strides)
    for level, (mask, logit) in enumerate(zip(masks, logits)):
        count = int(mask[0].sum())
        dense['q' + str(level)] = logit[0, :, :count].sigmoid().cpu().numpy()
    center = transform(dense, 'temperature2')
    all_response = responses(center, boxes, len(ready))
    # 旧评分接口读列 0 和列 3。列 3 在本版本来自真实时频参照；
    # 列 0 保留历史全局框内外响应的运算顺序，以便逐候选复核。
    from ours_final import profiles
    temporal = profiles(center, boxes, labels)
    temporal[:, 3] = location_support
    return score_candidates(boxes, raw, labels, cosines, ready, temporal, all_response)
