"""固定128维Prototype的目标实例缓存；仅重排已有检测，不更新模型或写入验证视频。"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch.nn import functional as F

SCHEMA = 'prototype-target-instance-cache-v1'
SHIFTS = tuple('S%d' % i for i in range(1, 7))


def specification():
    # 首版单一冻结配方；不根据目标GT搜参。每类每个视频最多贡献一个邻居。
    return dict(schema=SCHEMA, embedding_dim=128, alpha=.5, beta=5., neighbours=3,
                query_chunk=256, selected_candidates=40, candidate_pool=200,
                bank='original-selected-target-train-proposals-only',
                query='fixed-post-NMS-detections', query_level='original-source-duration-rule',
                affinity='mean-topk-distinct-video-confidence-weighted-exp-cosine',
                rescore='positive-same-class-log-odds-residual',
                absent_class='exact-original-score', trainable_parameters=0,
                validation_cache_writes=False, boundaries_and_labels='unchanged',
                checkpoint='original-source-selected-best-EMA', target_parameter_search=False)


def metadata_only(database, splits):
    wanted = {s.lower() for s in splits}
    # 保持原JSON的视频顺序；模型输入文件不携带annotations。
    return {'database': {k: {a: deepcopy(b) for a, b in v.items() if a != 'annotations'}
                         for k, v in database.items() if v['subset'].lower() in wanted}}


def selected_records(payload, target_ids, labels):
    if (payload.get('schema') != 'prototype-capacity-evidence-v1' or
            payload.get('teacher') != 'qwen3_vl_8b' or payload.get('target_gt_used') is not False or
            payload.get('candidate_count') != 200 or payload.get('selected_count') != 40):
        raise ValueError('不是已冻结的Qwen8B原200/40非oracle缓存')
    rows = [r for vid, items in payload['by_video'].items() for r in items]
    if len(rows) != 200 or len({r['candidate_id'] for r in rows}) != 200:
        raise ValueError('候选身份不唯一或总数改变')
    selected = [r for r in rows if r['participation_selected']]
    if len(selected) != 40:
        raise ValueError('原40名单改变')
    accepted, rejected = {}, []
    for r in selected:
        vid = r['video_id']
        if vid not in target_ids or r not in payload['by_video'].get(vid, []):
            raise ValueError('缓存候选不属于目标训练集')
        q = r.get('class_probabilities')
        name = r['semantic_label']
        if name not in labels or q is None:
            rejected.append(dict(candidate_id=r['candidate_id'], reason='unknown-or-abstain'))
            continue
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(labels),) or not np.isfinite(q).all() or (q < 0).any() or abs(q.sum()-1) > 1e-5:
            raise ValueError('缓存类别概率非法')
        if (q == q.max()).sum() != 1:
            rejected.append(dict(candidate_id=r['candidate_id'], reason='class-tie'))
            continue
        label = int(q.argmax())
        if labels[label] != name or not r['source_fpn_levels']:
            raise ValueError('缓存类别或原FPN来源不一致')
        accepted.setdefault(vid, []).append(dict(r, memory_label=label, memory_confidence=float(q[label])))
    return accepted, rejected


def duration_levels(segments, ranges):
    radius = (segments[:, 1]-segments[:, 0]).clamp_min(1.)/2.
    result = torch.full_like(radius, len(ranges)-1, dtype=torch.long)
    assigned = torch.zeros_like(radius, dtype=torch.bool)
    for level, (low, high) in enumerate(ranges):
        take = (~assigned) & (radius >= low) & (radius <= high)
        result[take] = level
        assigned |= take
    return result


@torch.no_grad()
def encode_segments(branch, features, segments, levels, strides, chunk=256):
    """向量化原7位置线性采样；只用于缓存读写，原检测前向完全不改。"""
    if len(segments) != len(levels):
        raise ValueError('候选/层数不匹配')
    result = features[0].new_empty((len(segments), branch.classifier.in_features))
    sampled = features[0].new_empty((len(segments), len(branch.sample_ratios), branch.input_projection.in_features))
    ratios = segments.new_tensor(branch.sample_ratios)
    for level, sequence in enumerate(features):
        indices = (levels == level).nonzero(as_tuple=True)[0]
        sequence = sequence[0]
        for ids in indices.split(chunk):
            if not len(ids):
                continue
            box = segments[ids]
            pos = (box[:, :1] + ratios[None]*(box[:, 1:]-box[:, :1]).clamp_min(1.))/float(strides[level])
            pos = pos.clamp(0, sequence.shape[-1]-1)
            left = pos.floor().long()
            right = (left+1).clamp(max=sequence.shape[-1]-1)
            w = (pos-left).unsqueeze(-1)
            a = sequence[:, left.reshape(-1)].T.reshape(len(ids), len(ratios), -1)
            b = sequence[:, right.reshape(-1)].T.reshape_as(a)
            sampled[ids] = a*(1-w)+b*w
    if len(levels) and ((levels < 0).any() or (levels >= len(features)).any()):
        raise ValueError('FPN层非法')
    # 恢复原候选顺序后统一经过编码器；不可按FPN层拆成不同的GEMM batch，
    # 否则真实GPU上小batch的累计舍入可能越过已冻结的等价性阈值。
    for start in range(0, len(segments), chunk):
        tokens = branch.input_projection(sampled[start:start+chunk]) + branch.position_embedding[None]
        embedding = branch.output_norm(branch.encoder(tokens).mean(1))
        result[start:start+chunk] = F.normalize(embedding, dim=-1)
    return result


@torch.no_grad()
def memory_support(query, labels, bank, spec=None):
    spec = specification() if spec is None else spec
    support = query.new_zeros(len(query))
    if not len(query) or not len(bank['embedding']):
        return support
    if query.shape[1] != bank['embedding'].shape[1]:
        raise ValueError('检索表示维度不一致')
    for label in labels.unique().tolist():
        keys = (bank['label'] == label).nonzero(as_tuple=True)[0]
        if not len(keys):
            continue
        ids = (labels == label).nonzero(as_tuple=True)[0]
        # 同一视频即使缓存了多个重叠候选，也只占一个邻居名额。
        groups = [keys[bank['video'][keys] == v] for v in bank['video'][keys].unique()]
        for chunk_ids in ids.split(spec['query_chunk']):
            affinities = []
            for group in groups:
                similarity = (query[chunk_ids] @ bank['embedding'][group].T).clamp(-1, 1)
                value = torch.exp(-spec['beta']*(1-similarity))*bank['confidence'][group][None]
                affinities.append(value.max(dim=1).values)
            values = torch.stack(affinities, dim=1)
            support[chunk_ids] = values.topk(min(spec['neighbours'], len(groups)), dim=1).values.mean(1)
    return support


def rescore(scores, support, alpha=.5):
    if scores.shape != support.shape or not torch.isfinite(scores).all() or not torch.isfinite(support).all():
        raise ValueError('分数/检索证据非法')
    if (scores < 0).any() or (scores > 1).any() or (support < 0).any() or (support > 1+1e-6).any():
        raise ValueError('概率或检索证据越界')
    if alpha == 0:
        return scores.clone()
    odds_factor = torch.exp(float(alpha)*support)
    value = scores*odds_factor/(1-scores+scores*odds_factor)
    # 缺失类别或关闭缓存时逐值保留原分数，不引入epsilon扰动。
    return torch.where(support > 0, value, scores)


@torch.no_grad()
def detector_features(model, video):
    if (model.backbone_source is not model.backbone_target or
            model.neck_source is not model.neck_target or
            model.cls_head is not model.cls_head_target or model.reg_head is not model.reg_head_target):
        raise ValueError('缓存首版只支持当前共享检测器')
    inputs, masks = model.preprocessing([video], mode='test')
    inputs = model.dropout(inputs)
    features, masks, _ = model.backbone_source(inputs, masks)
    features, masks = model.neck_source(features, masks)
    return features, masks


@torch.no_grad()
def detector_forward(model, video):
    """仅执行实际使用的共享检测路径，省去另一路相同检测及旧PL网络。"""
    features, masks = detector_features(model, video)
    logits, _ = model.cls_head(features, masks)
    offsets, _ = model.reg_head(features, masks)
    if model.use_trident_head:
        raise ValueError('首版必须匹配已有普通左右边界头')
    points = model.point_generator(features)
    result = model.inference([video], points, [m.squeeze(1) for m in masks],
                             [x.permute(0, 2, 1) for x in logits],
                             [x.permute(0, 2, 1) for x in offsets],
                             [None]*len(features), [None]*len(features))[0]
    return features, result


def assert_same_prediction(a, b):
    for key in ('segments', 'scores', 'labels', 'segments_pre_nms', 'scores_pre_nms', 'labels_pre_nms'):
        if not torch.equal(a[key], b[key]):
            raise ValueError('精简检测前向与原版预测不逐值相同: '+key)


def evaluation_columns():
    return {k: [] for k in ('video-id', 't-start', 't-end', 'label', 'score', 'cache-score')}


def append_prediction(columns, prediction, cached_scores):
    count = len(prediction['scores'])
    columns['video-id'].extend([prediction['video_id']]*count)
    for name, value in [('t-start', prediction['segments'][:, 0]), ('t-end', prediction['segments'][:, 1]),
                        ('label', prediction['labels']), ('score', prediction['scores']), ('cache-score', cached_scores)]:
        columns[name].extend(value.detach().cpu().tolist())
