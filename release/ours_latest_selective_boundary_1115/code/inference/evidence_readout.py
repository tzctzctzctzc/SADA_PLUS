"""训练同算子读出；A关系只调参照相似性，不产生边界伪标签。"""
import numpy as np
import torch
from paired_witness_core_1109 import describe, _kernel


@torch.inference_mode()
def describe_evidence(model, maps, masks, logits, metadata, boxes, labels, force_levels=None):
    from libs.models.SADA.b_evidence_operator import prototype_codes
    desc, levels = describe(model, maps, masks, logits, metadata, boxes, labels, force_levels)
    seg = torch.as_tensor(boxes, device=maps[0].device, dtype=maps[0].dtype).reshape(-1, 2)
    if metadata['use_temporal_grid_feats']:
        seg = (seg*metadata['fps']-.5*metadata['feat_num_frames'])/metadata['feat_stride']
    code = prototype_codes(model, maps, [seg])
    return desc, levels, None if code is None else code.cpu().numpy()


def rbf(a, b):
    distance = np.maximum(0., (a*a).sum(1)[:, None]+(b*b).sum(1)[None]-2*a@b.T)
    return sum(np.exp(-distance/h) for h in (.5, 1., 2.))/3


def score_evidence(desc, codes, labels, levels, bank, alpha=.1, query_ids=None):
    """源正负成对参照同权，源查询剔除同一物理视频；标准NMS保持外置。

    bank需包含pos/neg、pos_code/neg_code、labels/levels/ids、descriptor_dims。
    新库由describe_evidence生成，不能读旧库冒充新语义核。
    """
    result = np.ones(len(desc), np.float64)
    covered = np.zeros(len(desc), bool)
    if alpha and (codes is None or 'pos_code' not in bank):
        raise ValueError('语义核读出缺少同一A的代码，禁止静默退回旧评分')
    for c, level in sorted(set(zip(np.asarray(labels).tolist(), np.asarray(levels).tolist()))):
        query = np.flatnonzero((labels == c) & (levels == level))
        refs = np.flatnonzero((bank['labels'] == c) & (bank['levels'] == level))
        if not len(refs):
            continue
        for start in range(0, len(query), 128):
            rows = query[start:start+128]
            total, delta = np.zeros(len(rows)), np.zeros(len(rows))
            counts = np.zeros(len(rows), int)
            for j in range(0, len(refs), 256):
                ix = refs[j:j+256]
                kp = _kernel(desc[rows], bank['pos'][ix], bank['descriptor_dims'])
                kn = _kernel(desc[rows], bank['neg'][ix], bank['descriptor_dims'])
                if alpha:
                    kp *= (1-alpha)+alpha*rbf(codes[rows], bank['pos_code'][ix])
                    kn *= (1-alpha)+alpha*rbf(codes[rows], bank['neg_code'][ix])
                valid = np.ones(kp.shape, bool) if query_ids is None else np.asarray(query_ids)[rows, None] != bank['ids'][None, ix]
                counts += valid.sum(1)
                total += ((kp+kn)*valid).sum(1, dtype=np.float64)
                delta += ((kp-kn)*valid).sum(1, dtype=np.float64)
            covered[rows] = counts > 0
            result[rows] += delta/(total+1e-12*np.maximum(counts, 1))
    if not np.isfinite(result).all():
        raise ValueError('读出分数非有限')
    return result, covered
