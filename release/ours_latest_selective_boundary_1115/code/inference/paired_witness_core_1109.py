"""源域同类同尺度的完整/错位成对参照；真实与安慰剂只交换每对身份。"""
import numpy as np


def make_pairs(boxes, labels, duration):
    """构造固定的半宽左右平移；遇另一个同类动作或越界不作负样本。"""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 2)
    labels = np.asarray(labels, dtype=np.int64)
    positives, negatives, classes = [], [], []
    for idx, (start, end) in enumerate(boxes):
        if not (0 <= start < end <= duration):
            continue
        others = boxes[(labels == labels[idx]) & (np.arange(len(labels)) != idx)]
        for displacement in (-(end - start) / 2, (end - start) / 2):
            left, right = start + displacement, end + displacement
            if left < 0 or right > duration:
                continue
            if ((others[:, 0] < right) & (others[:, 1] > left)).any():
                continue
            positives.append((start, end))
            negatives.append((left, right))
            classes.append(labels[idx])
    return (np.asarray(positives, dtype=np.float64).reshape(-1, 2),
            np.asarray(negatives, dtype=np.float64).reshape(-1, 2),
            np.asarray(classes, dtype=np.int64))


def describe(model, maps, masks, logits, metadata, boxes, labels, force_levels=None):
    """与已归档 proposal_tf 同一坐标转换、FPN 尺度、32点采样、语义和投影。"""
    import torch
    from libs.models.SADA.uda_routes import sample
    from libs.models.SADA.class_path_alignment import path_features

    device = maps[0].device
    seg = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 2)
    classes = torch.as_tensor(labels, dtype=torch.long, device=device)
    branch = model.adapt_route
    enhanced = hasattr(branch, 'prepare_maps')
    dims = tuple(branch.descriptor_dims) if enhanced else (16, 64, 272)
    prepared = branch.prepare_maps(maps, [[int(x[b].sum()) for b in range(len(x))] for x in masks]) if enhanced else None
    with torch.inference_mode():
        if metadata['use_temporal_grid_feats']:
            seg = (seg * metadata['fps'] - .5 * metadata['feat_num_frames']) / metadata['feat_stride']
        if force_levels is None:
            radius = (seg[:, 1] - seg[:, 0]).clamp_min(1.) / 2
            ranges = model.rq3_capacity.regression_ranges
            levels = torch.full((len(seg),), len(ranges) - 1, dtype=torch.long, device=device)
            for level, (low, high) in reversed(list(enumerate(ranges))):
                levels[(radius >= low) & (radius <= high)] = level
        else:
            levels = torch.as_tensor(force_levels, dtype=torch.long, device=device)
            if levels.shape != classes.shape:
                raise ValueError("配对框尺度与类别数不同")
        result = np.empty((len(seg), sum(dims)), dtype=np.float32)
        ratios = torch.linspace(0, 1, 32, device=device)
        for level in levels.unique().tolist():
            valid = int(masks[level][0].sum())
            feature = maps[level][0, :, :valid]
            q = logits[level][0, :, :valid].sigmoid()
            for ix in torch.where(levels == level)[0].split(32):
                coords = seg[ix] / model.fpn_strides[level]
                positions = coords[:, :1] + (coords[:, 1:] - coords[:, :1]) * ratios
                windows = sample(feature, positions.flatten()).reshape(feature.shape[0], len(ix), 32)
                samples = sample(q, positions.flatten()).reshape(q.shape[0], len(ix), 32)
                semantics = samples[classes[ix], torch.arange(len(ix), device=device)][:, None, :]
                if enhanced:
                    zp, bp = prepared[level]
                    zw = sample(zp[0, :, :valid], positions.flatten()).reshape(16, len(ix), 32).permute(1, 0, 2)
                    bw = None if bp is None else sample(bp[0, :, :, :valid].flatten(0, 1), positions.flatten()).reshape(16, bp.shape[2], len(ix), 32).permute(2, 0, 1, 3)
                    components, _ = branch.describe_windows(zw, semantics, bw)
                else:
                    components, _ = path_features(windows.permute(1, 0, 2), semantics,
                                                  branch.projection, bins=4)
                result[ix.cpu().numpy()] = torch.cat([v[:, 0] for v in components], dim=1).cpu().numpy()
        return result, levels.cpu().numpy()


def _kernel(x, y, dims=(16, 64, 272)):
    """conditional_discrepancy 的三个归一化描述块和三带宽等权 RBF。"""
    result = np.zeros((len(x), len(y)), dtype=np.float32)
    edges = np.cumsum((0, *dims))
    for start, end in zip(edges[:-1], edges[1:]):
        a, b = x[:, start:end], y[:, start:end]
        distance = np.einsum('ij,ij->i', a, a)[:, None] + np.einsum('ij,ij->i', b, b)[None, :]
        distance -= 2 * (a @ b.T)
        np.maximum(distance, 0, out=distance)
        for bandwidth in (.5, 1., 2.):
            result += np.exp(-distance / bandwidth)
    return result / 9


def witness_scores(desc, labels, levels, bank):
    """输出真实因子、随机对内交换的因子、是否有参照。相同证据因子为1。"""
    desc = np.asarray(desc, dtype=np.float32)
    labels, levels = np.asarray(labels), np.asarray(levels)
    pos, neg = np.asarray(bank['pos'], np.float32), np.asarray(bank['neg'], np.float32)
    bank_labels, bank_levels = np.asarray(bank['labels']), np.asarray(bank['levels'])
    dims = tuple(int(x) for x in bank.get('descriptor_dims', (16, 64, 272)))
    if desc.shape != (len(labels), sum(dims)) or pos.shape != neg.shape or pos.shape != (len(bank_labels), sum(dims)):
        raise ValueError("时频参照数组形状不匹配")
    real, shuffled = np.ones(len(desc), np.float64), np.ones(len(desc), np.float64)
    covered = np.zeros(len(desc), bool)
    for label, level in sorted(set(zip(labels.tolist(), levels.tolist()))):
        query = np.flatnonzero((labels == label) & (levels == level))
        ref = np.flatnonzero((bank_labels == label) & (bank_levels == level))
        if not len(ref):
            continue
        covered[query] = True
        # 同一类别/层每次调用生成同一交换掩码；对每个匹配对独立抛硬币。
        rng = np.random.default_rng(np.random.SeedSequence([123, int(label), int(level)]))
        signs = rng.choice(np.array([-1., 1.]), len(ref))
        for i in range(0, len(query), 256):
            rows = query[i:i + 256]
            total = np.zeros(len(rows), np.float64)
            difference = np.zeros_like(total)
            null_difference = np.zeros_like(total)
            for j in range(0, len(ref), 512):
                indices = ref[j:j + 512]
                kp = _kernel(desc[rows], pos[indices], dims)
                kn = _kernel(desc[rows], neg[indices], dims)
                delta = kp - kn
                total += (kp + kn).sum(axis=1, dtype=np.float64)
                difference += delta.sum(axis=1, dtype=np.float64)
                null_difference += np.einsum('ij,j->i', delta, signs[j:j + len(indices)])
            denom = total + 1e-12 * len(ref)
            real[rows] += difference / denom
            shuffled[rows] += null_difference / denom
    if not np.isfinite(real).all() or not np.isfinite(shuffled).all():
        raise ValueError("时频证据含非有限数值")
    return real, shuffled, covered
