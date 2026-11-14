"""仅用源 TRAIN 学习 Soft-NMS 前的参照强度；所有方向共享算法和超参数。"""
import numpy as np


def competing_pairs(boxes, labels, scores, truth, classes):
    """同类且唯一归属同一实例，保留高质量框和真正与它重叠的最高分对手。"""
    pairs = []
    for c in np.unique(classes):
        ix = np.flatnonzero(labels == c)
        if not len(ix):
            continue
        b = boxes[ix]
        gt = truth[classes == c]
        def iou(a, other):
            ov = np.maximum(0., np.minimum(a[:, None, 1], other[:, 1]) -
                            np.maximum(a[:, None, 0], other[:, 0]))
            return ov / np.maximum(np.diff(a, axis=1) + (other[:, 1]-other[:, 0]) - ov, 1e-12)
        matches = iou(b, gt)
        best = matches.max(1)
        unique = ((matches == best[:, None]).sum(1) == 1) & (best > 0)
        assignment = matches.argmax(1)
        for j in np.unique(assignment[unique]):
            group = np.flatnonzero(unique & (assignment == j))
            good = group[best[group].argmax()]
            if best[good] < .5:
                continue
            overlap = iou(b[group], b[good:good+1])[:, 0]
            rivals = group[(overlap >= .5) & (best[group] <= best[good]-.2)]
            if len(rivals):
                bad = rivals[scores[ix[rivals]].argmax()]
                pairs.append((ix[good], ix[bad]))
    return np.asarray(pairs, dtype=int).reshape(-1, 2)


def fit_strength(margins):
    """每视频等权的成对 logistic 风险；只训练一个从1开始的有界残差。"""
    if not margins:
        raise ValueError('没有源训练竞争对')
    raw, evidence = np.concatenate(margins).T
    weights = np.concatenate([np.full(len(m), 1/(len(m)*len(margins))) for m in margins])
    curvature = np.dot(weights, evidence**2)
    def derivative(a):
        return curvature*(a-1) - np.dot(weights, evidence/(1+np.exp(np.clip(raw+a*evidence,-50,50))))
    left, right = 0., 2.
    if curvature == 0:
        a = 1.
    elif derivative(left) >= 0:
        a = left
    elif derivative(right) <= 0:
        a = right
    else:
        for _ in range(48):
            mid = (left+right)/2
            if derivative(mid) > 0:
                right = mid
            else:
                left = mid
        a = (left+right)/2
    old, new = raw+evidence, raw+a*evidence
    return dict(strength=float(a), videos=len(margins), pairs=len(raw),
                corrected=int(np.sum((old<=0)&(new>0))), broken=int(np.sum((old>0)&(new<=0))))
