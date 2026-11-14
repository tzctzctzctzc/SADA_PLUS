"""六向共享推理配方；参数由一个全局索引绑定，禁止按S分支。"""
import numpy as np
import torch
from ours_final import all_rival_factor, profiles, responses
from uniform_tricks import transform


def strength(boxes, scores):
    order = np.argsort(-scores, kind='stable')
    a, b = boxes[order].T
    w = b - a
    inter = np.maximum(0, np.minimum(b[:, None], b) - np.maximum(a[:, None], a))
    union = w[:, None] + w - inter
    overlap = np.triu(np.divide(inter, union, out=np.zeros_like(inter), where=union > 0), 1)
    comp = overlap.max(0)
    value = np.maximum(0, (overlap**2 - comp[:, None]**2).max(0))
    result = np.empty(len(order))
    result[order] = value
    return result


def factors(boxes, raw, labels, cos, ready, temporal, evidence, powers=(0, 1, 2), sigmas=(1, 2, 4)):
    classes = torch.as_tensor(labels).long()
    valid = torch.as_tensor(ready, dtype=torch.bool)
    similarity = torch.from_numpy(cos)
    rival = all_rival_factor(similarity, classes, valid).numpy()
    boost = np.ones(len(labels), dtype=cos.dtype)
    if valid.any():
        boost = torch.where(valid[classes], 2 * similarity[:, valid].amax(1).sigmoid(), torch.ones_like(similarity[:, 0])).numpy()
    own = evidence[np.arange(len(labels)), labels]
    competitor = evidence.argmax(1)
    top = evidence[np.arange(len(labels)), competitor]
    mask = (competitor != labels) & (top > 1) & ((evidence == top[:, None]).sum(1) == 1)
    discount = np.ones(len(labels))
    discount[mask] = 2 * own[mask] / (own[mask] + top[mask])
    result = {}
    for power in powers:
        branches = []
        for column in (0, 3):
            initial = raw.astype(float) * temporal[:, column] * rival**power
            penalty = np.zeros(len(raw))
            for c in np.unique(labels):
                ix = np.flatnonzero(labels == c)
                penalty[ix] = strength(boxes[ix], initial[ix])
            branches.append({sigma: initial * np.exp(-sigma * penalty) for sigma in sigmas})
        for sigma in sigmas:
            result[power, sigma] = (branches[0][sigma], branches[1][sigma], boost**power * discount)
    return result


def video_factors(cache, boxes, raw, labels, cos, ready, factor_rules):
    values = np.zeros((len(factor_rules), len(raw)))
    if not len(raw):
        return values
    for temperature in sorted({r['temp'] for r in factor_rules}):
        changed = transform(cache, 'control' if temperature == 1 else 'temperature2')
        temporal = profiles(changed, boxes, labels)
        evidence = responses(changed, boxes, len(ready))
        components = factors(boxes, raw, labels, cos, ready, temporal, evidence)
        for k, rule in enumerate(factor_rules):
            if rule['temp'] == temperature:
                a, b, other = components[rule['proto_power'], rule['sigma']]
                values[k] = (a + rule['blend'] * (b - a)) * other
    return values
