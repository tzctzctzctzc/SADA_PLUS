"""统一候选证据读出接口；只重组原计算，参数、运算顺序与输出保持不变。

时间支持来自检测器的密集类别响应，不把它称作模块B的时频描述或已学习的IoU。
Matrix抑制仍在两条时间读出分支内进行；语义支持与类别折扣仍在分支融合后应用。
"""
from contextlib import contextmanager

import numpy as np
import torch

import ours_final as frozen
from shared_recipe import strength
from shared_recipe_inference import duration_factor, installed_rule
from uniform_tricks import transform


COMMON_RULE = dict(temp=2, sigma=4, blend=.5, proto_power=1, width=1., duration_power=0.)


def semantic_evidence(labels, cosines, ready):
    """模块一已有原型证据：类别竞争与原型支持，计算不变。"""
    labels = torch.as_tensor(labels).long()
    valid = torch.as_tensor(ready, dtype=torch.bool)
    similarity = torch.from_numpy(cosines)
    rival = frozen.all_rival_factor(similarity, labels, valid).numpy()
    support = np.ones(len(labels), dtype=cosines.dtype)
    if valid.any():
        support = torch.where(valid[labels], 2 * similarity[:, valid].amax(1).sigmoid(),
                              torch.ones_like(similarity[:, 0])).numpy()
    return dict(class_competition=rival, prototype_support=support)


def temporal_evidence(labels, profiles, all_class_responses):
    """已有框内外时间支持；保留全局差值与优势比两种固定读出。"""
    own = all_class_responses[np.arange(len(labels)), labels]
    competitor = all_class_responses.argmax(1)
    top = all_class_responses[np.arange(len(labels)), competitor]
    conflict = ((competitor != labels) & (top > 1) &
                ((all_class_responses == top[:, None]).sum(1) == 1))
    discount = np.ones(len(labels))
    discount[conflict] = 2 * own[conflict] / (own[conflict] + top[conflict])
    return dict(global_support=profiles[:, 0], odds_support=profiles[:, 3],
                cross_class_discount=discount)


def suppress_branch(boxes, labels, initial, sigma):
    """原Matrix规则：稳定排序、同视频同类、分支内补偿式重叠衰减。"""
    penalty = np.zeros(len(initial))
    for c in np.unique(labels):
        ix = np.flatnonzero(labels == c)
        penalty[ix] = strength(boxes[ix], initial[ix])
    return initial * np.exp(-sigma * penalty), penalty


def score_candidates(boxes, raw, labels, cosines, ready, profiles, all_class_responses,
                     rule=COMMON_RULE, return_evidence=False):
    """同一函数产出最终分数与可检查证据，不引入任何新的质量预测头。"""
    assert all(float(rule[k]) == float(value) for k, value in COMMON_RULE.items())
    if not len(raw):
        return (raw.copy(), {}) if return_evidence else raw.copy()
    semantic = semantic_evidence(labels, cosines, ready)
    temporal = temporal_evidence(labels, profiles, all_class_responses)
    power, sigma = float(rule['proto_power']), float(rule['sigma'])
    trace = dict(**semantic, **temporal)
    branches = []
    for name in ('global', 'odds'):
        initial = raw.astype(float) * temporal[name + '_support'] * semantic['class_competition'] ** power
        suppressed, penalty = suppress_branch(boxes, labels, initial, sigma)
        branches.append(suppressed)
        if return_evidence:
            trace.update({name + '_before_suppression': initial, name + '_overlap_penalty': penalty,
                          name + '_after_suppression': suppressed})
    a, b = branches
    # 保留旧版的乘法结合顺序，不能为了叙事把所有语义项移到NMS之前。
    other = semantic['prototype_support'] ** power * temporal['cross_class_discount']
    final = (a + float(rule['blend']) * (b - a)) * other
    final *= duration_factor(boxes, float(rule['duration_power']))
    assert np.isfinite(final).all() and (final >= 0).all()
    if return_evidence:
        trace['final_score'] = final
        return final, trace
    return final


def video_readout(cache, boxes, raw, labels, cosines, ready, rule=COMMON_RULE, return_evidence=False):
    changed = transform(cache, 'control' if float(rule['temp']) == 1 else 'temperature2')
    profiles = frozen.profiles(changed, boxes, labels)
    responses = frozen.responses(changed, boxes, len(ready))
    return score_candidates(boxes, raw, labels, cosines, ready, profiles, responses, rule, return_evidence)


@contextmanager
def installed_evidence_readout(rule=COMMON_RULE):
    """兼容现有逐视频入口；仅在独立推理进程中暂时替换读出函数。"""
    with installed_rule(rule):
        previous = frozen.final_scores
        def readout(boxes, raw, labels, cosines, ready, profiles, responses):
            return score_candidates(boxes, raw, labels, cosines, ready, profiles, responses, rule)
        frozen.final_scores = readout
        try:
            yield
        finally:
            frozen.final_scores = previous
