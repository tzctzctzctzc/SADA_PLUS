"""一维密集伪标签的数学部分；不访问标注、文件或模型状态。"""
import torch
from torch.nn import functional as F


def map_time(x, view, inverse=False):
    if inverse:
        return (x + .5) / view['ratio'] - .5 + view['lo']
    return (x - view['lo'] + .5) * view['ratio'] - .5


def feature_view(video, step, index):
    x = video['feats']; length = x.shape[-1]
    width = min(length, max(8, int(length * .8)))
    lo = (step * 37 + index * 13) % max(1, length - width + 1)
    size = max(1, int(width * (.9, 1., 1.1)[(step + index) % 3]))
    view = dict(lo=lo, width=width, ratio=size / width, length=size)
    result = dict(video)
    result['feats'] = F.interpolate(x[:, lo:lo+width][None], size=size,
                                    mode='linear', align_corners=False)[0]
    result['segments'] = x.new_empty((0, 2))
    result['labels'] = torch.empty(0, dtype=torch.long, device=x.device)
    return result, view


def linear_sample(values, mask, coordinates, stride):
    """values=T,C；两端邻点都必须有效，禁止把padding当真实上下文。"""
    t = coordinates / stride
    lo = t.floor().long(); hi = t.ceil().long()
    valid = (lo >= 0) & (hi < len(values))
    a = lo.clamp(0, len(values)-1); b = hi.clamp(0, len(values)-1)
    valid &= mask[a].bool() & mask[b].bool()
    weight = (t-lo.to(t.dtype))[:, None]
    return values[a] * (1-weight) + values[b] * weight, valid


def decoded(points, offsets):
    center, stride = points[:, 0], points[:, 3]
    return torch.stack([center-offsets[:, 0]*stride,
                        center+offsets[:, 1]*stride], -1)


def interval_iou(a, b, pairwise=False):
    if pairwise:
        a, b = a[:, None], b[None]
    intersection = (torch.minimum(a[..., 1], b[..., 1]) -
                    torch.maximum(a[..., 0], b[..., 0])).clamp_min(0)
    union = (a[..., 1]-a[..., 0]) + (b[..., 1]-b[..., 0]) - intersection
    return intersection / union.clamp_min(1e-8)


def offset_iou(a, b):
    intersection = torch.minimum(a[:, 0], b[:, 0]) + torch.minimum(a[:, 1], b[:, 1])
    return intersection / (a.sum(-1)+b.sum(-1)-intersection).clamp_min(1e-8)


def union_contains(points, boxes):
    """排序与前缀max实现区间并集包含，避免每step构造T×T矩阵。"""
    if not len(boxes):
        return torch.zeros_like(points, dtype=torch.bool)
    order = boxes[:, 0].argsort()
    starts = boxes[order, 0].contiguous()
    ends = boxes[order, 1].cummax(0).values
    index = torch.searchsorted(starts, points.contiguous(), right=True)-1
    return (index >= 0) & (points <= ends[index.clamp_min(0)])


def class_thresholds(probabilities, valid, previous):
    """DSL公开代码的类累计置信度/平均类计数公式，使用在线EMA统计。"""
    score, label = probabilities.max(-1)
    chosen = valid & (score >= .1)
    count = torch.bincount(label[chosen], minlength=probabilities.shape[-1]).to(probabilities.dtype)
    sums = probabilities.new_zeros(probabilities.shape[-1])
    sums.scatter_add_(0, label[chosen], score[chosen])
    old_count, old_sum = previous
    counts, totals = old_count*.9+count*.1, old_sum*.9+sums*.1
    average_count = counts.sum()/len(counts)
    thresholds = ((totals/average_count.clamp_min(1e-8)).clamp_min(0).pow(.7)*.35).clamp(.25, .35)
    thresholds = torch.where(counts > 0, thresholds, thresholds.new_full(thresholds.shape, .3))
    return thresholds, (counts.detach(), totals.detach())


def quality_focal(probability, target, active, normalizer, negative_alpha=.75):
    """ARSL形式：联合概率本身参加BCE及调制；ignore位置梯度严格为0。"""
    p = probability.clamp(1e-7, 1-1e-7)
    target = target.detach()
    factor = (p-target).abs().square()
    factor = factor * torch.where(target > 0, factor.new_ones(()),
                                 factor.new_full((), negative_alpha))
    values = F.binary_cross_entropy(p, target, reduction='none')*factor
    return (values * active[:, None]).sum()/normalizer


def source_quality_targets(classes, offsets, targets, valid):
    positive = valid & (classes.sum(-1) > 0)
    iou = offsets.new_zeros(len(offsets))
    with torch.no_grad():
        iou[positive] = offset_iou(offsets[positive].detach(), targets[positive].detach())
    return classes.detach()*iou[:, None], iou, positive


@torch.no_grad()
def task_separation(probability, quality, boxes, points, levels, valid, protected, length):
    """完整TSA；一维适配显式拒绝互斥实例聚合，空正集合不强取top10。"""
    joint = probability * quality[:, None]
    score, label = joint.max(-1)
    raw_score = probability.max(-1).values
    eligible = valid & (score >= .1)
    selected = score[eligible]
    threshold = (selected.mean() + selected.std(unbiased=False)).clamp(max=.4) if len(selected) else score.new_tensor(.4)
    legal = (boxes[:, 1]-boxes[:, 0] > .05) & torch.isfinite(boxes).all(-1)
    positive = eligible & (score >= threshold) & legal
    soft = valid & (raw_score >= .1) & ~positive
    negative = valid & (raw_score < .1) & ~protected
    active = positive | soft | negative
    targets = torch.zeros_like(probability)
    ids = positive.nonzero().flatten()
    targets[ids, label[ids]] = score[ids]
    targets[soft] = joint[soft]
    reg_boxes = boxes.clone()
    visible = (boxes[:, 0] >= 0) & (boxes[:, 1] <= length)
    representable = (points[:, 0] > boxes[:, 0]) & (points[:, 0] < boxes[:, 1])
    regression = positive & visible & representable
    potential = soft & legal & visible
    pos_ids = (positive & visible).nonzero().flatten()
    cand_ids = potential.nonzero().flatten()
    mined = torch.zeros_like(positive); conflict = torch.zeros_like(positive)
    # 分块只改变内存，不丢弃任何候选。
    for chunk in cand_ids.split(128):
        if not len(pos_ids):
            break
        overlap = interval_iou(boxes[chunk], boxes[pos_ids], pairwise=True)
        matches = (overlap >= .6) & (label[chunk, None] == label[pos_ids][None])
        matches &= (levels[chunk, None]-levels[pos_ids][None]).abs() <= 1
        center = points[chunk, 0, None]
        matches &= (center > boxes[pos_ids, 0]) & (center < boxes[pos_ids, 1])
        seed = (overlap*matches).argmax(-1)
        agreement = interval_iou(boxes[pos_ids[seed]], boxes[pos_ids], pairwise=True) >= .6
        ambiguous_instance = (matches & ~agreement).any(-1)
        use = matches.any(-1) & ~ambiguous_instance
        conflict[chunk] = ambiguous_instance
        weights = matches.to(score.dtype)*score[pos_ids][None]
        aggregate = weights @ boxes[pos_ids] / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        use &= (aggregate[:, 0] < points[chunk, 0]) & (aggregate[:, 1] > points[chunk, 0])
        use &= (aggregate[:, 0] >= 0) & (aggregate[:, 1] <= length)
        reg_boxes[chunk[use]] = aggregate[use]
        regression[chunk[use]] = True; mined[chunk[use]] = True
    regression &= valid
    return dict(classes=targets, active=active, positive=positive, soft=soft,
                negative=negative, regression=regression, boxes=reg_boxes,
                quality=quality, mined=mined, conflict=conflict, threshold=threshold,
                ignored=valid & ~active)
