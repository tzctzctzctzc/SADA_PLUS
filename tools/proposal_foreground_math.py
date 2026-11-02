"""候选区间的任务背景标签、内外表征与只删除背景的干预。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


BACKGROUND, PARTIAL, COMPLETE = 0, 1, 2


def temporal_iou(boxes, truth):
    boxes, truth = boxes.detach(), truth.detach()
    if not len(truth):
        return boxes.new_zeros((len(boxes), 0))
    intersection = (torch.minimum(boxes[:, None, 1], truth[None, :, 1]) -
                    torch.maximum(boxes[:, None, 0], truth[None, :, 0])).clamp_min(0)
    union = (boxes[:, 1] - boxes[:, 0])[:, None] + (truth[:, 1] - truth[:, 0])[None] - intersection
    return intersection / union.clamp_min(1e-8)


def interval_labels(boxes, truth):
    """严格沿用旧后NMS Oracle的any-class tIoU<.01；不是物理零交集。"""
    matrix = temporal_iou(boxes, truth)
    best = matrix.max(1).values if len(truth) else boxes.new_zeros(len(boxes))
    labels = torch.full((len(boxes),), PARTIAL, dtype=torch.long, device=boxes.device)
    labels[best < .01] = BACKGROUND
    labels[best >= .5] = COMPLETE
    return labels.detach(), best.detach()


def seconds_to_grid(boxes, metadata):
    if metadata['use_temporal_grid_feats']:
        return (boxes * metadata['fps'] - .5 * metadata['feat_num_frames']) / metadata['feat_stride']
    return boxes


def grid_to_seconds(boxes, metadata):
    if metadata['use_temporal_grid_feats']:
        return (boxes * metadata['feat_stride'] + .5 * metadata['feat_num_frames']) / metadata['fps']
    return boxes


def sample_regions(feature, valid, boxes, stride=1., origin=0.):
    """4左+16内+4右真实FPN采样；超界/缺失上下文不复制边界特征。"""
    boxes = boxes.detach()
    if not len(boxes):
        return feature.new_empty((0, feature.shape[0] * 3 + 3))
    length = boxes[:, 1] - boxes[:, 0]
    regions = ((boxes[:, 0] - .25 * length, boxes[:, 0], 4),
               (boxes[:, 0], boxes[:, 1], 16),
               (boxes[:, 1], boxes[:, 1] + .25 * length, 4))
    pooled, coverage = [], []
    for left, right, count in regions:
        fractions = (torch.arange(count, device=feature.device, dtype=feature.dtype) + .5) / count
        coordinates = (left[:, None] + (right - left)[:, None] * fractions - origin) / stride
        floor = coordinates.floor().long()
        ceil = coordinates.ceil().long()
        weight = coordinates - floor
        lo, hi = floor.clamp(0, feature.shape[-1] - 1), ceil.clamp(0, feature.shape[-1] - 1)
        exists = (coordinates >= 0) & (coordinates <= feature.shape[-1] - 1) & valid[lo] & valid[hi]
        values = (feature[:, lo] * (1 - weight)[None] + feature[:, hi] * weight[None]).permute(1, 2, 0)
        maximum = values.masked_fill(~exists[..., None], -torch.inf).max(1).values
        maximum = torch.where(exists.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        pooled.append(maximum)
        coverage.append(exists.to(feature.dtype).mean(1, keepdim=True))
    left, inner, right = pooled
    return torch.cat((inner - left, inner, inner - right, *coverage), dim=1)


class ProposalForeground(nn.Module):
    """小共享投影+两层MLP；不读取原分类分数/自预测类，不移动端点。"""
    def __init__(self, channels, width=64):
        super().__init__()
        self.projection = nn.Conv1d(channels, width, 1)
        self.classifier = nn.Sequential(nn.Linear(width * 3 + 3, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, feature, valid, boxes, stride=1., origin=0., chunk=256):
        projected = F.relu(self.projection(feature[None]))[0]
        rows = [self.classifier(sample_regions(projected, valid, part, stride, origin))
                for part in boxes.detach().split(chunk)]
        return torch.cat(rows) if rows else projected.sum() * 0 + projected.new_empty((0, 3))


def balanced_loss(logits, labels):
    """每视频三类等权；背景与不完整分别监督，推理只看背景通道。"""
    losses = [F.cross_entropy(logits[labels == k], labels[labels == k])
              for k in range(3) if (labels == k).any()]
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def calibration_threshold(rows, retained_fraction=.99):
    """仅源train留出视频；按视频等权，保留至少99%的真实同类TP@.3。"""
    positives = [r['background_probability'][r['true_positive']].double().cpu()
                 for r in rows if r['true_positive'].any()]
    if not positives:
        return dict(threshold=1.01, enabled=False, reason='no_source_true_positives', positive_videos=0)
    values = torch.cat(positives)
    weights = torch.cat([torch.full_like(x, 1 / (len(positives) * len(x))) for x in positives])
    order = values.argsort()
    index = int(torch.searchsorted(weights[order].cumsum(0), torch.tensor(retained_fraction, dtype=torch.double)))
    threshold = float(values[order[min(index, len(order) - 1)]])
    return dict(threshold=threshold, enabled=True, positive_videos=len(positives),
                retained_fraction=retained_fraction, comparison='remove_iff_probability_strictly_greater')


def filter_result(result, probability, threshold):
    """只删除最终候选；保留者所有边界/分数/类别逐值不变，不重做NMS。"""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1.01:
        raise ValueError('非法背景阈值')
    keep = probability.detach().cpu() <= threshold
    if len(keep) != len(result['scores']):
        raise ValueError('候选数与背景预测数不符')
    output = dict(result)
    for key in ('segments', 'scores', 'labels'):
        output[key] = result[key][keep.to(result[key].device)]
    return output


def true_positive_flags(boxes, labels, scores, truth, truth_labels, threshold=.3):
    """按原分数一对一匹配；不将重复框都当校准TP。"""
    matrix = temporal_iou(boxes, truth)
    used, result = set(), torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    for i in scores.argsort(descending=True).tolist():
        for j in matrix[i].argsort(descending=True).tolist():
            if matrix[i, j] < threshold:
                break
            if j not in used and labels[i] == truth_labels[j]:
                result[i] = True
                used.add(j)
                break
    return result
