"""分类置信度与时间尺度定位稳定性分开；不使用目标标注或随机数。"""
import torch
import torch.nn.functional as F


def temporal_view(video, scale):
    assert 'segments' not in video and 'labels' not in video
    original = video['feats'].shape[-1]
    length = max(2, round(original * scale))
    out = dict(video)
    out['feats'] = F.interpolate(video['feats'][None], size=length,
                                  mode='linear', align_corners=False)[0]
    out['crop_length'] = length
    return out, length / original


def inverse_segments(segments, ratio, length):
    # align_corners=False: x_new + .5 = ratio * (x_original + .5).
    return ((segments + .5) / ratio - .5).clamp(0, length)


def interval_iou(a, b):
    overlap = (torch.minimum(a[:, None, 1], b[None, :, 1]) -
               torch.maximum(a[:, None, 0], b[None, :, 0])).clamp_min(0)
    union = (a[:, 1] - a[:, 0])[:, None] + (b[:, 1] - b[:, 0])[None] - overlap
    return overlap / union.clamp_min(1e-8)


def stability_filter(anchor, views, min_iou=.7, max_endpoint=.15):
    """每视图同类互为最佳IoU匹配，避免多个原候选共享一个证据。"""
    count = len(anchor['segments'])
    keep = torch.ones(count, dtype=torch.bool)
    min_overlap = torch.ones(count)
    max_error = torch.zeros(count)
    for view in views:
        if not count or not len(view['segments']):
            keep.zero_();min_overlap.zero_();continue
        overlap = interval_iou(anchor['segments'], view['segments'])
        same = anchor['labels'][:, None] == view['labels'][None]
        matching = overlap.masked_fill(~same, -1)
        score, index = matching.max(1)
        reverse = matching.argmax(0)
        mutual = reverse[index] == torch.arange(count)
        error = (anchor['segments'] - view['segments'][index]).abs().amax(1)
        error /= (anchor['segments'][:, 1] - anchor['segments'][:, 0]).clamp_min(1e-8)
        keep &= mutual & (score >= min_iou) & (error <= max_endpoint)
        min_overlap = torch.minimum(min_overlap, score.clamp_min(0))
        max_error = torch.maximum(max_error, error)
    return keep, min_overlap, max_error


@torch.no_grad()
def assignment_metadata(model, points, segments, labels, scores):
    """原官方中心/range/最短实例分配；分类点权重取并列实例最低置信度。"""
    pts = torch.cat(points)
    n = len(pts)
    if not len(segments):
        return pts.new_zeros(n, model.num_classes), pts.new_zeros(n, 2), pts.new_zeros(n)
    length = segments[:, 1] - segments[:, 0]
    left = pts[:, 0, None] - segments[None, :, 0]
    right = segments[None, :, 1] - pts[:, 0, None]
    distances = torch.stack((left, right), -1)
    if model.train_center_sample == 'radius':
        center = segments.mean(1)[None]
        radius = pts[:, 3, None] * model.train_center_sample_radius
        inside = ((pts[:, 0, None] > torch.maximum(center-radius, segments[None, :, 0])) &
                  (pts[:, 0, None] < torch.minimum(center+radius, segments[None, :, 1])))
    else:
        inside = distances.amin(-1) > 0
    maximum = distances.amax(-1)
    eligible = inside & (maximum >= pts[:, 1, None]) & (maximum <= pts[:, 2, None])
    lengths = length[None].expand(n, -1).masked_fill(~eligible, float('inf'))
    minimum, index = lengths.min(1)
    tied = (lengths <= minimum[:, None] + 1e-3) & torch.isfinite(lengths)
    cls = (tied.to(pts.dtype) @ F.one_hot(labels, model.num_classes).to(pts.dtype)).clamp(0, 1)
    offsets = distances[torch.arange(n, device=pts.device), index] / pts[:, 3, None]
    weights = scores[None].expand(n, -1).masked_fill(~tied, float('inf')).amin(1)
    weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
    return cls, offsets, weights
