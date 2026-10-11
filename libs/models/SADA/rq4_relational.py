"""只约束实例内部位置的绝对边界关系，不回归教师端点。"""
import math
import torch
from torch import nn
from .rq4_boundary import feature_span
from tools.rq4_protocol import load_cache
from tools.rq4_relational import validate_spec, RULE


def select_points(grid, valid, core, crop):
    absolute = grid[:, 0] + crop
    indices = torch.where(valid & (absolute > core[0]) & (absolute < core[1]))[0]
    if len(indices) > RULE['max_points']:
        indices = indices[torch.tensor([0, (len(indices)-1)//2, len(indices)-1], device=indices.device)]
    return indices


def consistency_loss(prediction, centers, stride, duration):
    # 不同位置的left/right距离本来就不同；先解码至同一全视频坐标。
    start = centers - prediction[:, 0] * stride
    end = centers + prediction[:, 1] * stride
    boundaries = torch.stack((start, end), dim=-1)
    return (boundaries[1:] - boundaries[:-1]).abs().mean() / duration


class RQ4Relational(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = dict(spec)
        cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
        regions = validate_spec(spec, cache)
        self.records = {}
        for record, region in zip(cache['records'], regions['rows']):
            self.records.setdefault(record['candidate']['video_id'], []).append((record,region))

    def forward(self, videos, points, masks, offsets, active):
        zero = sum(x.sum()*0 for x in offsets)
        stats = dict(queried=0, matched=0, routable=0, accepted=0, crop_rejected=0,
            anchor_rejected=0, gate_rejected=0, core_rejected=0, multipoint_rejected=0,
            relation_points=0, relation_pairs=0)
        terms = []
        if active:
            for batch, video in enumerate(videos):
                meta = video['metadata']
                crop = float(meta.get('crop_start',0))
                end = crop + float(meta['crop_feat_len'])
                for record, region in self.records.get(str(meta['video_id']), []):
                    stats['queried'] += 1
                    if record['target'] is None: continue
                    stats['matched'] += 1
                    row = record['candidate']
                    start,stop = feature_span(record['target'],meta)
                    old_start,old_end = feature_span(row['proposal'],meta)
                    # 与旧Gated相同的可路由分母，不将拒绝项重新放大。
                    if min(start,old_start) < crop or max(stop,old_end) > end:
                        stats['crop_rejected'] += 1
                        continue
                    center = feature_span([row['anchor_sec']],meta)[0]
                    level = row['fpn_level']
                    if level >= len(points): raise ValueError('cache FPN level outside student')
                    grid = points[level]
                    stride = float(grid[0,3])
                    if stride <= 0 or not torch.allclose(grid[:,0], grid[0,0] + torch.arange(
                            len(grid),device=grid.device)*stride):
                        raise ValueError('irregular FPN grid')
                    position = (center-crop-float(grid[0,0]))/stride
                    left,right = math.floor(position),math.ceil(position)
                    valid = masks[level].reshape(offsets[level].shape[:2]).bool()
                    if (not start < center < stop or not 0 <= left <= right < len(grid)
                            or not bool(valid[batch,left] & valid[batch,right])):
                        stats['anchor_rejected'] += 1
                        continue
                    stats['routable'] += 1
                    if region['core'] is None:
                        stats['gate_rejected'] += 1
                        continue
                    core = feature_span(region['core'],meta)
                    if not core[0] < center < core[1]:
                        stats['core_rejected'] += 1
                        continue
                    indices = select_points(grid, valid[batch], core, crop)
                    if len(indices) < RULE['min_points']:
                        stats['multipoint_rejected'] += 1
                        continue
                    stats['accepted'] += 1
                    stats['relation_points'] += len(indices)
                    stats['relation_pairs'] += len(indices)-1
                    terms.append(consistency_loss(offsets[level][batch,indices],
                        grid[indices,0]+crop, stride, old_end-old_start))
        raw = sum(terms,zero)/max(stats['routable'],1)
        loss = self.spec['weight']*raw
        norm = zero.detach()
        if terms and torch.is_grad_enabled() and loss.requires_grad:
            gradients = torch.autograd.grad(loss,offsets,retain_graph=True,allow_unused=True)
            norm = sum((g.detach().float().square().sum() for g in gradients if g is not None),norm).sqrt()
        return dict(stats,loss=loss,raw_loss=raw,gradient_norm=norm)
