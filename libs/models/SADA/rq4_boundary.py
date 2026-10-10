"""RQ4只监督定位输出：不读取目标GT，不创建分类损失/教师参数。"""
import json
import math
from pathlib import Path
import torch
from torch import nn
from .losses_aux import ctr_diou_loss_1d
from tools.rq4_protocol import load_cache, digest, file_sha, validate_gate


def feature_span(seconds, metadata):
    fps, stride = float(metadata['fps']), float(metadata['feat_stride'])
    if fps <= 0 or stride <= 0 or metadata.get('temporal_scale', 1) != 1:
        raise ValueError('unsupported temporal grid/resizing')
    return [(x * fps - .5 * float(metadata['feat_num_frames'])) / stride for x in seconds]


class RQ4Boundary(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = dict(spec)
        self.cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
        self.records = {}
        for record in self.cache['records']:
            self.records.setdefault(record['candidate']['video_id'], []).append(record)
        self.gate = None
        if spec['strategy'] == 'gated':
            if file_sha(spec['gate_path']) != spec['gate_file_sha256']:
                raise ValueError('gate file changed')
            self.gate = validate_gate(json.loads(Path(spec['gate_path']).read_text()), self.cache)

    def forward(self, videos, points, masks, offsets, active):
        # offsets: 每层B,T,2非负左右距离，已经由原SADA检测头输出。
        zero = sum((x.sum() * 0 for x in offsets))
        stats = dict(queried=0, matched=0, routable=0, accepted=0, crop_rejected=0,
                     anchor_rejected=0, gate_rejected=0)
        terms = []
        if active:
            for batch, video in enumerate(videos):
                meta = video['metadata']
                crop = float(meta.get('crop_start', 0))
                end = crop + float(meta['crop_feat_len'])
                for record in self.records.get(str(meta['video_id']), []):
                    stats['queried'] += 1
                    if record['target'] is None:
                        continue
                    stats['matched'] += 1
                    row = record['candidate']
                    start, stop = feature_span(record['target'], meta)
                    old_start, old_end = feature_span(row['proposal'], meta)
                    # 不把不完整的教师区间裁成新标签；Direct/Gated同样处理。
                    if min(start, old_start) < crop or max(stop, old_end) > end:
                        stats['crop_rejected'] += 1
                        continue
                    center = feature_span([row['anchor_sec']], meta)[0]
                    level = row['fpn_level']
                    if level >= len(points):
                        raise ValueError('cache FPN level outside student')
                    grid = points[level]
                    stride = float(grid[0, 3])
                    if stride <= 0 or not torch.allclose(grid[:, 0], grid[0, 0] + torch.arange(
                            len(grid), device=grid.device) * stride):
                        raise ValueError('irregular FPN grid')
                    position = (center - crop - float(grid[0, 0])) / stride
                    left, right = math.floor(position), math.ceil(position)
                    valid = masks[level].reshape(offsets[level].shape[:2]).bool()
                    if (not start < center < stop or not 0 <= left <= right < len(grid)
                            or not bool(valid[batch, left] & valid[batch, right])):
                        stats['anchor_rejected'] += 1
                        continue
                    stats['routable'] += 1
                    if self.gate is not None:
                        threshold = self.gate['threshold']
                        if threshold is None or self.gate['scores'][digest(row)] < threshold:
                            stats['gate_rejected'] += 1
                            continue
                    stats['accepted'] += 1
                    fraction = position - left
                    prediction = offsets[level][batch, left] * (1 - fraction) + offsets[level][batch, right] * fraction
                    target = prediction.new_tensor([(center - start) / stride, (stop - center) / stride])
                    terms.append(ctr_diou_loss_1d(prediction.unsqueeze(0), target.unsqueeze(0), reduction='sum'))
        raw = sum(terms, zero) / max(stats['routable'], 1)
        loss = self.spec['weight'] * raw
        norm = zero.detach()
        if terms and torch.is_grad_enabled() and loss.requires_grad:
            gradients = torch.autograd.grad(loss, offsets, retain_graph=True, allow_unused=True)
            norm = sum((g.detach().float().square().sum() for g in gradients if g is not None), norm).sqrt()
        return dict(stats, loss=loss, raw_loss=raw, gradient_norm=norm)
