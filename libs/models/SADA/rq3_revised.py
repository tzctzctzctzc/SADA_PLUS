"""RQ3修订：核验真实SADA伪标签；复用历史proposal embedding对齐分支。"""

import torch
from torch import nn

from .proposal_semantic import (
    VLMProposalCoAdaptationBranch, cached_proposals_for_videos, source_gt_proposals,
)


@torch.no_grad()
def verify_sada_targets(pseudo, videos, grid, valid, cache, level, confidence_threshold):
    """可信局部证据只核验同一中心/层的真实前景路由，不向整段广播。"""
    votes = torch.zeros((*valid.shape, pseudo.shape[-1]), device=pseudo.device, dtype=torch.bool)
    observed = torch.zeros_like(valid)
    for batch, video in enumerate(videos):
        meta = video['metadata']
        absolute = grid[:, 0] + float(meta.get('crop_start', 0))
        for entry in cache.get(str(meta['video_id']), ()):
            if level not in entry['source_fpn_levels']:
                continue
            if entry.get('evidence_scope') != 'anchor-local-center-only-v1':
                raise ValueError('Verify requires freshly scored local windows, not old whole-proposal scores')
            # 在特征索引空间比较，不把训练时裁掉的局部证据用来否决。
            start, end = feature_interval(entry, meta)
            index = entry['absolute_feature_index']
            half = 0.5 * float(meta['feat_num_frames']) / float(meta['feat_stride'])
            if abs(start - (index - half)) > 1e-5 or abs(end - (index + half)) > 1e-5:
                raise ValueError('Verify evidence window does not match the anchor reference grid')
            crop_start = float(meta.get('crop_start', 0))
            crop_end = crop_start + float(meta.get('crop_feat_len', video['feats'].shape[-1]))
            if start < crop_start or end > crop_end:
                continue
            matched = torch.isclose(absolute, absolute.new_tensor(float(index)), rtol=0, atol=1e-5) & valid[batch]
            observed[batch] |= matched
            if confident(entry, confidence_threshold):
                votes[batch, matched, entry['class_id']] = True
    votes = votes[valid]
    foreground = pseudo.sum(-1) > 0
    covered = (votes.sum(-1) == 1) & foreground
    agree = votes.gather(1, pseudo.argmax(-1, keepdim=True)).squeeze(1)
    rejected = covered & ~agree
    return ~rejected, {
        'verify_eligible': covered.sum().float(),
        'verify_accepted': (covered & agree).sum().float(),
        'verify_rejected': rejected.sum().float(),
        'verify_ambiguous': ((votes.sum(-1) > 1) & foreground).sum().float(),
        'verify_observed': (observed[valid] & foreground).sum().float(),
        'verify_uncertain': (observed[valid] & (votes.sum(-1) == 0) & foreground).sum().float(),
    }


def feature_interval(entry, metadata):
    offset = 0.5 * float(metadata['feat_num_frames'])
    return tuple((entry[key] * metadata['fps'] - offset) / metadata['feat_stride']
                 for key in ('proposal_start_sec', 'proposal_end_sec'))


def confident(entry, threshold):
    # null表示源域未找到合格门槛；保持SADA，不强行选最好的坏门槛。
    if threshold is None:
        return False
    scores = entry['class_probabilities']
    top = max(scores)
    return top >= threshold and scores.count(top) == 1


def fully_visible_cache(cache, videos, threshold):
    """只保留整段落在当前训练crop内的可信proposal，不裁短后沿用原标签。"""
    selected, candidates, visible = [], 0, 0
    for video in videos:
        meta = video['metadata']
        identity = str(meta['video_id'])
        crop_start = float(meta.get('crop_start', 0))
        crop_end = crop_start + float(meta.get('crop_feat_len', video['feats'].shape[-1]))
        rows = []
        for entry in cache.get(identity, ()):
            start, end = feature_interval(entry, meta)
            candidates += 1
            if start < crop_start or end > crop_end:
                continue
            visible += 1
            if confident(entry, threshold):
                rows.append(entry)
        # 相同视频可在一个batch重复，且crop不同；调用方逐视频使用结果。
        selected.append(rows)
    return selected, candidates, visible


def gradient_norm(loss, features):
    if not loss.requires_grad:
        return loss.detach().new_zeros(())
    gradients = torch.autograd.grad(loss, features, retain_graph=True, allow_unused=True)
    return sum((g.detach().square().sum() for g in gradients if g is not None),
               loss.detach().new_zeros(())).sqrt()


class RQ3ProposalPrototype(nn.Module):
    """保留24b5dfb的embedding目标：source CE + 0.1原型对齐；只修正证据筛选。

    VLM提供类别概率，embedding由学生proposal Transformer产生；不是VLM隐层向量。
    原型只由源域GT更新，目标端不读取标签或标注边界。
    """
    def __init__(self, fpn_dim, num_classes, cache, strides, regression_ranges, confidence_threshold):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.branch = VLMProposalCoAdaptationBranch(
                fpn_dim, num_classes, hidden_dim=256, embedding_dim=128,
                confidence_threshold=0.0, prototype_momentum=0.9,
            )
        self.cache, self.strides, self.regression_ranges = cache, strides, regression_ranges
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

    def forward(self, source_videos, target_videos, source_features, target_features, epoch):
        source = source_gt_proposals(source_videos, self.num_classes, len(self.strides), self.regression_ranges)
        encoded_source = self.branch.encode(source_features, source, self.strides)
        source_targets = self.branch.flatten_field(source, 'targets')
        source_loss = self.branch.source_loss(encoded_source, source)
        kept, candidates, visible = fully_visible_cache(self.cache, target_videos, self.confidence_threshold)
        target = []
        for index, video in enumerate(target_videos):
            target += cached_proposals_for_videos(
                {str(video['metadata']['video_id']): kept[index]}, [video], self.num_classes,
                len(self.strides), target_features[0].device, target_features[0].dtype)
        for proposal in target:
            for key in proposal:
                proposal[key] = proposal[key][:256]
        encoded_target = self.branch.encode(target_features, target, self.strides)
        selection = self.branch.select_vlm_target(target)
        if epoch >= 1:
            alignment, classes, _ = self.branch.prototype_alignment(
                encoded_source, source_targets, encoded_target, selection)
        else:
            self.branch._update_source_prototypes(encoded_source['embedding'], source_targets)
            alignment, classes = source_loss * 0.0, 0
        loss = source_loss + 0.1 * alignment
        return {
            'loss': loss, 'source_loss': source_loss.detach(), 'alignment_loss': alignment.detach(),
            'eligible': len(encoded_target['embedding']),
            'candidate_count': candidates, 'fully_visible_count': visible,
            'selected': int(selection['selected'].sum()) if epoch >= 1 else 0,
            'aligned_classes': classes,
            'gradient_norm': gradient_norm(loss, source_features + target_features),
            'target_gradient_norm': gradient_norm(0.1 * alignment, target_features),
        }
