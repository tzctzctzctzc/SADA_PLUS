"""RQ3修订：核验真实SADA伪标签；复用历史proposal embedding对齐分支。"""

import torch
from torch import nn

from .proposal_semantic import (
    VLMProposalCoAdaptationBranch, cached_proposals_for_videos, source_gt_proposals,
)


@torch.no_grad()
def verify_sada_targets(pseudo, videos, grid, valid, cache, level):
    """只否决有一致VLM证据覆盖的前景路由；不重标注、不把拒绝项变成背景。

    证据作用域为原proposal区间和原FPN层。重叠proposal类别冲突时不干预。
    未覆盖位置及SADA原有背景保持原样，GT字段完全不读取。
    """
    votes = torch.zeros((*valid.shape, pseudo.shape[-1]), device=pseudo.device, dtype=torch.bool)
    for batch, video in enumerate(videos):
        meta = video['metadata']
        absolute = grid[:, 0] + float(meta.get('crop_start', 0))
        for entry in cache.get(str(meta['video_id']), ()):
            if level not in entry['source_fpn_levels']:
                continue
            offset = 0.5 * float(meta['feat_num_frames'])
            start = (entry['proposal_start_sec'] * meta['fps'] - offset) / meta['feat_stride']
            end = (entry['proposal_end_sec'] * meta['fps'] - offset) / meta['feat_stride']
            inside = (absolute >= start) & (absolute < end) & valid[batch]
            votes[batch, inside, entry['class_id']] = True
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
    }


def gradient_norm(loss, features):
    if not loss.requires_grad:
        return loss.detach().new_zeros(())
    gradients = torch.autograd.grad(loss, features, retain_graph=True, allow_unused=True)
    return sum((g.detach().square().sum() for g in gradients if g is not None),
               loss.detach().new_zeros(())).sqrt()


class RQ3ProposalPrototype(nn.Module):
    """与24b5dfb的vlm_cotrain embedding策略相同：source CE + 0.1原型对齐。

    VLM提供类别概率，embedding由学生proposal Transformer产生；不是VLM隐层向量。
    原型只由源域GT更新，目标端不读取标签或标注边界。
    """
    def __init__(self, fpn_dim, num_classes, cache, strides, regression_ranges):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.branch = VLMProposalCoAdaptationBranch(
                fpn_dim, num_classes, hidden_dim=256, embedding_dim=128,
                confidence_threshold=0.1, prototype_momentum=0.9,
            )
        self.cache, self.strides, self.regression_ranges = cache, strides, regression_ranges
        self.num_classes = num_classes

    def forward(self, source_videos, target_videos, source_features, target_features, epoch):
        source = source_gt_proposals(source_videos, self.num_classes, len(self.strides), self.regression_ranges)
        encoded_source = self.branch.encode(source_features, source, self.strides)
        source_targets = self.branch.flatten_field(source, 'targets')
        source_loss = self.branch.source_loss(encoded_source, source)
        target = cached_proposals_for_videos(self.cache, target_videos, self.num_classes,
                                            len(self.strides), target_features[0].device,
                                            target_features[0].dtype)
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
            'selected': int(selection['selected'].sum()) if epoch >= 1 else 0,
            'aligned_classes': classes,
            'gradient_norm': gradient_norm(loss, source_features + target_features),
            'target_gradient_norm': gradient_norm(0.1 * alignment, target_features),
        }
