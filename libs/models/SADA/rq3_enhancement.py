"""候选段编码/源原型增强。VLM只提供旧名单中的类别，不输入目标GT。"""
import torch
from torch import nn
from torch.nn import functional as F

from .proposal_semantic import VLMProposalCoAdaptationBranch, _sample_sequence
from .rq3_revised import (RQ3ProposalPrototype, source_gt_proposals,
                          fully_visible_cache, cached_proposals_for_videos, gradient_norm)


def source_supcon(embedding, targets, temperature):
    labels = targets.argmax(-1)
    known = targets.sum(-1) > 0
    z, labels = embedding[known], labels[known]
    if len(z) < 2:
        return embedding.sum() * 0., 0
    diagonal = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.any(1)
    if not valid.any():
        return embedding.sum() * 0., 0
    logits = (z @ z.T) / temperature
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True)
    # where避免0 * inf；只在存在真实同类正样本的source anchor上计算。
    per_anchor = -torch.where(positive, log_prob, torch.zeros_like(log_prob)).sum(1)
    return (per_anchor[valid] / positive.sum(1)[valid]).mean(), int(valid.sum())


class EnhancedBranch(VLMProposalCoAdaptationBranch):
    def __init__(self, fpn_dim, num_classes, spec):
        super().__init__(fpn_dim, num_classes, hidden_dim=spec['hidden_dim'],
                         embedding_dim=spec['embedding_dim'], confidence_threshold=0.,
                         prototype_momentum=.9)
        self.spec = dict(spec)
        if spec['layers'] != 1:
            layer = nn.TransformerEncoderLayer(spec['embedding_dim'], 4, spec['hidden_dim'],
                        dropout=0., activation='gelu', batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, spec['layers'])
        if spec['pooling'] == 'split-attention':
            d = spec['embedding_dim']
            self.pool_score = nn.Linear(d, 1, bias=False)
            self.pool_fusion = nn.Linear(2*d, d)
        if spec['prototypes_per_class'] > 1:
            self.register_buffer('subprototypes', torch.zeros(num_classes, spec['prototypes_per_class'], spec['embedding_dim']))
            self.register_buffer('subprototype_ready', torch.zeros(num_classes, spec['prototypes_per_class'], dtype=torch.bool))

    def encode(self, fpn_feats, proposals, fpn_strides):
        if self.spec['pooling'] == 'mean':
            return super().encode(fpn_feats, proposals, fpn_strides)
        rows = []
        for batch_index, proposal in enumerate(proposals):
            for segment, level in zip(proposal['segments'], proposal['levels']):
                level = int(level)
                duration = (segment[1]-segment[0]).clamp(min=1.)
                positions = torch.stack([segment[0]+r*duration for r in self.sample_ratios]) / float(fpn_strides[level])
                rows.append(_sample_sequence(fpn_feats[level][batch_index], positions).T)
        if not rows:
            return super().encode(fpn_feats, proposals, fpn_strides)
        tokens = self.encoder(self.input_projection(torch.stack(rows)) + self.position_embedding[None])
        # 5个段内点和2个段外点各自加权，避免背景在最终mean中被等权混合。
        inside, context = tokens[:, 1:6], tokens[:, [0, 6]]
        def pool(x):
            return (self.pool_score(x).softmax(1) * x).sum(1)
        h = self.output_norm(self.pool_fusion(torch.cat((pool(inside), pool(context)), dim=-1)))
        return dict(embedding=F.normalize(h, dim=-1), logits=self.classifier(h))

    @torch.no_grad()
    def update_subprototypes(self, embedding, targets):
        for cls in range(self.num_classes):
            rows = embedding[targets[:, cls] > 0].detach()
            if not len(rows):
                continue
            # 用真实source样本、最远点初始化空槽；不复制同一点填满空槽。
            unused = torch.ones(len(rows), device=rows.device, dtype=torch.bool)
            for slot in range(self.spec['prototypes_per_class']):
                if self.subprototype_ready[cls, slot] or not unused.any():
                    continue
                ready = self.subprototype_ready[cls]
                scores = (rows @ self.subprototypes[cls, ready].T).max(1).values if ready.any() else rows.new_zeros(len(rows))
                idx = scores.masked_fill(~unused, torch.inf).argmin()
                self.subprototypes[cls, slot].copy_(rows[idx])
                self.subprototype_ready[cls, slot] = True
                unused[idx] = False
            ready = self.subprototype_ready[cls]
            ready_slots = ready.nonzero(as_tuple=False).flatten()
            assignment = (rows @ self.subprototypes[cls, ready].T).argmax(1)
            for local, slot in enumerate(ready_slots):
                members = rows[assignment == local]
                if len(members):
                    center = .9*self.subprototypes[cls, slot] + .1*members.mean(0)
                    self.subprototypes[cls, slot].copy_(F.normalize(center, dim=0))

    def prototype_alignment(self, source, source_targets, target, selection):
        if self.spec['prototypes_per_class'] == 1:
            return super().prototype_alignment(source, source_targets, target, selection)
        self.update_subprototypes(source['embedding'], source_targets)
        selected = selection['selected'].clone()
        labels = selection['labels']
        selected &= self.subprototype_ready.any(1)[labels]
        zero = source['embedding'].sum()*0. + target['embedding'].sum()*0.
        if not selected.any():
            return zero, 0, zero.detach()
        y = labels[selected]
        prototypes = self.subprototypes[y].detach().clone()
        cosine = (target['embedding'][selected, None] * prototypes).sum(-1)
        cosine = cosine.masked_fill(~self.subprototype_ready[y], -torch.inf).max(1).values
        return 1.-cosine.mean(), int(y.unique().numel()), cosine.mean().detach()


class EnhancedPrototype(RQ3ProposalPrototype):
    def __init__(self, fpn_dim, num_classes, cache, strides, regression_ranges, confidence_threshold, spec):
        # 保留旧分支独立seed0初始化，不消耗学生全局RNG；control严格走旧实现。
        super().__init__(fpn_dim, num_classes, cache, strides, regression_ranges, confidence_threshold)
        self.spec = dict(spec)
        if spec['variant'] != 'control':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0)
                self.branch = EnhancedBranch(fpn_dim, num_classes, spec)

    def forward(self, source_videos, target_videos, source_features, target_features, epoch):
        if self.spec['variant'] in ('control', 'wide', 'deep', 'pool', 'multiprototype'):
            return super().forward(source_videos, target_videos, source_features, target_features, epoch)
        source = source_gt_proposals(source_videos, self.num_classes, len(self.strides), self.regression_ranges)
        es = self.branch.encode(source_features, source, self.strides)
        targets = self.branch.flatten_field(source, 'targets')
        ce = self.branch.source_loss(es, source)
        kept, candidates, visible = fully_visible_cache(self.cache, target_videos, self.confidence_threshold)
        proposals = []
        for i, video in enumerate(target_videos):
            proposals += cached_proposals_for_videos({str(video['metadata']['video_id']): kept[i]},
                [video], self.num_classes, len(self.strides), target_features[0].device, target_features[0].dtype)
        for row in proposals:
            for key in row:
                row[key] = row[key][:256]
        et = self.branch.encode(target_features, proposals, self.strides)
        selection = self.branch.select_vlm_target(proposals)
        alignment, classes, _ = self.branch.prototype_alignment(es, targets, et, selection)
        contrast, anchors = source_supcon(es['embedding'], targets, self.spec['temperature']) if targets is not None else (ce*0., 0)
        warmup = self.spec['alignment_warmup_epochs']
        factor = min(1., max(0., float(epoch)/warmup)) if warmup else 1.
        if epoch < 1:
            factor = 0.
        loss = ce + self.spec['source_supcon_weight']*contrast + .1*factor*alignment
        return dict(loss=loss, source_loss=ce.detach(), alignment_loss=alignment.detach(),
            source_supcon_loss=contrast.detach(), source_supcon_anchors=anchors,
            alignment_weight=.1*factor, eligible=len(et['embedding']), candidate_count=candidates,
            fully_visible_count=visible, selected=int(selection['selected'].sum()) if epoch >= 1 else 0,
            aligned_classes=classes, gradient_norm=gradient_norm(loss, source_features+target_features),
            target_gradient_norm=gradient_norm(.1*factor*alignment, target_features))
