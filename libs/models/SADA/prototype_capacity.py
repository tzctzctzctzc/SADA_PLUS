"""复用旧Prototype的采样/CE/余弦目标，仅比较原版与加宽加深。"""
import math
import json
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def specification(variant):
    if variant in ('temporal_pool','multi_prototype'):
        from .prototype_methods import method_specification
        return method_specification(variant)
    if variant not in ('control','wide_deep'):raise ValueError('仅允许原版与直接加宽加深')
    return dict(variant=variant,embedding_dim=128 if variant=='control' else 256,
        hidden_dim=256 if variant=='control' else 512,layers=1 if variant=='control' else 2,
        heads=4,dropout=0.,pooling='mean',prototypes_per_class=1,
        prototype_momentum=.9,source_weight=1.,target_alignment_weight=.1)


def build_capacity(cfg,fpn_dim,num_classes,strides,regression_ranges,device):
    spec=specification(cfg['variant'])
    path=Path(cfg['prepared_cache'])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=cfg['prepared_cache_sha256']:
        raise ValueError('冻结教师缓存发生变化')
    payload=json.loads(path.read_text())
    oracle = payload.get('schema') == 'prototype-gt-oracle-cache-v1'
    if oracle != (cfg.get('oracle_auxiliary_ground_truth') is True):
        raise ValueError('GT oracle缓存必须显式授权，不能冒充普通VLM缓存')
    # 与基线共享检测器初始化与采样轨迹，新增分支只在隔离随机域中初始化。
    devices=[0] if torch.device(device).type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        result=RQ3ProposalPrototype(fpn_dim,num_classes,payload['by_video'],strides,regression_ranges,0.)
        if cfg['variant'] in ('temporal_pool','multi_prototype'):
            from .prototype_methods import make_method_branch
            torch.manual_seed(0)
            result.branch=make_method_branch(cfg['variant'],fpn_dim,num_classes)
        if cfg['variant']=='wide_deep':
            torch.manual_seed(0)
            result.branch=VLMProposalCoAdaptationBranch(fpn_dim,num_classes,
                hidden_dim=512,embedding_dim=256,confidence_threshold=0.,prototype_momentum=.9)
            layer=nn.TransformerEncoderLayer(256,4,512,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
            result.branch.encoder=nn.TransformerEncoder(layer,2)
    result.specification=spec
    return result


# 旧实现来源SHA256: proposal_semantic.py 8a421a523d62fcb7b1e5a7ecafad768752194a5da1ab6f2ea843d96408855a5a

KNOWN_LABELS = (
    "TAKE",
    "PUT",
    "WASH",
    "OPEN",
    "CLOSE",
    "INSERT",
    "TURN_ON",
    "CUT",
    "TURN_OFF",
    "POUR",
)

def evidence_vector(label, num_classes, device, dtype, probabilities=None):
    vector = torch.zeros(num_classes + 4, device=device, dtype=dtype)
    known_label_count = min(int(num_classes), len(KNOWN_LABELS))
    if label in KNOWN_LABELS[:known_label_count]:
        if probabilities is None:
            vector[KNOWN_LABELS.index(label)] = 1.0
        else:
            probabilities = tuple(float(value) for value in probabilities)
            if len(probabilities) == known_label_count:
                probabilities = probabilities
            elif len(probabilities) == len(KNOWN_LABELS):
                probabilities = probabilities[:known_label_count]
            else:
                raise ValueError("VLM class probabilities have an invalid size")
            vector[:known_label_count] = torch.as_tensor(
                probabilities, device=device, dtype=dtype
            )
    elif label == "UNKNOWN_ACTION":
        vector[num_classes] = 1.0
    elif label == "BACKGROUND":
        vector[num_classes + 1] = 1.0
    elif label == "ABSTAIN":
        vector[num_classes + 2] = 1.0
    else:
        raise ValueError("unknown proposal semantic label")
    return vector

def cached_proposals_for_videos(cache, videos, num_classes, n_levels, device, dtype):
    proposals = []
    for video in videos:
        metadata = video["metadata"]
        video_id = str(metadata["video_id"])
        crop_start = float(metadata.get("crop_start", 0))
        crop_length = float(metadata.get("crop_feat_len", video["feats"].shape[-1]))
        stride = float(metadata["feat_stride"])
        frame_offset = 0.5 * float(metadata["feat_num_frames"])
        fps = float(metadata["fps"])
        rows = []
        for entry in cache.get(video_id, []):
            start = (entry["proposal_start_sec"] * fps - frame_offset) / stride
            end = (entry["proposal_end_sec"] * fps - frame_offset) / stride
            start -= crop_start
            end -= crop_start
            if end <= 0.0 or start >= crop_length:
                continue
            level = entry["source_fpn_levels"][0]
            if level >= n_levels:
                continue
            rows.append((
                max(start, 0.0), min(end, crop_length), level,
                entry["semantic_label"], entry["candidate_id"],
                entry.get("class_probabilities"),
                entry.get("teacher_score", 1.0),
                entry.get("proposal_quality_weight", 1.0),
            ))
        if rows:
            segments = torch.tensor(
                [[row[0], row[1]] for row in rows], device=device, dtype=dtype
            )
            levels = torch.tensor(
                [row[2] for row in rows], device=device, dtype=torch.long
            )
            evidence = torch.stack([
                evidence_vector(
                    row[3], num_classes, device, dtype, row[5]
                ) for row in rows
            ])
            eligible = evidence[:, :num_classes].sum(dim=-1) > 0
            teacher_scores = torch.tensor(
                [row[6] for row in rows], device=device, dtype=dtype
            )
            quality_weights = torch.tensor(
                [row[7] for row in rows], device=device, dtype=dtype
            )
        else:
            segments = torch.zeros((0, 2), device=device, dtype=dtype)
            levels = torch.empty(0, device=device, dtype=torch.long)
            evidence = torch.zeros((0, num_classes + 4), device=device, dtype=dtype)
            eligible = torch.empty(0, device=device, dtype=torch.bool)
            teacher_scores = torch.empty(0, device=device, dtype=dtype)
            quality_weights = torch.empty(0, device=device, dtype=dtype)
        proposals.append({
            "segments": segments,
            "levels": levels,
            "evidence": evidence,
            "evidence_eligible": eligible,
            "teacher_scores": teacher_scores,
            "quality_weights": quality_weights,
        })
    return proposals

def source_gt_proposals(videos, num_classes, n_levels, regression_ranges):
    proposals = []
    for video in videos:
        segments = video["segments"]
        labels = video["labels"]
        levels = []
        for segment in segments:
            radius = float((segment[1] - segment[0]).clamp(min=1.0) / 2.0)
            level = n_levels - 1
            for candidate, bounds in enumerate(regression_ranges):
                if float(bounds[0]) <= radius <= float(bounds[1]):
                    level = candidate
                    break
            levels.append(level)
        evidence = segments.new_zeros((segments.shape[0], num_classes + 4))
        if segments.shape[0]:
            evidence[:, -1] = 1.0
        targets = segments.new_zeros((segments.shape[0], num_classes))
        known = labels < num_classes
        if known.any():
            targets[known] = F.one_hot(labels[known], num_classes=num_classes).float()
        proposals.append({
            "segments": segments.detach(),
            "levels": torch.tensor(levels, dtype=torch.long, device=segments.device),
            "evidence": evidence,
            "evidence_eligible": known,
            "targets": targets,
        })
    return proposals

def _sample_sequence(sequence, positions):
    """Linearly sample a [C, T] sequence at feature-index positions."""
    length = sequence.shape[-1]
    positions = positions.clamp(0, max(length - 1, 0))
    left = positions.floor().long()
    right = (left + 1).clamp(max=max(length - 1, 0))
    weight = (positions - left.float()).unsqueeze(0)
    return sequence[:, left] * (1.0 - weight) + sequence[:, right] * weight

class VLMProposalCoAdaptationBranch(nn.Module):
    """Learn domain-adapted proposal representations from positive VLM evidence.

    Unlike the logit residual, this branch consumes only visual proposal tokens and
    sends gradients into the shared FPN.  VLM outputs choose a positive source
    prototype; they are never part of the visual encoder input and never provide
    negative class supervision.
    """

    sample_ratios = (-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25)

    def __init__(
        self,
        fpn_dim,
        num_classes,
        hidden_dim=256,
        embedding_dim=128,
        confidence_threshold=0.7,
        prototype_momentum=0.9,
    ):
        super().__init__()
        if embedding_dim % 4:
            raise ValueError("proposal embedding dimension must be divisible by 4")
        self.num_classes = int(num_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.prototype_momentum = float(prototype_momentum)
        self.input_projection = nn.Linear(int(fpn_dim), int(embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(
            len(self.sample_ratios), int(embedding_dim)
        ))
        layer = nn.TransformerEncoderLayer(
            d_model=int(embedding_dim),
            nhead=4,
            dim_feedforward=int(hidden_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.output_norm = nn.LayerNorm(int(embedding_dim))
        # No bias: target positive supervision must reshape visual proposal
        # embeddings instead of being absorbed by a class-wise constant.
        self.classifier = nn.Linear(
            int(embedding_dim), self.num_classes, bias=False
        )
        self.register_buffer(
            "source_prototypes",
            torch.zeros(self.num_classes, int(embedding_dim)),
        )
        self.register_buffer(
            "source_prototype_ready",
            torch.zeros(self.num_classes, dtype=torch.bool),
        )

    @staticmethod
    def flatten_field(proposals, field):
        values = [
            proposal[field]
            for proposal in proposals
            if field in proposal and proposal[field].numel()
        ]
        return torch.cat(values) if values else None

    def encode(self, fpn_feats, proposals, fpn_strides):
        rows = []
        for batch_index, proposal in enumerate(proposals):
            for segment, level in zip(proposal["segments"], proposal["levels"]):
                level = int(level)
                duration = (segment[1] - segment[0]).clamp(min=1.0)
                positions = torch.stack([
                    segment[0] + ratio * duration for ratio in self.sample_ratios
                ]) / float(fpn_strides[level])
                rows.append(
                    _sample_sequence(fpn_feats[level][batch_index], positions)
                    .transpose(0, 1)
                )
        if not rows:
            empty = fpn_feats[0].new_zeros((0, self.classifier.in_features))
            return {
                "embedding": empty,
                "logits": fpn_feats[0].new_zeros((0, self.num_classes)),
            }
        tokens = self.input_projection(torch.stack(rows))
        tokens = tokens + self.position_embedding.unsqueeze(0)
        tokens = self.encoder(tokens)
        embedding = self.output_norm(tokens.mean(dim=1))
        logits = self.classifier(embedding)
        return {
            "embedding": F.normalize(embedding, dim=-1),
            "logits": logits,
        }

    def source_loss(self, encoded, proposals):
        targets = self.flatten_field(proposals, "targets")
        if targets is None or not targets.numel():
            return encoded["logits"].sum() * 0.0
        known = targets.sum(dim=-1) > 0
        if not known.any():
            return encoded["logits"].sum() * 0.0
        loss = -(
            targets[known]
            * encoded["logits"][known].log_softmax(dim=-1)
        ).sum(dim=-1).mean()
        return loss / math.log(max(self.num_classes, 2))

    def select_vlm_target(self, proposals):
        evidence = self.flatten_field(proposals, "evidence")
        eligible = self.flatten_field(proposals, "evidence_eligible")
        if evidence is None:
            empty = self.source_prototypes.new_zeros((0, self.num_classes))
            return {
                "q": empty,
                "labels": torch.empty(
                    0, dtype=torch.long, device=self.source_prototypes.device
                ),
                "selected": torch.empty(
                    0, dtype=torch.bool, device=self.source_prototypes.device
                ),
                "confidence": empty.new_zeros(0),
                "entropy": empty.new_zeros(0),
            }
        q = evidence[:, :self.num_classes].detach()
        confidence, labels = q.max(dim=-1)
        selected = eligible.detach().bool() & (
            confidence >= self.confidence_threshold
        )
        known_count = min(self.num_classes, len(KNOWN_LABELS))
        entropy = -(q.clamp(min=1e-8).log() * q).sum(dim=-1)
        entropy = entropy / math.log(max(known_count, 2))
        return {
            "q": q,
            "labels": labels,
            "selected": selected,
            "confidence": confidence,
            "entropy": entropy,
        }

    def target_positive_loss(self, encoded, selection):
        selected = selection["selected"]
        if not selected.any():
            return encoded["logits"].sum() * 0.0, 0
        labels = selection["labels"][selected]
        logits = encoded["logits"][selected]
        positive_logits = logits.gather(1, labels[:, None]).squeeze(1)
        return F.softplus(-positive_logits).mean(), int(selected.sum())

    def prototype_alignment(self, source, source_targets, target, selection):
        zero = source["embedding"].sum() * 0.0 + target["embedding"].sum() * 0.0
        self._update_source_prototypes(source["embedding"], source_targets)
        selected = selection["selected"].clone()
        if not selected.any():
            return zero, 0, zero.detach()
        labels = selection["labels"]
        selected &= self.source_prototype_ready[labels]
        if not selected.any():
            return zero, 0, zero.detach()
        target_embedding = target["embedding"][selected]
        prototypes = self.source_prototypes[labels[selected]].detach()
        similarity = (target_embedding * prototypes).sum(dim=-1)
        aligned_classes = int(labels[selected].unique().numel())
        return 1.0 - similarity.mean(), aligned_classes, similarity.mean().detach()

    @torch.no_grad()
    def _update_source_prototypes(self, embeddings, targets):
        if targets is None or not targets.numel():
            return
        class_sums = targets.transpose(0, 1) @ embeddings.detach()
        class_weights = targets.sum(dim=0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(class_sums)
            torch.distributed.all_reduce(class_weights)
        for class_id in range(self.num_classes):
            if float(class_weights[class_id]) <= 1e-6:
                continue
            center = class_sums[class_id] / class_weights[class_id]
            if self.source_prototype_ready[class_id]:
                center = (
                    self.prototype_momentum * self.source_prototypes[class_id]
                    + (1.0 - self.prototype_momentum) * center
                )
            self.source_prototypes[class_id].copy_(F.normalize(center, dim=0))
            self.source_prototype_ready[class_id] = True

# 旧实现来源SHA256: rq3_revised.py d3d39bdd6cfad43a4bcef3e2066cbcd1aa7f56c26410bb67ef5fc04e9ef9284f

def feature_interval(entry, metadata):
    offset = 0.5 * float(metadata['feat_num_frames'])
    return tuple((entry[key] * metadata['fps'] - offset) / metadata['feat_stride']
                 for key in ('proposal_start_sec', 'proposal_end_sec'))

def confident(entry, threshold):
    if not entry.get('participation_selected', True):
        return False
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

    def forward(self, source_videos, target_videos, source_features, target_features, epoch,
                source_masks=None,target_masks=None):
        source = source_gt_proposals(source_videos, self.num_classes, len(self.strides), self.regression_ranges)
        encoded_source = (self.branch.encode_masked(source_features,source,self.strides,source_masks)
                          if hasattr(self.branch,'encode_masked') else self.branch.encode(source_features,source,self.strides))
        source_targets = self.branch.flatten_field(source, 'targets')
        if source_targets is not None and 'valid' in encoded_source:
            source_targets=source_targets*encoded_source['valid'][:,None].to(source_targets.dtype)
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
        encoded_target = (self.branch.encode_masked(target_features,target,self.strides,target_masks)
                          if hasattr(self.branch,'encode_masked') else self.branch.encode(target_features,target,self.strides))
        selection = self.branch.select_vlm_target(target)
        if 'valid' in encoded_target:selection['selected'] &= encoded_target['valid']
        if epoch >= 1:
            alignment, classes, _ = self.branch.prototype_alignment(
                encoded_source, source_targets, encoded_target, selection)
        else:
            self.branch._update_source_prototypes(encoded_source['embedding'], source_targets)
            alignment, classes = source_loss * 0.0, 0
        loss = source_loss + 0.1 * alignment
        result = {
            'loss': loss, 'source_loss': source_loss.detach(), 'alignment_loss': alignment.detach(),
            'eligible': len(encoded_target['embedding']),
            'candidate_count': candidates, 'fully_visible_count': visible,
            'selected': int(selection['selected'].sum()) if epoch >= 1 else 0,
            'aligned_classes': classes,
            'gradient_norm': gradient_norm(loss, source_features + target_features),
            'target_gradient_norm': gradient_norm(0.1 * alignment, target_features),
        }
        for domain,encoded in [('source',encoded_source),('target',encoded_target)]:
            result.update({domain+'_'+k:v for k,v in encoded.get('statistics',{}).items()})
        if hasattr(self.branch,'diagnostics'):result.update(self.branch.diagnostics())
        return result
