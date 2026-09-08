import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROPOSAL_SEMANTIC_MODES = (
    "off",
    "source",
    "teacher",
    "oracle_center",
    "oracle_dominant",
    "oracle_soft",
    "oracle_adapter_center",
    "oracle_adapter_dominant",
    "oracle_adapter_soft",
    "vlm",
    "vlm_logit_soft",
    "vlm_logit_confident",
    "vlm_logit_agree",
)
PROPOSAL_SEMANTIC_CACHE_SCHEMA = "proposal-semantic-evidence-v1"
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
SPECIAL_LABELS = ("UNKNOWN_ACTION", "BACKGROUND", "ABSTAIN")


def load_proposal_semantic_cache(path, num_classes):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != PROPOSAL_SEMANTIC_CACHE_SCHEMA:
        raise ValueError("unsupported proposal semantic cache schema")
    if payload.get("source") not in (
        "offline_vlm_proposal_semantics",
        "offline_vlm_consensus_proposal_semantics",
    ):
        raise ValueError("invalid proposal semantic cache source")
    if payload.get("domain") not in ("source", "target"):
        raise ValueError("proposal semantic cache has an invalid domain")
    if payload.get("split") not in ("train", "val"):
        raise ValueError("proposal semantic cache has an invalid split")
    if not isinstance(payload.get("selection_teacher_provenance"), dict):
        raise ValueError("proposal semantic cache has no selection provenance")
    if not isinstance(payload.get("prompt_protocol"), dict):
        raise ValueError("proposal semantic cache has no prompt protocol")

    known_label_count = min(int(num_classes), len(KNOWN_LABELS))
    by_video = {}
    seen = set()
    for entry in payload.get("entries", []):
        if set(entry).intersection({
            "gt_assigned_class_ids",
            "center_all_gt_class_ids",
            "semantic_gt_kind",
        }):
            raise ValueError("proposal semantic cache contains forbidden GT fields")
        candidate_id = str(entry["candidate_id"])
        if candidate_id in seen:
            raise ValueError("duplicate proposal semantic candidate")
        seen.add(candidate_id)
        video_id = str(entry["video_id"])
        start = float(entry["proposal_start_sec"])
        end = float(entry["proposal_end_sec"])
        center = float(entry["absolute_center_time"])
        if not all(math.isfinite(value) for value in (start, end, center)):
            raise ValueError("proposal semantic cache contains non-finite times")
        if start < 0.0 or end <= start or not start <= center <= end:
            raise ValueError("proposal semantic cache contains an invalid interval")
        label = str(entry["semantic_label"])
        if label not in KNOWN_LABELS[:known_label_count] + SPECIAL_LABELS:
            raise ValueError("proposal semantic cache contains an invalid label")
        levels = tuple(int(level) for level in entry.get("source_fpn_levels", []))
        if not levels or len(set(levels)) != len(levels) or any(
            level < 0 for level in levels
        ):
            raise ValueError("proposal semantic cache contains invalid FPN levels")
        probabilities = entry.get("class_probabilities")
        if probabilities is not None:
            probabilities = tuple(float(value) for value in probabilities)
            if (
                # VLM predictions use the fixed ten-class benchmark vocabulary;
                # TAL may still be configured with the original 97 classes.
                len(probabilities) != len(KNOWN_LABELS)
                or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-4
            ):
                raise ValueError("proposal semantic cache has invalid probabilities")
            if label not in KNOWN_LABELS[:known_label_count]:
                raise ValueError("special proposal label cannot have class probabilities")
        by_video.setdefault(video_id, []).append({
            "candidate_id": candidate_id,
            "absolute_feature_index": int(entry["absolute_feature_index"]),
            "proposal_start_sec": start,
            "proposal_end_sec": end,
            "absolute_center_time": center,
            "semantic_label": label,
            "source_fpn_levels": levels,
            "class_probabilities": probabilities,
        })
    return by_video, len(seen), hashlib.sha256(raw).hexdigest(), payload


def _temporal_iou(one, many):
    left = torch.maximum(one[0], many[:, 0])
    right = torch.minimum(one[1], many[:, 1])
    intersection = (right - left).clamp(min=0)
    union = (one[1] - one[0]) + (many[:, 1] - many[:, 0]) - intersection
    return intersection / union.clamp(min=1e-6)


def temporal_nms(segments, scores, max_proposals, iou_threshold):
    if segments.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=segments.device)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() and len(keep) < max_proposals:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        order = rest[_temporal_iou(segments[current], segments[rest]) <= iou_threshold]
    return torch.stack(keep)


def build_teacher_proposals(
    points,
    masks,
    logits,
    offsets,
    levels,
    score_threshold,
    max_proposals,
    nms_threshold,
):
    """Decode class-agnostic proposals from a frozen SADA teacher."""
    enabled = set(int(level) for level in levels)
    batch_size = logits[0].shape[0]
    proposals = []
    for batch_index in range(batch_size):
        segments, scores, probabilities, proposal_levels = [], [], [], []
        for level, (grid, valid, level_logits, level_offsets) in enumerate(
            zip(points, masks, logits, offsets)
        ):
            if level not in enabled:
                continue
            valid = valid[batch_index].bool()
            probs = level_logits[batch_index].sigmoid()
            level_scores = probs.max(dim=-1).values
            selected = valid & (level_scores >= score_threshold)
            indices = torch.nonzero(selected).flatten()
            if not indices.numel():
                continue
            centers = grid[indices, 0]
            strides = grid[indices, 3]
            decoded = level_offsets[batch_index, indices].detach()
            level_segments = torch.stack(
                (centers - decoded[:, 0] * strides,
                 centers + decoded[:, 1] * strides),
                dim=-1,
            )
            segments.append(level_segments)
            scores.append(level_scores[indices])
            probabilities.append(probs[indices])
            proposal_levels.append(torch.full(
                (indices.numel(),), level, dtype=torch.long, device=grid.device
            ))
        if not segments:
            proposals.append({
                "segments": logits[0].new_zeros((0, 2)),
                "levels": torch.empty(0, dtype=torch.long, device=logits[0].device),
                "evidence": logits[0].new_zeros((0, logits[0].shape[-1] + 4)),
                "evidence_eligible": torch.empty(
                    0, dtype=torch.bool, device=logits[0].device
                ),
            })
            continue
        segments = torch.cat(segments)
        scores = torch.cat(scores)
        probabilities = torch.cat(probabilities)
        proposal_levels = torch.cat(proposal_levels)
        keep = temporal_nms(segments, scores, max_proposals, nms_threshold)
        probabilities = probabilities[keep]
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp(
            min=1e-6
        )
        evidence = F.pad(probabilities, (0, 4))
        proposals.append({
            "segments": segments[keep].detach(),
            "levels": proposal_levels[keep],
            "evidence": evidence.detach(),
            "evidence_eligible": torch.ones(
                keep.numel(), dtype=torch.bool, device=segments.device
            ),
            "teacher_scores": scores[keep].detach(),
        })
    return proposals


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
        else:
            segments = torch.zeros((0, 2), device=device, dtype=dtype)
            levels = torch.empty(0, device=device, dtype=torch.long)
            evidence = torch.zeros((0, num_classes + 4), device=device, dtype=dtype)
            eligible = torch.empty(0, device=device, dtype=torch.bool)
            teacher_scores = torch.empty(0, device=device, dtype=dtype)
        proposals.append({
            "segments": segments,
            "levels": levels,
            "evidence": evidence,
            "evidence_eligible": eligible,
            "teacher_scores": teacher_scores,
        })
    return proposals


def labels_for_proposals(proposals, gt_segments, gt_labels, num_classes, mode):
    """Return observable proposal semantics; target GT is used only by Oracle modes."""
    if mode not in ("center", "dominant", "soft"):
        raise ValueError("unknown proposal label mode")
    result = []
    for proposal, segments, labels in zip(proposals, gt_segments, gt_labels):
        targets = proposal["segments"].new_zeros(
            (proposal["segments"].shape[0], num_classes)
        )
        if segments.numel():
            for index, candidate in enumerate(proposal["segments"]):
                if mode == "center":
                    center = candidate.mean()
                    weights = ((segments[:, 0] <= center) & (center <= segments[:, 1])).float()
                else:
                    left = torch.maximum(candidate[0], segments[:, 0])
                    right = torch.minimum(candidate[1], segments[:, 1])
                    weights = (right - left).clamp(min=0)
                for class_id in range(num_classes):
                    targets[index, class_id] = weights[labels == class_id].sum()
                if mode == "dominant" and targets[index].sum() > 0:
                    top = int(targets[index].argmax())
                    targets[index].zero_()
                    targets[index, top] = 1.0
                elif targets[index].sum() > 0:
                    targets[index] /= targets[index].sum()
        result.append(targets)
    return result


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


class ProposalTargetAdapter(nn.Module):
    """Zero-initialized target residual used only on classification features."""

    def __init__(self, channels, bottleneck=64):
        super().__init__()
        self.down = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.activation = nn.ReLU()
        self.up = nn.Conv1d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(self.activation(self.down(features)))


class ProposalLogitAdapter(nn.Module):
    """Small target-only classifier residual with an exact identity start."""

    def __init__(
        self, channels, num_classes, max_delta=2.0, kernel_size=1, bias=True
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.kernel_size = int(kernel_size)
        if self.kernel_size not in (1, 3, 5):
            raise ValueError("proposal logit kernel size must be 1, 3, or 5")
        # Construct parameters directly at zero so adding this branch does not
        # consume RNG state and perturb the matched SADA continuation trajectory.
        self.weight = nn.Parameter(torch.zeros(
            num_classes, channels, self.kernel_size
        ))
        self.bias = (
            nn.Parameter(torch.zeros(num_classes)) if bias else None
        )

    def forward(self, features):
        residual = F.conv1d(
            features.detach(),
            self.weight,
            self.bias,
            padding=self.kernel_size // 2,
        )
        return self.max_delta * torch.tanh(residual)


class ProposalMLPLogitAdapter(nn.Module):
    """A small nonlinear target expert with an exactly zero initial residual."""

    def __init__(
        self,
        channels,
        num_classes,
        max_delta=2.0,
        bottleneck=64,
        kernel_size=1,
        bias=True,
    ):
        super().__init__()
        if bottleneck < 1 or kernel_size not in (1, 3, 5):
            raise ValueError("invalid proposal MLP logit adapter")
        # A fixed seed inside fork_rng gives every paired run the same expert
        # initialization without perturbing the detector/data RNG trajectory.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            self.input_proj = nn.Conv1d(channels, bottleneck, 1)
        self.output_weight = nn.Parameter(torch.zeros(
            num_classes, bottleneck, kernel_size
        ))
        self.output_bias = (
            nn.Parameter(torch.zeros(num_classes)) if bias else None
        )
        self.max_delta = float(max_delta)
        self.kernel_size = int(kernel_size)

    def forward(self, features):
        hidden = F.gelu(self.input_proj(features.detach()))
        residual = F.conv1d(
            hidden,
            self.output_weight,
            self.output_bias,
            padding=self.kernel_size // 2,
        )
        return self.max_delta * torch.tanh(residual)


def proposal_logit_gate(base_logits, mode):
    """Gate a target residual using only frozen detector confidence."""
    confidence = base_logits.detach().sigmoid().amax(dim=1, keepdim=True)
    if mode == "base_confidence":
        return confidence
    if mode == "base_ambiguity":
        return 4.0 * confidence * (1.0 - confidence)
    if mode == "none":
        return torch.ones_like(confidence)
    raise ValueError("unknown proposal logit gate")


def prepare_vlm_proposal_targets(
    class_logits,
    proposals,
    fpn_strides,
    policy="soft",
    confidence_threshold=0.5,
):
    """Attach observable VLM targets and optionally gate them with frozen TAL logits."""
    if policy not in ("soft", "confident", "agree"):
        raise ValueError("unknown VLM proposal target policy")
    selected_count = 0
    eligible_count = 0
    agreement_count = 0
    selected_confidence = []
    for batch_index, proposal in enumerate(proposals):
        targets = proposal["evidence"][:, :class_logits[0].shape[-1]].clone()
        eligible = proposal["evidence_eligible"].clone()
        confidence = (
            targets.max(dim=-1).values if targets.numel() else targets.new_zeros(0)
        )
        eligible_count += int(eligible.sum())
        if policy in ("confident", "agree"):
            eligible &= confidence >= float(confidence_threshold)
        if policy == "agree":
            agreement = torch.zeros_like(eligible)
            for index in torch.nonzero(eligible).flatten().tolist():
                segment = proposal["segments"][index]
                level = int(proposal["levels"][index])
                duration = (segment[1] - segment[0]).clamp(min=1.0)
                positions = torch.stack((
                    segment[0],
                    segment[0] + 0.25 * duration,
                    0.5 * (segment[0] + segment[1]),
                    segment[1] - 0.25 * duration,
                    segment[1],
                )) / float(fpn_strides[level])
                sampled = _sample_sequence(
                    class_logits[level][batch_index].detach().transpose(0, 1),
                    positions,
                ).transpose(0, 1)
                agreement[index] = (
                    sampled.mean(dim=0).argmax() == targets[index].argmax()
                )
            agreement_count += int((eligible & agreement).sum())
            eligible &= agreement
        targets[~eligible] = 0.0
        proposal["targets"] = targets
        selected_count += int(eligible.sum())
        if eligible.any():
            selected_confidence.append(confidence[eligible])
    if selected_confidence:
        mean_confidence = torch.cat(selected_confidence).mean()
    else:
        mean_confidence = class_logits[0].new_tensor(0.0)
    return {
        "eligible_count": eligible_count,
        "selected_count": selected_count,
        "agreement_count": agreement_count,
        "mean_confidence": mean_confidence,
    }


def proposal_class_balance_weights(
    entries,
    num_classes,
    confidence_threshold,
    power=0.5,
    max_weight=2.0,
):
    """Macro-oriented weights from observable cached VLM label frequencies."""
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    known_label_count = min(int(num_classes), len(KNOWN_LABELS))
    for entry in entries:
        label = entry.get("semantic_label")
        if label not in KNOWN_LABELS[:known_label_count]:
            continue
        probabilities = entry.get("class_probabilities")
        confidence = max(probabilities) if probabilities else 1.0
        if confidence >= float(confidence_threshold):
            counts[KNOWN_LABELS.index(label)] += 1.0
    supported = counts > 0
    weights = torch.zeros_like(counts)
    if supported.any():
        mean_count = counts[supported].mean()
        weights[supported] = (mean_count / counts[supported]).pow(float(power))
        weights.clamp_(max=float(max_weight))
        normalizer = (
            (weights[supported] * counts[supported]).sum()
            / counts[supported].sum()
        )
        weights[supported] /= normalizer.clamp(min=1e-6)
    return weights, counts


def load_proposal_class_reliability(
    path,
    num_classes,
    target_entries,
    confidence_threshold,
):
    """Load source-only class reliability and preserve target loss scale."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != "proposal-class-reliability-v1":
        raise ValueError("unsupported proposal class reliability schema")
    if payload.get("target_gt_usage") != "forbidden":
        raise ValueError("proposal reliability must forbid target GT")
    reliabilities = payload.get("smoothed_reliabilities")
    if (
        not isinstance(reliabilities, list)
        or len(reliabilities) != len(KNOWN_LABELS)
        or any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in reliabilities
        )
    ):
        raise ValueError("invalid proposal class reliabilities")
    weights = torch.zeros(int(num_classes), dtype=torch.float32)
    known_count = min(int(num_classes), len(KNOWN_LABELS))
    weights[:known_count] = torch.tensor(reliabilities[:known_count])
    target_counts = torch.zeros(int(num_classes), dtype=torch.float32)
    for entry in target_entries:
        label = entry.get("semantic_label")
        if label not in KNOWN_LABELS[:known_count]:
            continue
        probabilities = entry.get("class_probabilities")
        confidence = max(probabilities) if probabilities else 1.0
        if confidence >= float(confidence_threshold):
            target_counts[KNOWN_LABELS.index(label)] += 1.0
    normalizer = (weights * target_counts).sum() / target_counts.sum().clamp(
        min=1.0
    )
    weights[:known_count] /= normalizer.clamp(min=1e-6)
    return weights, target_counts, hashlib.sha256(raw).hexdigest(), payload


def proposal_detection_loss(
    class_logits,
    proposals,
    fpn_strides,
    loss_type="bce",
    pooling="mean",
    class_weights=None,
    base_logits=None,
):
    """Supervise the detector through proposal-average class logits."""
    if loss_type not in (
        "bce",
        "soft_ce",
        "known_soft_ce",
        "known_top1_ce",
        "positive_bce",
        "positive_bce_rank",
        "positive_bce_rank_advantage",
        "positive_bce_rank_teacher_score",
        "positive_bce_outer_anchor",
        "top2_positive_bce",
        "soft_positive_bce",
    ):
        raise ValueError("unknown proposal detection loss")
    if pooling not in ("mean", "top1", "top2", "top3"):
        raise ValueError("unknown proposal detection pooling")
    pooled_logits = []
    pooled_outer_logits = []
    pooled_outer_delta_sq = []
    pooled_base_logits = []
    pooled_teacher_scores = []
    pooled_targets = []
    for batch_index, proposal in enumerate(proposals):
        targets = proposal.get("targets")
        if targets is None:
            continue
        teacher_scores = proposal.get("teacher_scores")
        if teacher_scores is None:
            teacher_scores = targets.new_ones(targets.shape[0])
        for segment, level, target, teacher_score in zip(
            proposal["segments"], proposal["levels"], targets, teacher_scores
        ):
            if float(target.sum()) <= 0.0:
                continue
            level = int(level)
            duration = (segment[1] - segment[0]).clamp(min=1.0)
            positions = torch.stack((
                segment[0],
                segment[0] + 0.25 * duration,
                0.5 * (segment[0] + segment[1]),
                segment[1] - 0.25 * duration,
                segment[1],
            )) / float(fpn_strides[level])
            sampled = _sample_sequence(
                class_logits[level][batch_index].transpose(0, 1), positions
            ).transpose(0, 1)
            if pooling.startswith("top"):
                top_k = int(pooling[-1])
                pooled = sampled.topk(
                    k=min(top_k, sampled.shape[0]), dim=0
                ).values.mean(dim=0)
            else:
                pooled = sampled.mean(dim=0)
            pooled_logits.append(pooled)
            pooled_teacher_scores.append(teacher_score.detach())
            if loss_type in (
                "positive_bce_rank", "positive_bce_rank_advantage",
                "positive_bce_rank_teacher_score",
                "positive_bce_outer_anchor"
            ):
                outer_positions = torch.stack((
                    segment[0] - 0.50 * duration,
                    segment[0] - 0.25 * duration,
                    segment[1] + 0.25 * duration,
                    segment[1] + 0.50 * duration,
                )) / float(fpn_strides[level])
                outer_sampled = _sample_sequence(
                    class_logits[level][batch_index].transpose(0, 1),
                    outer_positions,
                ).transpose(0, 1)
                pooled_outer_logits.append(
                    outer_sampled.max(dim=0).values
                )
                if loss_type == "positive_bce_outer_anchor":
                    if base_logits is None:
                        raise ValueError(
                            "outer anchor requires frozen base logits"
                        )
                    outer_base = _sample_sequence(
                        base_logits[level][batch_index].transpose(0, 1),
                        outer_positions,
                    ).transpose(0, 1)
                    pooled_outer_delta_sq.append(
                        (outer_sampled - outer_base.detach()).pow(2).mean(dim=0)
                    )
            if loss_type == "positive_bce_rank_advantage":
                if base_logits is None:
                    raise ValueError("language advantage requires frozen base logits")
                sampled_base = _sample_sequence(
                    base_logits[level][batch_index].transpose(0, 1),
                    positions,
                ).transpose(0, 1)
                if pooling.startswith("top"):
                    top_k = int(pooling[-1])
                    pooled_base = sampled_base.topk(
                        k=min(top_k, sampled_base.shape[0]), dim=0
                    ).values.mean(dim=0)
                else:
                    pooled_base = sampled_base.mean(dim=0)
                pooled_base_logits.append(pooled_base.detach())
            pooled_targets.append(target.detach())
    if not pooled_logits:
        zero = sum(logits.sum() * 0.0 for logits in class_logits)
        return zero, 0
    logits = torch.stack(pooled_logits)
    targets = torch.stack(pooled_targets)
    if loss_type in ("known_soft_ce", "known_top1_ce"):
        known_count = min(logits.shape[-1], len(KNOWN_LABELS))
        known_logits = logits[:, :known_count]
        known_targets = targets[:, :known_count]
        if loss_type == "known_soft_ce":
            loss = -(
                known_targets.detach()
                * F.log_softmax(known_logits, dim=-1)
            ).sum(dim=-1)
        else:
            loss = F.cross_entropy(
                known_logits,
                known_targets.argmax(dim=-1),
                reduction="none",
            )
    elif loss_type == "soft_ce":
        loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    elif loss_type == "soft_positive_bce":
        loss = (F.softplus(-logits) * targets.detach()).sum(dim=-1)
    elif loss_type == "top2_positive_bce":
        top_targets, top_classes = targets.topk(
            k=min(2, targets.shape[-1]), dim=-1
        )
        top_logits = logits.gather(1, top_classes)
        loss = (
            F.softplus(-top_logits) * top_targets.detach()
        ).sum(dim=-1)
    elif loss_type in (
        "positive_bce", "positive_bce_rank",
        "positive_bce_rank_advantage", "positive_bce_rank_teacher_score",
        "positive_bce_outer_anchor"
    ):
        top_targets, top_classes = targets.max(dim=-1)
        top_logits = logits.gather(1, top_classes.unsqueeze(1)).squeeze(1)
        loss = F.softplus(-top_logits) * top_targets.detach()
        if loss_type in (
            "positive_bce_rank", "positive_bce_rank_advantage",
            "positive_bce_rank_teacher_score"
        ):
            outer_logits = torch.stack(pooled_outer_logits).gather(
                1, top_classes.unsqueeze(1)
            ).squeeze(1)
            rank_loss = F.softplus(outer_logits - top_logits)
            loss = 0.5 * (loss + rank_loss * top_targets.detach())
            if loss_type == "positive_bce_rank_advantage":
                base_top_logits = torch.stack(pooled_base_logits).gather(
                    1, top_classes.unsqueeze(1)
                ).squeeze(1)
                advantage = (
                    top_targets.detach() - base_top_logits.sigmoid()
                ).clamp(min=0.0)
                advantage = advantage / advantage.mean().clamp(min=1e-6)
                loss = loss * advantage.detach()
            elif loss_type == "positive_bce_rank_teacher_score":
                teacher_weights = torch.stack(pooled_teacher_scores)
                teacher_weights = teacher_weights / teacher_weights.mean().clamp(
                    min=1e-6
                )
                loss = loss * teacher_weights.detach()
        elif loss_type == "positive_bce_outer_anchor":
            outer_anchor = torch.stack(pooled_outer_delta_sq).gather(
                1, top_classes.unsqueeze(1)
            ).squeeze(1)
            loss = loss + outer_anchor
    else:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        ).sum(dim=-1)
    if class_weights is not None:
        top_classes = targets.argmax(dim=-1)
        weights = torch.as_tensor(
            class_weights, device=loss.device, dtype=loss.dtype
        )
        loss = loss * weights[top_classes]
    return loss.mean(), logits.shape[0]


def proposal_background_anchor_loss(
    adapted_logits,
    base_logits,
    masks,
    proposals,
    fpn_strides,
    confidence_threshold=0.05,
):
    """Keep VLM residuals zero on confident SADA background outside its bags."""
    protected = [
        torch.zeros(
            logits.shape[:2], dtype=torch.bool, device=logits.device
        )
        for logits in adapted_logits
    ]
    for batch_index, proposal in enumerate(proposals):
        targets = proposal.get("targets")
        if targets is None:
            continue
        for segment, level, target in zip(
            proposal["segments"], proposal["levels"], targets
        ):
            if float(target.sum()) <= 0.0:
                continue
            level = int(level)
            positions = torch.arange(
                adapted_logits[level].shape[1],
                device=segment.device,
                dtype=segment.dtype,
            )
            start, end = segment / float(fpn_strides[level])
            protected[level][batch_index] |= (
                (positions >= start) & (positions <= end)
            )

    total = adapted_logits[0].new_tensor(0.0)
    count = 0
    for adapted, base, mask, positive in zip(
        adapted_logits, base_logits, masks, protected
    ):
        valid = mask.squeeze(1).bool() if mask.ndim == 3 else mask.bool()
        base_confidence = base.detach().sigmoid().amax(dim=-1)
        selected = (
            valid
            & ~positive
            & (base_confidence <= float(confidence_threshold))
        )
        if selected.any():
            residual = adapted - base.detach()
            per_position = residual[selected].pow(2).mean(dim=-1)
            total = total + per_position.sum()
            count += int(per_position.numel())
    if not count:
        return sum(logits.sum() * 0.0 for logits in adapted_logits), 0
    return total / count, count


class ProposalSemanticBranch(nn.Module):
    def __init__(
        self,
        fpn_dim,
        num_classes,
        hidden_dim=256,
        embedding_dim=128,
        confidence_threshold=0.5,
        entropy_threshold=0.75,
        prototype_momentum=0.9,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.confidence_threshold = float(confidence_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.prototype_momentum = float(prototype_momentum)
        input_dim = 5 * int(fpn_dim) + self.num_classes + self.num_classes + 4
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, self.num_classes)
        self.register_buffer(
            "source_prototypes", torch.zeros(self.num_classes, embedding_dim)
        )
        self.register_buffer(
            "source_prototype_ready", torch.zeros(self.num_classes, dtype=torch.bool)
        )

    def encode(self, fpn_feats, class_logits, proposals, fpn_strides):
        rows = []
        for batch_index, proposal in enumerate(proposals):
            for segment, level, evidence in zip(
                proposal["segments"], proposal["levels"], proposal["evidence"]
            ):
                level = int(level)
                duration = (segment[1] - segment[0]).clamp(min=1.0)
                positions = torch.stack((
                    segment[0] - 0.25 * duration,
                    segment[0],
                    0.5 * (segment[0] + segment[1]),
                    segment[1],
                    segment[1] + 0.25 * duration,
                )) / float(fpn_strides[level])
                visual = _sample_sequence(
                    fpn_feats[level][batch_index], positions
                ).transpose(0, 1).reshape(-1)
                sampled_logits = _sample_sequence(
                    class_logits[level][batch_index].transpose(0, 1), positions
                ).transpose(0, 1)
                sada = sampled_logits.sigmoid().mean(dim=0)
                sada = sada / sada.sum().clamp(min=1e-6)
                if float(evidence[-1]) > 0.5:
                    evidence = evidence.clone()
                    evidence[:self.num_classes] = sada.detach()
                    evidence[-1] = 0.0
                rows.append(torch.cat((visual, sada, evidence)))
        if not rows:
            empty = fpn_feats[0].new_zeros((0, self.classifier.in_features))
            return {
                "embedding": empty,
                "logits": fpn_feats[0].new_zeros((0, self.num_classes)),
                "probabilities": fpn_feats[0].new_zeros((0, self.num_classes)),
            }
        embedding = self.encoder(torch.stack(rows))
        logits = self.classifier(embedding)
        return {
            "embedding": F.normalize(embedding, dim=-1),
            "logits": logits,
            "probabilities": logits.softmax(dim=-1),
        }

    @staticmethod
    def flatten_field(proposals, field):
        values = [proposal[field] for proposal in proposals if proposal[field].numel()]
        if not values:
            return None
        return torch.cat(values)

    def source_loss(self, encoded, proposals):
        targets = self.flatten_field(proposals, "targets")
        if targets is None or not targets.numel():
            return encoded["logits"].sum() * 0.0
        known = targets.sum(dim=-1) > 0
        if not known.any():
            return encoded["logits"].sum() * 0.0
        return -(targets[known] * encoded["logits"][known].log_softmax(dim=-1)).sum(
            dim=-1
        ).mean()

    def select_target(self, encoded, proposals):
        probabilities = encoded["probabilities"]
        evidence_eligible = self.flatten_field(proposals, "evidence_eligible")
        if evidence_eligible is None:
            evidence_eligible = torch.empty(
                0, dtype=torch.bool, device=probabilities.device
            )
        confidence = probabilities.max(dim=-1).values
        entropy = -(
            probabilities.clamp(min=1e-8).log() * probabilities
        ).sum(dim=-1) / math.log(self.num_classes)
        selected = (
            evidence_eligible
            & (confidence >= self.confidence_threshold)
            & (entropy <= self.entropy_threshold)
        )
        return {
            "q": probabilities.detach(),
            "selected": selected,
            "confidence": confidence.detach(),
            "entropy": entropy.detach(),
        }

    def prototype_alignment(self, source, source_targets, target, target_selection):
        zero = source["embedding"].sum() * 0.0 + target["embedding"].sum() * 0.0
        if source_targets is None or not source_targets.numel():
            return zero, 0
        source_embedding = source["embedding"]
        target_embedding = target["embedding"]
        selected = target_selection["selected"]
        if not selected.any():
            self._update_source_prototypes(source_embedding, source_targets)
            return zero, 0

        q = target_selection["q"][selected]
        target_embedding = target_embedding[selected]
        source_prototypes = self.source_prototypes.detach().clone()
        source_prototype_ready = self.source_prototype_ready.clone()
        losses = []
        for class_id in range(self.num_classes):
            source_weight = source_targets[:, class_id]
            target_weight = q[:, class_id]
            target_weight = target_weight * (
                target_weight > 1.0 / self.num_classes
            )
            if float(target_weight.sum()) <= 1e-6:
                continue
            if float(source_weight.sum()) > 1e-6:
                source_center = (
                    source_embedding * source_weight[:, None]
                ).sum(dim=0) / source_weight.sum()
                source_center = source_center.detach()
            elif bool(source_prototype_ready[class_id]):
                source_center = source_prototypes[class_id]
            else:
                continue
            target_center = (
                target_embedding * target_weight[:, None]
            ).sum(dim=0) / target_weight.sum()
            losses.append(F.mse_loss(target_center, source_center))
        self._update_source_prototypes(source_embedding, source_targets)
        return (torch.stack(losses).mean(), len(losses)) if losses else (zero, 0)

    @torch.no_grad()
    def _update_source_prototypes(self, embeddings, targets):
        for class_id in range(self.num_classes):
            weight = targets[:, class_id]
            if float(weight.sum()) <= 1e-6:
                continue
            center = (embeddings.detach() * weight[:, None]).sum(dim=0) / weight.sum()
            if self.source_prototype_ready[class_id]:
                center = (
                    self.prototype_momentum * self.source_prototypes[class_id]
                    + (1.0 - self.prototype_momentum) * center
                )
            self.source_prototypes[class_id].copy_(F.normalize(center, dim=0))
            self.source_prototype_ready[class_id] = True
