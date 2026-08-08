"""Source-only anchor-validity features, models, and strict artifacts."""

import hashlib
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


STRUCTURE_PROBE_SCHEMA_VERSION = "source-supervised-anchor-validity-v5"
BASE_FEATURE_NAMES = (
    "teacher_score",
    "teacher_margin",
    "teacher_entropy",
    "log_predicted_duration",
    "log_predicted_max_distance",
    "predicted_center_asymmetry",
    "level_0",
    "level_1",
    "level_2",
    "level_3",
    "level_4",
    "level_5",
)
FEATURE_GROUPS = {
    "confidence_only": BASE_FEATURE_NAMES[:3],
    "structure_pure": BASE_FEATURE_NAMES[3:],
    "reliability_full": BASE_FEATURE_NAMES,
}
MODEL_TYPES = {"logistic", "mlp", "gbdt", "fpn_mlp"}
FEATURE_NAMES = BASE_FEATURE_NAMES
TEACHER_PROVENANCE_FIELDS = (
    "routing_teacher_checkpoint_sha256",
    "routing_teacher_state_sha256",
    "routing_teacher_weights",
)
ASSIGNMENT_PROTOCOL_FIELDS = (
    "regression_range",
    "center_sample",
    "center_sample_radius",
    "reg_distance",
    "fpn_strides",
    "n_levels",
)
COHORT_PROTOCOL_FIELDS = (
    "target_class_ids",
    "action_subsampling",
    "semantic_condition",
    "confidence_condition",
    "beta",
    "alpha",
    "source_train_split",
    "source_calibration_split",
)
SAMPLING_PROTOCOL_FIELDS = (
    "max_seq_len",
    "crop_ratio",
    "trunc_thresh",
    "crop_alignment",
    "sampling_seed",
    "max_train_anchors",
    "max_videos",
    "train_crops_per_video",
    "calibration_crops_per_video",
    "stratification_policy",
)
MODEL_PROVENANCE_FIELDS = (
    "model_type",
    "feature_group",
    "evidence_family",
    "threshold",
    "model_sha256",
)
FIT_PROTOCOL_FIELDS = (
    "model_type",
    "feature_group",
    "max_iter",
    "epochs",
    "learning_rate",
    "l2",
    "gbdt_estimators",
    "gbdt_learning_rate",
    "gbdt_max_depth",
    "fit_batch_size",
    "seed",
)
VALIDATION_PARTITION_FIELDS = (
    "policy",
    "seed",
    "calibration_fraction",
    "calibration_video_ids",
    "report_video_ids",
)
PROTOCOL_FIELDS = {
    "teacher_provenance": TEACHER_PROVENANCE_FIELDS,
    "assignment_protocol": ASSIGNMENT_PROTOCOL_FIELDS,
    "cohort_protocol": COHORT_PROTOCOL_FIELDS,
    "sampling_protocol": SAMPLING_PROTOCOL_FIELDS,
    "fit_protocol": FIT_PROTOCOL_FIELDS,
    "validation_partition": VALIDATION_PARTITION_FIELDS,
    "model_provenance": MODEL_PROVENANCE_FIELDS,
}


def protocol_sha256(context):
    canonical = json.dumps(
        context, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_protocol(payload, name):
    protocol = payload.get(name)
    if not isinstance(protocol, dict):
        raise ValueError("validity artifact has no {}".format(name))
    if set(protocol) != set(PROTOCOL_FIELDS[name]):
        raise ValueError("validity {} fields differ".format(name))
    if payload.get(name + "_sha256") != protocol_sha256(protocol):
        raise ValueError("validity {} hash mismatch".format(name))
    return protocol


def validate_validity_provenance(
    payload,
    expected_teacher_provenance,
    expected_assignment_protocol,
    expected_cohort_protocol,
):
    expected = {
        "teacher_provenance": expected_teacher_provenance,
        "assignment_protocol": expected_assignment_protocol,
        "cohort_protocol": expected_cohort_protocol,
    }
    mismatches = {}
    for name in PROTOCOL_FIELDS:
        actual = _validate_protocol(payload, name)
        if name in expected and actual != expected[name]:
            mismatches[name] = (actual, expected[name])
    if mismatches:
        raise ValueError(
            "validity artifact provenance differs from this run: {}".format(
                mismatches
            )
        )


def assignment_protocol_from_model(model):
    return {
        "regression_range": [list(value) for value in model.reg_range],
        "center_sample": model.train_center_sample,
        "center_sample_radius": float(model.train_center_sample_radius),
        "reg_distance": bool(model.train_reg_distance),
        "fpn_strides": [int(value) for value in model.fpn_strides],
        "n_levels": int(model.n_levels),
    }


def cohort_protocol_from_config(config, model, beta, alpha):
    return {
        "target_class_ids": list(range(int(model.num_class_orig))),
        "action_subsampling": config["dataset"].get(
            "action_subsampling", "all"
        ),
        "semantic_condition": "center_inside_target_event_strict",
        "confidence_condition": "beta < teacher_score < alpha",
        "beta": float(beta),
        "alpha": float(alpha),
        "source_train_split": list(config["src_train_split"]),
        "source_calibration_split": list(config["src_val_split"]),
    }


def sampling_protocol_from_config(
    config,
    sampling_seed,
    max_train_anchors,
    max_videos,
    train_crops_per_video,
    calibration_crops_per_video,
):
    crop_ratio = config["dataset"].get("crop_ratio")
    return {
        "max_seq_len": int(config["dataset"]["max_seq_len"]),
        "crop_ratio": None if crop_ratio is None else list(crop_ratio),
        "trunc_thresh": float(config["dataset"]["trunc_thresh"]),
        "crop_alignment": int(config["dataset"].get("crop_alignment", 1)),
        "sampling_seed": int(sampling_seed),
        "max_train_anchors": int(max_train_anchors),
        "max_videos": None if max_videos is None else int(max_videos),
        "train_crops_per_video": int(train_crops_per_video),
        "calibration_crops_per_video": int(calibration_crops_per_video),
        "stratification_policy": "none_random_without_replacement",
    }


def partition_source_validation_videos(data_list, calibration_fraction, seed):
    """Create deterministic, video-disjoint calibration and report subsets."""
    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("source calibration fraction must be in (0, 1)")
    rows = sorted(data_list, key=lambda item: str(item["id"]))
    if len(rows) < 2:
        raise ValueError("source validation needs at least two videos")
    video_ids = [str(item["id"]) for item in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("source validation video ids must be unique")
    shuffled = list(video_ids)
    random.Random(int(seed)).shuffle(shuffled)
    calibration_count = max(
        1, min(len(rows) - 1, int(round(len(rows) * float(calibration_fraction))))
    )
    calibration_ids = set(shuffled[:calibration_count])
    calibration_rows = tuple(
        item for item in rows if str(item["id"]) in calibration_ids
    )
    report_rows = tuple(
        item for item in rows if str(item["id"]) not in calibration_ids
    )
    protocol = {
        "policy": "deterministic_video_disjoint_holdout",
        "seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "calibration_video_ids": [str(item["id"]) for item in calibration_rows],
        "report_video_ids": [str(item["id"]) for item in report_rows],
    }
    return calibration_rows, report_rows, protocol


def partition_frozen_head_source_videos(
    data_list,
    selection_videos,
    calibration_videos,
    report_videos,
    seed,
):
    """Create deterministic train/selection/calibration/report video splits."""
    rows = tuple(sorted(data_list, key=lambda item: str(item["id"])))
    holdout_count = (
        int(selection_videos) + int(calibration_videos) + int(report_videos)
    )
    if min(selection_videos, calibration_videos, report_videos) <= 0:
        raise ValueError("frozen-head holdout counts must be positive")
    if holdout_count >= len(rows):
        raise ValueError("frozen-head holdouts leave no training videos")
    shuffled_ids = [str(item["id"]) for item in rows]
    random.Random(int(seed)).shuffle(shuffled_ids)
    selection_ids = set(shuffled_ids[:selection_videos])
    calibration_start = selection_videos
    calibration_end = calibration_start + calibration_videos
    calibration_ids = set(shuffled_ids[calibration_start:calibration_end])
    report_ids = set(shuffled_ids[calibration_end:holdout_count])
    train_ids = set(shuffled_ids[holdout_count:])

    def select(ids):
        return tuple(item for item in rows if str(item["id"]) in ids)

    groups = (
        select(train_ids),
        select(selection_ids),
        select(calibration_ids),
        select(report_ids),
    )
    protocol = {
        "policy": "deterministic_video_disjoint_four_way_holdout",
        "seed": int(seed),
        "train_video_ids": [str(item["id"]) for item in groups[0]],
        "selection_video_ids": [str(item["id"]) for item in groups[1]],
        "calibration_video_ids": [str(item["id"]) for item in groups[2]],
        "report_video_ids": [str(item["id"]) for item in groups[3]],
    }
    return (*groups, protocol)


def anchor_validity_features(
    probabilities,
    left_offset,
    right_offset,
    stride,
    level,
):
    """Build the 12 observable features; no unsupervised actionness output."""
    probabilities = probabilities.float()
    top2 = probabilities.topk(k=2, dim=-1).values
    score = top2[..., 0]
    margin = top2[..., 0] - top2[..., 1]
    clipped = probabilities.clamp(1e-7, 1.0 - 1e-7)
    entropy = -(
        clipped * clipped.log()
        + (1.0 - clipped) * (1.0 - clipped).log()
    ).mean(dim=-1)

    left = left_offset.float().clamp_min(0.0) * float(stride)
    right = right_offset.float().clamp_min(0.0) * float(stride)
    duration = left + right
    max_distance = torch.maximum(left, right)
    asymmetry = (left - right).abs() / duration.clamp_min(1e-6)
    level_columns = [
        torch.full_like(score, float(level == index)) for index in range(6)
    ]
    return torch.stack(
        (
            score,
            margin,
            entropy,
            torch.log1p(duration),
            torch.log1p(max_distance),
            asymmetry,
            *level_columns,
        ),
        dim=-1,
    )


def predicted_geometry_scalars(
    offsets,
    stride,
    regression_range=None,
    center_sample_radius=None,
):
    """Return log-duration, log-max-distance and center asymmetry."""
    if offsets.ndim != 3 or offsets.shape[1] != 2:
        raise ValueError("validity offsets must have shape [B, 2, T]")
    left = offsets[:, 0].float().clamp_min(0.0) * float(stride)
    right = offsets[:, 1].float().clamp_min(0.0) * float(stride)
    duration = left + right
    max_distance = torch.maximum(left, right)
    asymmetry = (left - right).abs() / duration.clamp_min(1e-6)
    scalars = [torch.log1p(duration), torch.log1p(max_distance), asymmetry]
    if regression_range is not None or center_sample_radius is not None:
        if regression_range is None or center_sample_radius is None:
            raise ValueError("assignment geometry requires range and center radius")
        lower, upper = (float(value) for value in regression_range)
        if lower < 0.0 or upper <= lower or float(center_sample_radius) <= 0.0:
            raise ValueError("invalid assignment geometry protocol")
        center_distance = 0.5 * (left - right).abs()
        scalars.extend((
            torch.log1p(lower / max_distance.clamp_min(1e-6)),
            torch.log1p(max_distance / upper),
            torch.log1p(
                center_distance / (float(stride) * float(center_sample_radius))
            ),
        ))
    return torch.stack(scalars, dim=1)


class FPNValidityHead(nn.Module):
    """Online per-anchor validity branch attached to the current SADA FPN."""

    def __init__(
        self,
        fpn_dim,
        n_levels,
        hidden_dim=128,
        fusion_dim=64,
        level_embedding_dim=8,
        use_level_embedding=True,
        per_level_output_heads=False,
        assignment_aware_geometry=False,
        regression_ranges=None,
        center_sample_radius=None,
        detach_fpn=False,
        detach_geometry=True,
    ):
        super().__init__()
        self.n_levels = int(n_levels)
        self.use_level_embedding = bool(use_level_embedding)
        self.per_level_output_heads = bool(per_level_output_heads)
        self.assignment_aware_geometry = bool(assignment_aware_geometry)
        self.detach_fpn = bool(detach_fpn)
        self.detach_geometry = bool(detach_geometry)
        if self.assignment_aware_geometry:
            if regression_ranges is None or len(regression_ranges) != self.n_levels:
                raise ValueError("assignment geometry requires one range per level")
            if center_sample_radius is None or float(center_sample_radius) <= 0.0:
                raise ValueError("assignment geometry requires a positive center radius")
            self.regression_ranges = tuple(
                tuple(float(value) for value in bounds)
                for bounds in regression_ranges
            )
            self.center_sample_radius = float(center_sample_radius)
            geometry_dim = 6
        else:
            self.regression_ranges = None
            self.center_sample_radius = None
            geometry_dim = 3
        self.fpn_projection = nn.Sequential(
            nn.Conv1d(int(fpn_dim), int(hidden_dim), kernel_size=1),
            nn.GELU(),
        )
        if self.use_level_embedding:
            self.level_embedding = nn.Embedding(
                self.n_levels, int(level_embedding_dim)
            )
            output_dim = int(hidden_dim) + geometry_dim + int(level_embedding_dim)
        else:
            self.level_embedding = None
            output_dim = int(hidden_dim) + geometry_dim

        def make_output():
            return nn.Sequential(
                nn.Conv1d(output_dim, int(fusion_dim), kernel_size=1),
                nn.GELU(),
                nn.Conv1d(int(fusion_dim), 1, kernel_size=1),
            )

        if self.per_level_output_heads:
            self.output_heads = nn.ModuleList(
                make_output() for _ in range(self.n_levels)
            )
            self.output = None
        else:
            self.output_heads = None
            self.output = make_output()

    def forward(self, fpn_features, offsets, masks, strides):
        if not (
            len(fpn_features)
            == len(offsets)
            == len(masks)
            == len(strides)
            == self.n_levels
        ):
            raise ValueError("validity head inputs have inconsistent FPN levels")
        logits = []
        for level, (features, level_offsets, mask, stride) in enumerate(
            zip(fpn_features, offsets, masks, strides)
        ):
            projected = self.fpn_projection(
                features.detach() if self.detach_fpn else features
            )
            geometry = predicted_geometry_scalars(
                level_offsets.detach() if self.detach_geometry else level_offsets,
                stride,
                self.regression_ranges[level]
                if self.assignment_aware_geometry else None,
                self.center_sample_radius,
            )
            inputs = [projected, geometry]
            if self.level_embedding is not None:
                embedding = self.level_embedding.weight[level].view(1, -1, 1)
                embedding = embedding.expand(
                    features.shape[0], -1, features.shape[-1]
                )
                inputs.append(embedding)
            output = (
                self.output_heads[level]
                if self.output_heads is not None
                else self.output
            )
            level_logits = output(torch.cat(inputs, dim=1)).squeeze(1)
            level_mask = mask.squeeze(1).bool() if mask.ndim == 3 else mask.bool()
            logits.append(level_logits.masked_fill(~level_mask, 0.0))
        return tuple(logits)


def freeze_sada_except_validity_head(model):
    """Freeze the complete SADA model and leave only its Validity Head trainable."""
    validity_head = getattr(model, "validity_head", None)
    if validity_head is None:
        raise ValueError("frozen-SADA training requires an enabled Validity Head")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in validity_head.parameters():
        parameter.requires_grad_(True)
    return {
        "frozen_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "trainable_head_parameters": sum(
            parameter.numel() for parameter in validity_head.parameters()
        ),
    }


def clip_grad_norm_with_isolated_validity_head(model, max_norm):
    """Clip base-model and Validity Head gradients without norm coupling."""
    validity_head = getattr(model, "validity_head", None)
    if validity_head is None:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm), None
    head_parameters = tuple(validity_head.parameters())
    head_parameter_ids = {id(parameter) for parameter in head_parameters}
    base_parameters = tuple(
        parameter
        for parameter in model.parameters()
        if id(parameter) not in head_parameter_ids
    )
    base_norm = torch.nn.utils.clip_grad_norm_(base_parameters, max_norm)
    head_norm = torch.nn.utils.clip_grad_norm_(head_parameters, max_norm)
    return base_norm, head_norm


def source_validity_loss(
    logits,
    labels,
    cohorts,
    masks,
    level_balanced=False,
):
    """Balanced source-only BCE over anchors centered inside source events."""
    selected_logits = []
    selected_labels = []
    per_level = []
    level_losses = []
    for level_logits, level_labels, cohort, mask in zip(
        logits, labels, cohorts, masks
    ):
        level_mask = mask.squeeze(1).bool() if mask.ndim == 3 else mask.bool()
        selected = cohort.bool() & level_mask
        if selected.any():
            current_logits = level_logits[selected]
            current_labels = level_labels[selected].float()
            selected_logits.append(current_logits)
            selected_labels.append(current_labels)
            positive = int(current_labels.sum().item())
            negative = int(current_labels.numel() - positive)
            if level_balanced and positive > 0 and negative > 0:
                positive_loss = F.softplus(
                    -current_logits[current_labels.bool()]
                ).mean()
                negative_loss = F.softplus(
                    current_logits[~current_labels.bool()]
                ).mean()
                level_losses.append(0.5 * (positive_loss + negative_loss))
        else:
            positive = 0
            negative = 0
        per_level.append({
            "count": positive + negative,
            "positive": positive,
            "negative": negative,
        })
    if not selected_logits:
        zero = sum(value.sum() for value in logits) * 0.0
        return zero, {
            "count": 0,
            "positive": 0,
            "negative": 0,
            "active_levels": 0,
            "per_level": per_level,
        }

    flat_logits = torch.cat(selected_logits)
    flat_labels = torch.cat(selected_labels)
    positive = int(flat_labels.sum().item())
    negative = int(flat_labels.numel() - positive)
    if level_balanced:
        if level_losses:
            loss = torch.stack(level_losses).mean()
        else:
            loss = sum(value.sum() for value in logits) * 0.0
    else:
        if positive > 0 and negative > 0:
            pos_weight = flat_labels.new_tensor(
                float(negative) / float(positive)
            )
        else:
            pos_weight = flat_labels.new_tensor(1.0)
        loss = F.binary_cross_entropy_with_logits(
            flat_logits, flat_labels, pos_weight=pos_weight
        )
    return loss, {
        "count": int(flat_labels.numel()),
        "positive": positive,
        "negative": negative,
        "active_levels": len(level_losses) if level_balanced else len(selected_logits),
        "per_level": per_level,
    }


def balanced_probability_loss(scores, labels, levels=None):
    if levels is not None:
        level_losses = []
        for level in sorted(levels.unique().tolist()):
            mask = levels == level
            level_labels = labels[mask]
            if level_labels.any() and (~level_labels.bool()).any():
                level_losses.append(
                    balanced_probability_loss(scores[mask], level_labels)
                )
        if not level_losses:
            raise ValueError("selection split has no level with both label classes")
        return sum(level_losses) / len(level_losses)
    labels = labels.float()
    positives = labels.sum().clamp_min(1.0)
    negatives = (1.0 - labels).sum().clamp_min(1.0)
    positive_weight = negatives / positives
    scores = scores.clamp(1e-6, 1.0 - 1e-6)
    return F.binary_cross_entropy(
        scores,
        labels,
        weight=torch.where(labels.bool(), positive_weight, 1.0),
    ).item()


def feature_indices(feature_group):
    if feature_group not in FEATURE_GROUPS:
        raise ValueError("unsupported validity feature group: {}".format(feature_group))
    wanted = FEATURE_GROUPS[feature_group]
    return [BASE_FEATURE_NAMES.index(name) for name in wanted]


def select_feature_group(features, feature_group):
    return features[..., feature_indices(feature_group)]


def _check_binary_labels(labels):
    labels = labels.float()
    if labels.numel() == 0 or labels.min() == labels.max():
        raise ValueError("validity model requires both valid and invalid anchors")
    return labels


def fit_logistic_probe(features, labels, max_iter=50, l2=0.01, feature_group="reliability_full"):
    labels = _check_binary_labels(labels)
    selected = select_feature_group(features, feature_group)
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = ((selected - mean) / std).double()
    target = labels.double()
    weight = torch.zeros(normalized.shape[1], dtype=torch.double, requires_grad=True)
    bias = torch.zeros((), dtype=torch.double, requires_grad=True)
    positives = target.sum().clamp_min(1.0)
    negatives = (1.0 - target).sum().clamp_min(1.0)
    optimizer = torch.optim.LBFGS(
        (weight, bias), max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        logits = normalized.matmul(weight) + bias
        loss = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=negatives / positives
        ) + float(l2) * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return {
        "model_type": "logistic",
        "feature_group": feature_group,
        "mean": mean.float(),
        "std": std.float(),
        "weight": weight.detach().float(),
        "bias": float(bias.detach()),
    }


class _SmallMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)


class _FPNValidityMLP(nn.Module):
    def __init__(self, fpn_dim, level_embedding_dim=8):
        super().__init__()
        self.fpn_projection = nn.Sequential(nn.Linear(fpn_dim, 128), nn.GELU())
        self.level_embedding = nn.Embedding(6, level_embedding_dim)
        self.output = nn.Sequential(
            nn.Linear(128 + 3 + level_embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, fpn_features, structural_scalars, levels):
        hidden = self.fpn_projection(fpn_features)
        joined = torch.cat(
            (hidden, structural_scalars, self.level_embedding(levels.long())), dim=-1
        )
        return self.output(joined).squeeze(-1)


def _weighted_bce(logits, labels, pos_weight=None):
    labels = labels.float()
    if pos_weight is None:
        positives = labels.sum().clamp_min(1.0)
        negatives = (1.0 - labels).sum().clamp_min(1.0)
        pos_weight = negatives / positives
    return F.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=pos_weight
    )


def _fit_torch_model(
    model,
    train_args,
    train_labels,
    epochs,
    learning_rate,
    l2,
    seed,
    device=None,
    batch_size=None,
):
    torch.manual_seed(seed)
    device = torch.device(device or train_labels.device)
    model = model.to(device)
    train_args = tuple(value.to(device) for value in train_args)
    train_labels = train_labels.to(device)
    count = len(train_labels)
    batch_size = count if batch_size is None else min(int(batch_size), count)
    if batch_size <= 0:
        raise ValueError("fit batch size must be positive")
    positives = train_labels.sum().clamp_min(1.0)
    negatives = (1.0 - train_labels).sum().clamp_min(1.0)
    pos_weight = negatives / positives
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2)
    model.train()
    for _ in range(int(epochs)):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, batch_size):
            indices = order[start:start + batch_size].to(device)
            optimizer.zero_grad()
            logits = model(*(value[indices] for value in train_args))
            loss = _weighted_bce(
                logits, train_labels[indices], pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
    model.eval()
    return model.cpu()


def _linear_payload(linear):
    return {
        "weight": linear.weight.detach().cpu(),
        "bias": linear.bias.detach().cpu(),
    }


def fit_mlp_probe(
    features,
    labels,
    feature_group="reliability_full",
    epochs=200,
    learning_rate=1e-3,
    l2=1e-4,
    seed=20260808,
    device=None,
    batch_size=None,
):
    labels = _check_binary_labels(labels)
    torch.manual_seed(seed)
    selected = select_feature_group(features, feature_group).float()
    mean = selected.mean(dim=0)
    std = selected.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (selected - mean) / std
    model = _fit_torch_model(
        _SmallMLP(normalized.shape[-1]),
        (normalized,),
        labels,
        epochs,
        learning_rate,
        l2,
        seed,
        device,
        batch_size,
    )
    linear_layers = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    return {
        "model_type": "mlp",
        "feature_group": feature_group,
        "mean": mean,
        "std": std,
        "layers": [_linear_payload(layer) for layer in linear_layers],
    }


def fit_gbdt_probe(
    features,
    labels,
    feature_group="reliability_full",
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    seed=20260808,
):
    from sklearn.ensemble import GradientBoostingClassifier

    labels = _check_binary_labels(labels)
    selected = select_feature_group(features, feature_group).cpu().numpy()
    target = labels.cpu().numpy().astype("int64")
    positives = max(int(target.sum()), 1)
    negatives = max(int((1 - target).sum()), 1)
    sample_weight = target * (len(target) / (2.0 * positives)) + (
        1 - target
    ) * (len(target) / (2.0 * negatives))
    model = GradientBoostingClassifier(
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        random_state=int(seed),
    )
    model.fit(selected, target, sample_weight=sample_weight)
    prior = float(model.init_.class_prior_[1])
    trees = []
    for estimator in model.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append({
            "children_left": tree.children_left.tolist(),
            "children_right": tree.children_right.tolist(),
            "feature": tree.feature.tolist(),
            "threshold": tree.threshold.tolist(),
            "value": tree.value[:, 0, 0].tolist(),
        })
    return {
        "model_type": "gbdt",
        "feature_group": feature_group,
        "initial_log_odds": math.log(prior / max(1.0 - prior, 1e-12)),
        "learning_rate": float(model.learning_rate),
        "trees": trees,
    }


def fit_fpn_mlp_probe(
    features,
    fpn_features,
    labels,
    epochs=200,
    learning_rate=1e-3,
    l2=1e-4,
    seed=20260808,
    device=None,
    batch_size=None,
):
    labels = _check_binary_labels(labels)
    torch.manual_seed(seed)
    fpn_features = fpn_features.float()
    structural = features[:, 3:6].float()
    levels = features[:, 6:12].argmax(dim=-1)
    fpn_mean = fpn_features.mean(dim=0)
    fpn_std = fpn_features.std(dim=0, unbiased=False).clamp_min(1e-6)
    structural_mean = structural.mean(dim=0)
    structural_std = structural.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized_fpn = (fpn_features - fpn_mean) / fpn_std
    normalized_structural = (structural - structural_mean) / structural_std
    model = _fit_torch_model(
        _FPNValidityMLP(fpn_features.shape[-1]),
        (normalized_fpn, normalized_structural, levels),
        labels,
        epochs,
        learning_rate,
        l2,
        seed,
        device,
        batch_size,
    )
    return {
        "model_type": "fpn_mlp",
        "feature_group": "structure_pure",
        "fpn_dim": int(fpn_features.shape[-1]),
        "fpn_mean": fpn_mean,
        "fpn_std": fpn_std,
        "structural_mean": structural_mean,
        "structural_std": structural_std,
        "fpn_projection": _linear_payload(model.fpn_projection[0]),
        "level_embedding": model.level_embedding.weight.detach().cpu(),
        "output_layers": [
            _linear_payload(layer) for layer in model.output if isinstance(layer, nn.Linear)
        ],
    }


def _linear(features, payload):
    return F.linear(features, payload["weight"], payload["bias"])


def _gbdt_logits(features, probe):
    logits = torch.full(
        features.shape[:-1],
        float(probe["initial_log_odds"]),
        dtype=features.dtype,
        device=features.device,
    )
    flat = features.reshape(-1, features.shape[-1])
    flat_logits = logits.reshape(-1)
    for tree in probe["trees"]:
        left = torch.as_tensor(
            tree["children_left"], dtype=torch.long, device=features.device
        )
        right = torch.as_tensor(
            tree["children_right"], dtype=torch.long, device=features.device
        )
        feature = torch.as_tensor(
            tree["feature"], dtype=torch.long, device=features.device
        )
        threshold = torch.as_tensor(
            tree["threshold"], dtype=features.dtype, device=features.device
        )
        value = torch.as_tensor(
            tree["value"], dtype=features.dtype, device=features.device
        )
        node = torch.zeros(flat.shape[0], dtype=torch.long, device=features.device)
        while bool((left[node] >= 0).any()):
            active = left[node] >= 0
            active_rows = torch.nonzero(active, as_tuple=False).flatten()
            active_nodes = node[active]
            columns = feature[active_nodes]
            go_left = flat[active_rows, columns] <= threshold[active_nodes]
            node[active_rows] = torch.where(
                go_left, left[active_nodes], right[active_nodes]
            )
        flat_logits += float(probe["learning_rate"]) * value[node]
    return logits


def probe_probabilities(features, probe, fpn_features=None):
    feature_group = probe["feature_group"]
    model_type = probe["model_type"]
    if model_type == "fpn_mlp":
        if fpn_features is None:
            raise ValueError("fpn_mlp validity model requires FPN features")
        fpn = (fpn_features - probe["fpn_mean"]) / probe["fpn_std"]
        structural = (features[..., 3:6] - probe["structural_mean"]) / probe[
            "structural_std"
        ]
        levels = features[..., 6:12].argmax(dim=-1)
        hidden = F.gelu(_linear(fpn, probe["fpn_projection"]))
        embedded = F.embedding(levels, probe["level_embedding"])
        hidden = torch.cat((hidden, structural, embedded), dim=-1)
        hidden = F.gelu(_linear(hidden, probe["output_layers"][0]))
        return torch.sigmoid(_linear(hidden, probe["output_layers"][1]).squeeze(-1))

    selected = select_feature_group(features, feature_group)
    if model_type == "logistic":
        normalized = (selected - probe["mean"]) / probe["std"]
        logits = normalized.matmul(probe["weight"]) + probe["bias"]
    elif model_type == "mlp":
        hidden = (selected - probe["mean"]) / probe["std"]
        hidden = F.gelu(_linear(hidden, probe["layers"][0]))
        hidden = F.gelu(_linear(hidden, probe["layers"][1]))
        logits = _linear(hidden, probe["layers"][2]).squeeze(-1)
    elif model_type == "gbdt":
        logits = _gbdt_logits(selected, probe)
    else:
        raise ValueError("unsupported validity model type: {}".format(model_type))
    return torch.sigmoid(logits)


def high_precision_threshold(scores, labels, minimum_precision):
    """Select the source-val threshold with maximum recall at required precision."""
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    labels = labels.bool()
    candidates = torch.unique(scores).sort(descending=True).values
    best = None
    observed = None
    positives = int(labels.sum())
    for threshold in candidates:
        selected = scores >= threshold
        true_positive = int((selected & labels).sum())
        count = int(selected.sum())
        precision = true_positive / max(count, 1)
        recall = true_positive / max(positives, 1)
        observation = (precision, count, float(threshold), recall)
        if observed is None or observation[:2] > observed[:2]:
            observed = observation
        if precision + 1e-12 < minimum_precision:
            continue
        candidate = (recall, count, float(threshold), precision)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        precision, count, threshold, recall = observed
        raise ValueError(
            "source val cannot satisfy minimum validity precision; "
            "maximum observed precision={:.6f}, selected={}, threshold={:.6f}, "
            "recall={:.6f}".format(precision, count, threshold, recall)
        )
    recall, count, threshold, precision = best
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "selected": count,
    }


def video_robust_high_precision_threshold(
    scores,
    labels,
    video_ids,
    minimum_precision,
    minimum_selected_per_video,
):
    """Calibrate one threshold that satisfies every source calibration video."""
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    if minimum_selected_per_video <= 0:
        raise ValueError("minimum_selected_per_video must be positive")
    if len(scores) != len(labels) or len(scores) != len(video_ids):
        raise ValueError("scores, labels, and video_ids must have equal length")
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().bool()
    ordered_videos = tuple(sorted(set(str(value) for value in video_ids)))
    if not ordered_videos:
        raise ValueError("video-robust calibration requires at least one video")
    video_index = {video_id: index for index, video_id in enumerate(ordered_videos)}
    encoded_videos = torch.tensor(
        [video_index[str(value)] for value in video_ids], dtype=torch.long
    )
    order = scores.argsort(descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_videos = encoded_videos[order]
    boundary_mask = torch.ones(len(scores), dtype=torch.bool)
    boundary_mask[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    boundaries = boundary_mask.nonzero(as_tuple=False).squeeze(1)
    cumulative_selected = []
    cumulative_true_positive = []
    for index in range(len(ordered_videos)):
        video_mask = sorted_videos == index
        cumulative_selected.append(video_mask.cumsum(0))
        cumulative_true_positive.append(
            (video_mask & sorted_labels).cumsum(0)
        )
    cumulative_selected = torch.stack(cumulative_selected)
    cumulative_true_positive = torch.stack(cumulative_true_positive)
    total_true_positive = sorted_labels.cumsum(0)
    positives = int(labels.sum())
    best = None
    observed = None
    for boundary in boundaries.tolist():
        selected_per_video = cumulative_selected[:, boundary]
        if int(selected_per_video.min()) < minimum_selected_per_video:
            continue
        true_positive_per_video = cumulative_true_positive[:, boundary]
        precision_per_video = (
            true_positive_per_video.float() / selected_per_video.float()
        )
        per_video = {
            video_id: {
                "selected": int(selected_per_video[index]),
                "precision": float(precision_per_video[index]),
            }
            for index, video_id in enumerate(ordered_videos)
        }
        total_selected = boundary + 1
        true_positive = int(total_true_positive[boundary])
        recall = true_positive / max(positives, 1)
        worst_precision = float(precision_per_video.min())
        threshold = float(sorted_scores[boundary])
        observation = (
            worst_precision,
            recall,
            total_selected,
            float(threshold),
            per_video,
        )
        if observed is None or observation[:3] > observed[:3]:
            observed = observation
        if worst_precision + 1e-12 < minimum_precision:
            continue
        candidate = (
            recall,
            total_selected,
            float(threshold),
            worst_precision,
            per_video,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if observed is None:
        raise ValueError(
            "no threshold selects the minimum number of anchors for every video"
        )
    if best is None:
        worst_precision, recall, count, threshold, per_video = observed
        return False, {
            "threshold": threshold,
            "worst_video_precision": worst_precision,
            "recall": recall,
            "selected": count,
            "per_video": per_video,
        }
    recall, count, threshold, worst_precision, per_video = best
    return True, {
        "threshold": threshold,
        "worst_video_precision": worst_precision,
        "recall": recall,
        "selected": count,
        "per_video": per_video,
    }


def _json_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


def serialize_probe(probe, threshold, metadata):
    evidence_family = (
        "representation_validity"
        if probe["model_type"] == "fpn_mlp"
        else probe["feature_group"]
    )
    serialized_model = _json_value(probe)
    model_provenance = {
        "model_type": probe["model_type"],
        "feature_group": probe["feature_group"],
        "evidence_family": evidence_family,
        "threshold": float(threshold),
        "model_sha256": protocol_sha256(serialized_model),
    }
    return {
        "schema_version": STRUCTURE_PROBE_SCHEMA_VERSION,
        "model_type": probe["model_type"],
        "feature_group": probe["feature_group"],
        "evidence_family": evidence_family,
        "feature_names": list(FEATURE_GROUPS[probe["feature_group"]]),
        "base_feature_names": list(BASE_FEATURE_NAMES),
        "threshold": float(threshold),
        "model": serialized_model,
        "model_provenance": model_provenance,
        "model_provenance_sha256": protocol_sha256(model_provenance),
        **metadata,
    }


def _tensorize_model(value):
    if isinstance(value, dict):
        return {key: _tensorize_model(item) for key, item in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(item, (int, float)) for item in value):
            return torch.tensor(value, dtype=torch.float32)
        if value and all(isinstance(item, list) for item in value):
            try:
                return torch.tensor(value, dtype=torch.float32)
            except (TypeError, ValueError):
                pass
        return [_tensorize_model(item) for item in value]
    return value


def load_validity_model(path):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != STRUCTURE_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported validity model schema")
    model_type = payload.get("model_type")
    feature_group = payload.get("feature_group")
    if model_type not in MODEL_TYPES:
        raise ValueError("unsupported validity model type")
    if feature_group not in FEATURE_GROUPS:
        raise ValueError("unsupported validity feature group")
    expected_family = (
        "representation_validity"
        if model_type == "fpn_mlp"
        else feature_group
    )
    if payload.get("evidence_family") != expected_family:
        raise ValueError("validity evidence family differs")
    if tuple(payload.get("base_feature_names", ())) != BASE_FEATURE_NAMES:
        raise ValueError("validity base feature schema differs")
    if tuple(payload.get("feature_names", ())) != FEATURE_GROUPS[feature_group]:
        raise ValueError("validity selected feature schema differs")
    if payload.get("target_gt_usage") != "forbidden":
        raise ValueError("validity model does not forbid target GT")
    if payload.get("supervision") != (
        "source_gt_official_label_points_conditioned_on_target_event"
    ):
        raise ValueError("validity model supervision is not source-only")
    threshold = float(payload["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid validity threshold")
    for protocol_name in PROTOCOL_FIELDS:
        _validate_protocol(payload, protocol_name)
    model_provenance = payload["model_provenance"]
    if (
        model_provenance["model_type"] != model_type
        or model_provenance["feature_group"] != feature_group
        or model_provenance["evidence_family"] != expected_family
        or model_provenance["threshold"] != threshold
        or model_provenance["model_sha256"]
        != protocol_sha256(payload["model"])
    ):
        raise ValueError("validity model provenance differs from serialized model")
    model = _tensorize_model(payload["model"])
    model["model_type"] = model_type
    model["feature_group"] = feature_group
    return payload, model, hashlib.sha256(raw).hexdigest()


def validity_model_to(model, device):
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        if isinstance(value, list):
            return [move(item) for item in value]
        return value
    return move(model)


load_structure_probe = load_validity_model
