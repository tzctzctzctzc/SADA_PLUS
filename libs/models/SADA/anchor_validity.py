"""Source-supervised anchor-validity features and linear probe utilities."""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


STRUCTURE_PROBE_SCHEMA_VERSION = "source-supervised-anchor-validity-v1"
FEATURE_NAMES = (
    "teacher_score",
    "teacher_margin",
    "teacher_entropy",
    "teacher_actionness",
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


def anchor_validity_features(
    probabilities,
    actionness,
    left_offset,
    right_offset,
    stride,
    level,
):
    """Build features observable on both source and unlabeled target anchors."""
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
            actionness.float(),
            torch.log1p(duration),
            torch.log1p(max_distance),
            asymmetry,
            *level_columns,
        ),
        dim=-1,
    )


def fit_logistic_probe(features, labels, max_iter=50, l2=0.01):
    labels = labels.float()
    if labels.numel() == 0 or labels.min() == labels.max():
        raise ValueError("structure probe requires both valid and invalid anchors")
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = ((features - mean) / std).double()
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
        "mean": mean.float(),
        "std": std.float(),
        "weight": weight.detach().float(),
        "bias": float(bias.detach()),
    }


def probe_probabilities(features, probe):
    normalized = (features - probe["mean"]) / probe["std"]
    return torch.sigmoid(normalized.matmul(probe["weight"]) + probe["bias"])


def high_precision_threshold(scores, labels, minimum_precision):
    """Select the source-val threshold with maximum recall at required precision."""
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    labels = labels.bool()
    candidates = torch.unique(scores).sort(descending=True).values
    best = None
    positives = int(labels.sum())
    for threshold in candidates:
        selected = scores >= threshold
        true_positive = int((selected & labels).sum())
        count = int(selected.sum())
        precision = true_positive / max(count, 1)
        recall = true_positive / max(positives, 1)
        if precision + 1e-12 < minimum_precision:
            continue
        candidate = (recall, count, float(threshold), precision)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise ValueError("source val cannot satisfy minimum structure precision")
    recall, count, threshold, precision = best
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "selected": count,
    }


def serialize_probe(probe, threshold, metadata):
    return {
        "schema_version": STRUCTURE_PROBE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "mean": probe["mean"].tolist(),
        "std": probe["std"].tolist(),
        "weight": probe["weight"].tolist(),
        "bias": probe["bias"],
        "threshold": float(threshold),
        **metadata,
    }


def load_structure_probe(path):
    path = Path(path)
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("schema_version") != STRUCTURE_PROBE_SCHEMA_VERSION:
        raise ValueError("unsupported structure probe schema")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("structure probe feature schema differs")
    if payload.get("target_gt_usage") != "forbidden":
        raise ValueError("structure probe does not forbid target GT")
    if payload.get("supervision") != (
        "source_gt_official_label_points_conditioned_on_target_event"
    ):
        raise ValueError("structure probe supervision is not source-only")
    size = len(FEATURE_NAMES)
    if any(len(payload.get(key, ())) != size for key in ("mean", "std", "weight")):
        raise ValueError("invalid structure probe dimensions")
    threshold = float(payload["threshold"])
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("invalid structure probe threshold")
    probe = {
        "mean": torch.tensor(payload["mean"], dtype=torch.float32),
        "std": torch.tensor(payload["std"], dtype=torch.float32),
        "weight": torch.tensor(payload["weight"], dtype=torch.float32),
        "bias": float(payload["bias"]),
    }
    return payload, probe, hashlib.sha256(raw).hexdigest()
