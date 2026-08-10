"""Shared utilities for the offline VLM boundary audit."""

import hashlib
import json
from pathlib import Path

import torch


PROPOSAL_SCHEMA_VERSION = "vlm-boundary-proposal-v1"
QUERY_SCHEMA_VERSION = "vlm-boundary-query-v1"
PREDICTION_SCHEMA_VERSION = "vlm-boundary-prediction-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def json_sha256(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def temporal_iou(segment, target):
    left = max(float(segment[0]), float(target[0]))
    right = min(float(segment[1]), float(target[1]))
    intersection = max(0.0, right - left)
    union = max(0.0, float(segment[1]) - float(segment[0]))
    union += max(0.0, float(target[1]) - float(target[0]))
    union -= intersection
    return intersection / max(union, 1e-12)


def pairwise_temporal_iou(segments, reference):
    """IoU between ``N x 2`` segments and one reference segment."""
    if segments.ndim != 2 or segments.shape[-1] != 2:
        raise ValueError("segments must have shape [N, 2]")
    reference = torch.as_tensor(reference, dtype=segments.dtype, device=segments.device)
    left = torch.maximum(segments[:, 0], reference[0])
    right = torch.minimum(segments[:, 1], reference[1])
    intersection = (right - left).clamp_min(0)
    union = (segments[:, 1] - segments[:, 0]).clamp_min(0)
    union = union + (reference[1] - reference[0]).clamp_min(0) - intersection
    return intersection / union.clamp_min(1e-12)


def hard_class_agnostic_nms_indices(segments, scores, iou_threshold, max_count):
    """Deterministic hard NMS that preserves the exact originating indices."""
    if segments.ndim != 2 or segments.shape[-1] != 2:
        raise ValueError("segments must have shape [N, 2]")
    if scores.ndim != 1 or scores.shape[0] != segments.shape[0]:
        raise ValueError("scores must align with segments")
    if not 0 <= float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be in [0, 1]")
    if int(max_count) <= 0:
        raise ValueError("max_count must be positive")

    order = sorted(
        range(scores.numel()),
        key=lambda index: (-float(scores[index]), index),
    )
    kept = []
    while order and len(kept) < int(max_count):
        current = order.pop(0)
        kept.append(current)
        if not order:
            break
        remaining = torch.as_tensor(order, dtype=torch.long, device=segments.device)
        overlaps = pairwise_temporal_iou(segments[remaining], segments[current])
        order = [
            index
            for index, overlap in zip(order, overlaps.tolist())
            if overlap <= float(iou_threshold)
        ]
    return torch.as_tensor(kept, dtype=torch.long, device=segments.device)


def grid_to_seconds(value, feat_stride, num_frames, fps):
    return (float(value) * float(feat_stride) + 0.5 * float(num_frames)) / float(fps)


def clipped_segment(segment, duration):
    left = min(max(float(segment[0]), 0.0), float(duration))
    right = min(max(float(segment[1]), 0.0), float(duration))
    return [left, right]


def fine_step_seconds(origin_stride_seconds, maximum=0.3, minimum=0.1):
    return max(float(minimum), min(float(maximum), float(origin_stride_seconds) / 2.0))


def centered_candidates(center, step, count, duration):
    if count < 3 or count % 2 == 0:
        raise ValueError("candidate count must be an odd integer >= 3")
    radius = count // 2
    values = []
    for offset in range(-radius, radius + 1):
        value = min(max(float(center) + offset * float(step), 0.0), float(duration))
        value = round(value, 6)
        if not values or abs(value - values[-1]) > 1e-9:
            values.append(value)
    return values


def valid_segment(left, right, minimum_duration=1e-6):
    return float(right) - float(left) >= float(minimum_duration)
