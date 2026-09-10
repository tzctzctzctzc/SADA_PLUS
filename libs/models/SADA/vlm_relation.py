"""Target-only cross-proposal relation distillation from frozen VLM features."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


CACHE_SCHEMA = "vlm-cross-proposal-relation-cache-v1"


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vlm_relation_cache(path):
    path = Path(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text(encoding="utf-8"))
    if (
        meta.get("schema_version") != CACHE_SCHEMA
        or meta.get("domain") != "target"
        or meta.get("split") != "train"
        or meta.get("target_gt_usage") != "forbidden"
        or meta.get("selection_uses_gt") is not False
    ):
        raise ValueError("invalid VLM relation cache provenance")
    arrays = np.load(path, allow_pickle=False)
    required = {
        "candidate_ids", "video_ids", "starts", "ends", "centers", "levels",
        "teacher_scores", "static_features", "transition_features",
    }
    if set(arrays.files) != required:
        raise ValueError("VLM relation cache fields differ")
    count = len(arrays["candidate_ids"])
    if count != int(meta.get("entry_count", -1)) or count == 0:
        raise ValueError("VLM relation cache count differs from metadata")
    if any(len(arrays[name]) != count for name in required):
        raise ValueError("VLM relation cache arrays have inconsistent lengths")
    static = arrays["static_features"].astype(np.float32)
    transition = arrays["transition_features"].astype(np.float32)
    calibration = meta.get("relation_calibration", {})
    distance_scale = float(calibration.get("distance_scale", 0.0))
    calibration_max_gap = float(meta.get("calibration_max_gap_seconds", 0.0))
    if (
        static.ndim != 2
        or transition.shape != static.shape
        or static.shape[1] != int(meta.get("feature_dim", -1))
        or not np.isfinite(static).all()
        or not np.isfinite(transition).all()
        or not math.isfinite(distance_scale)
        or distance_scale <= 1e-6
        or not math.isfinite(calibration_max_gap)
        or calibration_max_gap <= 0.0
    ):
        raise ValueError("VLM relation features are invalid")
    by_video = {}
    for index in range(count):
        start = float(arrays["starts"][index])
        end = float(arrays["ends"][index])
        center = float(arrays["centers"][index])
        level = int(arrays["levels"][index])
        score = float(arrays["teacher_scores"][index])
        if (
            not all(math.isfinite(value) for value in (start, end, center, score))
            or start < 0.0
            or end <= start
            or not start <= center <= end
            or level < 0
        ):
            raise ValueError("VLM relation cache contains an invalid proposal")
        video_id = str(arrays["video_ids"][index])
        by_video.setdefault(video_id, []).append({
            "candidate_id": str(arrays["candidate_ids"][index]),
            "proposal_start_sec": start,
            "proposal_end_sec": end,
            "absolute_center_time": center,
            "source_fpn_level": level,
            "teacher_score": score,
            "static_feature": torch.from_numpy(static[index].copy()),
            "transition_feature": torch.from_numpy(transition[index].copy()),
        })
    for entries in by_video.values():
        entries.sort(key=lambda entry: entry["absolute_center_time"])
    return by_video, count, _sha256_file(path), meta


def relation_proposals_for_videos(cache, videos, n_levels, device, dtype, levels):
    allowed_levels = set(int(level) for level in levels)
    output = []
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
            level = entry["source_fpn_level"]
            if level >= n_levels or level not in allowed_levels:
                continue
            start = (entry["proposal_start_sec"] * fps - frame_offset) / stride
            end = (entry["proposal_end_sec"] * fps - frame_offset) / stride
            start -= crop_start
            end -= crop_start
            if end <= 0.0 or start >= crop_length:
                continue
            rows.append((
                max(start, 0.0), min(end, crop_length), level,
                entry["absolute_center_time"], entry["teacher_score"],
                entry["static_feature"], entry["transition_feature"],
            ))
        if rows:
            output.append({
                "segments": torch.tensor(
                    [[row[0], row[1]] for row in rows], device=device, dtype=dtype
                ),
                "levels": torch.tensor(
                    [row[2] for row in rows], device=device, dtype=torch.long
                ),
                "centers": torch.tensor(
                    [row[3] for row in rows], device=device, dtype=dtype
                ),
                "teacher_scores": torch.tensor(
                    [row[4] for row in rows], device=device, dtype=dtype
                ),
                "static_features": torch.stack([row[5] for row in rows]).to(
                    device=device, dtype=dtype
                ),
                "transition_features": torch.stack([row[6] for row in rows]).to(
                    device=device, dtype=dtype
                ),
            })
        else:
            output.append({
                "segments": torch.zeros((0, 2), device=device, dtype=dtype),
                "levels": torch.empty(0, device=device, dtype=torch.long),
                "centers": torch.empty(0, device=device, dtype=dtype),
                "teacher_scores": torch.empty(0, device=device, dtype=dtype),
                "static_features": torch.empty((0, 0), device=device, dtype=dtype),
                "transition_features": torch.empty((0, 0), device=device, dtype=dtype),
            })
    return output


def _sample_sequence(sequence, positions):
    length = sequence.shape[-1]
    positions = positions.clamp(0, max(length - 1, 0))
    left = positions.floor().long()
    right = (left + 1).clamp(max=max(length - 1, 0))
    weight = (positions - left.float()).unsqueeze(0)
    return sequence[:, left] * (1.0 - weight) + sequence[:, right] * weight


def _pool_proposal(fpn_feats, batch_index, segment, level, fpn_strides):
    duration = (segment[1] - segment[0]).clamp(min=1.0)
    positions = torch.stack((
        segment[0],
        segment[0] + 0.25 * duration,
        0.5 * (segment[0] + segment[1]),
        segment[1] - 0.25 * duration,
        segment[1],
    )) / float(fpn_strides[level])
    return _sample_sequence(fpn_feats[level][batch_index], positions).mean(dim=1)


def proposal_relation_loss(
    fpn_feats,
    proposals,
    fpn_strides,
    max_gap_seconds,
    distance_scale=1.0,
):
    if not math.isfinite(distance_scale) or distance_scale <= 0.0:
        raise ValueError("relation distance scale must be positive")
    losses, target_distances, model_distances = [], [], []
    for batch_index, proposal in enumerate(proposals):
        count = proposal["segments"].shape[0]
        for left in range(max(count - 1, 0)):
            right = left + 1
            if float(proposal["centers"][right] - proposal["centers"][left]) > max_gap_seconds:
                continue
            left_level = int(proposal["levels"][left])
            right_level = int(proposal["levels"][right])
            left_feature = _pool_proposal(
                fpn_feats, batch_index, proposal["segments"][left],
                left_level, fpn_strides,
            )
            right_feature = _pool_proposal(
                fpn_feats, batch_index, proposal["segments"][right],
                right_level, fpn_strides,
            )
            model_distance = 1.0 - F.cosine_similarity(
                left_feature[None], right_feature[None]
            )[0]
            target_distance = 1.0 - F.cosine_similarity(
                proposal["static_features"][left][None],
                proposal["static_features"][right][None],
            )[0]
            losses.append(F.smooth_l1_loss(
                model_distance / distance_scale,
                target_distance.detach() / distance_scale,
            ))
            target_distances.append(target_distance.detach())
            model_distances.append(model_distance.detach())
    if not losses:
        zero = sum(feature.sum() * 0.0 for feature in fpn_feats)
        return zero, {
            "pair_count": 0,
            "target_distance": zero.detach(),
            "model_distance": zero.detach(),
        }
    return torch.stack(losses).mean(), {
        "pair_count": len(losses),
        "target_distance": torch.stack(target_distances).mean(),
        "model_distance": torch.stack(model_distances).mean(),
    }
