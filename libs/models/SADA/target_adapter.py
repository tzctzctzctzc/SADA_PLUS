import hashlib
import json
import math
from pathlib import Path

import torch


ORACLE_TARGET_DISTILLATION_MODES = ("off", "selected", "full")
VLM_TARGET_ADAPTER_SCHEMA_VERSIONS = (
    "offline-vlm-target-adapter-v1",
    "offline-clip-target-adapter-v2",
)
VLM_TARGET_ADAPTER_SOURCE = "offline_vlm_proposal_semantics"
VLM_TARGET_ADAPTER_POLICIES = ("verify", "hard", "soft")
TARGET_CLASS_NAMES = (
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


def validate_vlm_target_adapter_resume(checkpoint, expected):
    """Require a continuation checkpoint to use the exact same adapter setup."""
    config = checkpoint.get("config")
    actual = config.get("vlm_target_adapter") if config is not None else None
    if not isinstance(actual, dict) or not actual.get("enabled"):
        raise ValueError(
            "VLM target adapter selection checkpoint differs from resume"
        )
    fields = (
        "cache_sha256",
        "budget",
        "levels",
        "policy",
        "confidence_threshold",
    )
    if any(actual.get(field) != expected.get(field) for field in fields):
        raise ValueError(
            "VLM target adapter continuation settings differ from resume"
        )


def build_oracle_target_distillation(
    scores,
    gt_labels,
    valid_mask,
    enabled_levels,
    alpha,
    mode,
):
    """Select target anchors for diagnostic direct task supervision."""
    if mode not in ORACLE_TARGET_DISTILLATION_MODES[1:]:
        raise ValueError(
            "unknown Oracle target distillation mode: {}".format(mode)
        )
    if (
        scores.shape != valid_mask.shape
        or scores.shape != enabled_levels.shape
        or gt_labels.shape[:-1] != scores.shape
    ):
        raise ValueError(
            "Oracle target distillation inputs have incompatible shapes"
        )

    selected = valid_mask & enabled_levels
    if mode == "selected":
        selected = selected & (scores >= float(alpha))

    foreground = selected & (gt_labels.sum(dim=-1) > 0)
    background = selected & ~foreground
    return {
        "selected": selected,
        "foreground": foreground,
        "background": background,
        "targets": gt_labels.detach().clone(),
    }


def load_vlm_target_adapter_cache(path):
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    schema_version = payload.get("schema_version")
    if schema_version not in VLM_TARGET_ADAPTER_SCHEMA_VERSIONS:
        raise ValueError("unsupported VLM target adapter cache schema")
    if payload.get("source") != VLM_TARGET_ADAPTER_SOURCE:
        raise ValueError("invalid VLM target adapter cache source")
    if payload.get("key_fields") != ["video_id", "absolute_feature_index"]:
        raise ValueError("VLM target adapter cache must use exact absolute indices")
    if not isinstance(payload.get("prompt_protocol"), dict):
        raise ValueError("VLM target adapter cache has no prompt protocol")
    provenance = payload.get("selection_teacher_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("VLM target adapter cache has no selection provenance")

    by_video = {}
    seen = set()
    for entry in payload.get("entries", []):
        video_id = str(entry["video_id"])
        absolute_index = entry["absolute_feature_index"]
        if (
            isinstance(absolute_index, bool)
            or int(absolute_index) != absolute_index
            or int(absolute_index) < 0
        ):
            raise ValueError("invalid VLM target adapter absolute index")
        absolute_index = int(absolute_index)
        key = (video_id, absolute_index)
        if key in seen:
            raise ValueError("duplicate VLM target adapter cache entry")
        seen.add(key)

        class_id = entry["class_id"]
        if (
            isinstance(class_id, bool)
            or int(class_id) != class_id
            or not 0 <= int(class_id) < len(TARGET_CLASS_NAMES)
        ):
            raise ValueError("invalid VLM target adapter class id")
        class_id = int(class_id)
        if entry.get("semantic_label") != TARGET_CLASS_NAMES[class_id]:
            raise ValueError("VLM target adapter class id and label differ")
        levels = tuple(int(level) for level in entry.get("source_fpn_levels", []))
        if not levels or len(set(levels)) != len(levels) or any(
            level < 0 or level > 5 for level in levels
        ):
            raise ValueError("invalid VLM target adapter source levels")
        absolute_time = float(entry["absolute_center_time"])
        if not math.isfinite(absolute_time) or absolute_time < 0:
            raise ValueError("invalid VLM target adapter center time")
        probabilities = entry.get("class_probabilities")
        confidence = float(entry.get("confidence", 1.0))
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("invalid VLM target adapter confidence")
        if probabilities is not None:
            probabilities = tuple(float(value) for value in probabilities)
            if (
                len(probabilities) != len(TARGET_CLASS_NAMES)
                or any(not math.isfinite(value) or value < 0.0 for value in probabilities)
                or abs(sum(probabilities) - 1.0) > 1e-4
            ):
                raise ValueError("invalid VLM target adapter probabilities")
            if max(range(len(probabilities)), key=probabilities.__getitem__) != class_id:
                raise ValueError("VLM target adapter probabilities and class id differ")
            if abs(max(probabilities) - confidence) > 1e-4:
                raise ValueError("VLM target adapter probabilities and confidence differ")
        elif schema_version == "offline-clip-target-adapter-v2":
            raise ValueError("CLIP target adapter entry has no probabilities")
        by_video.setdefault(video_id, {})[absolute_index] = {
            "class_id": class_id,
            "semantic_label": TARGET_CLASS_NAMES[class_id],
            "source_fpn_levels": levels,
            "absolute_center_time": absolute_time,
            "class_probabilities": probabilities,
            "confidence": confidence,
        }

    return (
        by_video,
        len(seen),
        hashlib.sha256(raw).hexdigest(),
        payload,
    )


def build_vlm_target_distillation(
    video_list,
    points,
    valid_masks,
    logits,
    cache_by_video,
    enabled_levels,
    alpha,
    policy="verify",
    confidence_threshold=0.0,
):
    """Match frozen proposal semantics under a fixed class-assistance policy."""
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("VLM target adapter alpha must be in (0, 1]")
    if policy not in VLM_TARGET_ADAPTER_POLICIES:
        raise ValueError("unknown VLM target adapter policy: {}".format(policy))
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("VLM target adapter confidence threshold must be in [0, 1]")
    enabled_levels = set(int(level) for level in enabled_levels)
    batch_size = len(video_list)
    num_classes = logits[0].shape[-1]
    selected_levels = []
    target_levels = []
    hit_levels = []
    agreement_levels = []
    confidence_levels = []
    weight_levels = []

    for level, (point_grid, valid, level_logits) in enumerate(
        zip(points, valid_masks, logits)
    ):
        selected = torch.zeros_like(valid, dtype=torch.bool)
        hits = torch.zeros_like(valid, dtype=torch.bool)
        agreement = torch.zeros_like(valid, dtype=torch.bool)
        confidence = torch.zeros_like(valid, dtype=level_logits.dtype)
        weights = torch.zeros_like(valid, dtype=level_logits.dtype)
        targets = torch.zeros_like(level_logits)
        if level in enabled_levels:
            coordinates = [int(round(value)) for value in point_grid[:, 0].tolist()]
            if any(
                abs(float(value) - coordinate) > 1e-6
                for value, coordinate in zip(point_grid[:, 0].tolist(), coordinates)
            ):
                raise ValueError("FPN point is not an integer feature coordinate")
            coordinate_to_index = {
                coordinate: index for index, coordinate in enumerate(coordinates)
            }
            scores, classes = level_logits.detach().sigmoid().max(dim=-1)
            for batch_index in range(batch_size):
                metadata = video_list[batch_index]["metadata"]
                video_id = str(metadata["video_id"])
                crop_start = int(metadata.get("crop_start", 0))
                for absolute_index, evidence in cache_by_video.get(
                    video_id, {}
                ).items():
                    if level not in evidence["source_fpn_levels"]:
                        continue
                    local_index = coordinate_to_index.get(
                        absolute_index - crop_start
                    )
                    if local_index is None or not bool(
                        valid[batch_index, local_index]
                    ):
                        continue
                    hits[batch_index, local_index] = True
                    class_id = evidence["class_id"]
                    probabilities = evidence.get("class_probabilities")
                    if policy == "soft":
                        if probabilities is None:
                            raise ValueError("soft CLIP policy requires class probabilities")
                        targets[
                            batch_index,
                            local_index,
                            :len(TARGET_CLASS_NAMES),
                        ] = torch.as_tensor(
                            probabilities,
                            dtype=targets.dtype,
                            device=targets.device,
                        )
                    else:
                        targets[batch_index, local_index, class_id] = 1.0
                    clip_confidence = float(evidence.get("confidence", 1.0))
                    confidence[batch_index, local_index] = clip_confidence
                    agrees = int(classes[batch_index, local_index]) == class_id
                    agreement[batch_index, local_index] = agrees
                    confident = clip_confidence >= float(confidence_threshold)
                    if policy == "verify":
                        selected[batch_index, local_index] = (
                            confident
                            and agrees
                            and float(scores[batch_index, local_index]) >= float(alpha)
                        )
                    else:
                        selected[batch_index, local_index] = confident
                    weights[batch_index, local_index] = (
                        clip_confidence if policy == "soft" and confident else float(confident)
                    )
        selected_levels.append(selected)
        target_levels.append(targets)
        hit_levels.append(hits)
        agreement_levels.append(agreement)
        confidence_levels.append(confidence)
        weight_levels.append(weights)

    return {
        "selected": torch.cat(selected_levels, dim=1),
        "targets": torch.cat(target_levels, dim=1),
        "cache_hits": torch.cat(hit_levels, dim=1),
        "agreement": torch.cat(agreement_levels, dim=1),
        "confidence": torch.cat(confidence_levels, dim=1),
        "weights": torch.cat(weight_levels, dim=1),
    }
