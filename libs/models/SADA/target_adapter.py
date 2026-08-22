import hashlib
import json
import math
from pathlib import Path

import torch


ORACLE_TARGET_DISTILLATION_MODES = ("off", "selected", "full")
VLM_TARGET_ADAPTER_SCHEMA_VERSION = "offline-vlm-target-adapter-v1"
VLM_TARGET_ADAPTER_SOURCE = "offline_vlm_proposal_semantics"
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
    if payload.get("schema_version") != VLM_TARGET_ADAPTER_SCHEMA_VERSION:
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
        by_video.setdefault(video_id, {})[absolute_index] = {
            "class_id": class_id,
            "semantic_label": TARGET_CLASS_NAMES[class_id],
            "source_fpn_levels": levels,
            "absolute_center_time": absolute_time,
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
):
    """Match frozen proposal semantics and keep current teacher agreement only."""
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("VLM target adapter alpha must be in (0, 1]")
    enabled_levels = set(int(level) for level in enabled_levels)
    batch_size = len(video_list)
    num_classes = logits[0].shape[-1]
    selected_levels = []
    target_levels = []
    hit_levels = []
    agreement_levels = []

    for level, (point_grid, valid, level_logits) in enumerate(
        zip(points, valid_masks, logits)
    ):
        selected = torch.zeros_like(valid, dtype=torch.bool)
        hits = torch.zeros_like(valid, dtype=torch.bool)
        agreement = torch.zeros_like(valid, dtype=torch.bool)
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
                    targets[batch_index, local_index, class_id] = 1.0
                    agrees = int(classes[batch_index, local_index]) == class_id
                    agreement[batch_index, local_index] = agrees
                    selected[batch_index, local_index] = (
                        agrees
                        and float(scores[batch_index, local_index]) >= float(alpha)
                    )
        selected_levels.append(selected)
        target_levels.append(targets)
        hit_levels.append(hits)
        agreement_levels.append(agreement)

    return {
        "selected": torch.cat(selected_levels, dim=1),
        "targets": torch.cat(target_levels, dim=1),
        "cache_hits": torch.cat(hit_levels, dim=1),
        "agreement": torch.cat(agreement_levels, dim=1),
    }
