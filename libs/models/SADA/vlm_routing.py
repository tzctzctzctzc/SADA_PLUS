"""Exact semantic-cache lookup plus crop-local online validity routing."""

import json
import math
from pathlib import Path

import torch

from .anchor_validity import anchor_validity_features, probe_probabilities


SEMANTIC_CACHE_SCHEMA_VERSION = "offline-vlm-semantic-evidence-v1"
ROUTING_MODES = {"semantic_only", "structure_only", "semantic_and_structure"}
SEMANTIC_LABELS = {
    "TARGET_ACTION",
    "OTHER_ACTIVITY",
    "TRUE_BACKGROUND",
    "UNCERTAIN",
}


def load_semantic_cache(path):
    """Load candidate-blind VLM evidence keyed only by absolute feature index."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("semantic_cache_schema_version") != SEMANTIC_CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported VLM semantic cache schema")
    if payload.get("source") != "offline_vlm_candidate_blind_semantics":
        raise ValueError("VLM cache is not candidate-blind semantic evidence")
    if payload.get("key_fields") != ["video_id", "absolute_feature_index"]:
        raise ValueError("VLM semantic cache must use the exact absolute-index key")

    cache = {}
    for entry in payload.get("entries", []):
        if any(
            key in entry
            for key in (
                "teacher_top1_class",
                "class_id",
                "structure_score",
                "structural_pass",
                "decision",
                "fpn_level",
            )
        ):
            raise ValueError("VLM semantic cache contains online routing evidence")
        video_id = str(entry["video_id"])
        absolute_index = entry["absolute_feature_index"]
        if isinstance(absolute_index, bool) or int(absolute_index) != absolute_index:
            raise ValueError("absolute_feature_index must be an integer")
        absolute_index = int(absolute_index)
        if absolute_index < 0:
            raise ValueError("absolute_feature_index must be non-negative")
        semantic_pass = entry.get("semantic_pass")
        if not isinstance(semantic_pass, bool):
            raise ValueError("semantic_pass must be boolean")
        semantic_label = entry.get("semantic_label")
        if semantic_label not in SEMANTIC_LABELS:
            raise ValueError("invalid VLM semantic label")
        if semantic_pass != (semantic_label == "TARGET_ACTION"):
            raise ValueError("semantic_pass disagrees with semantic_label")
        absolute_time = float(entry["absolute_center_time"])
        if not math.isfinite(absolute_time) or absolute_time < 0:
            raise ValueError("invalid absolute_center_time")
        key = (video_id, absolute_index)
        if key in cache:
            raise ValueError("duplicate VLM semantic cache entry: {}".format(key))
        cache[key] = {
            "semantic_pass": semantic_pass,
            "semantic_label": semantic_label,
            "absolute_center_time": absolute_time,
        }
    return cache, len(cache)


def _exact_absolute_index(local_coordinate, crop_start):
    coordinate = float(local_coordinate)
    rounded = int(round(coordinate))
    if abs(coordinate - rounded) > 1e-6:
        raise ValueError("FPN point coordinate is not an integer feature index")
    return int(crop_start) + rounded


def build_online_vlm_routing(
    semantic_cache,
    validity_model,
    validity_threshold,
    routing_mode,
    video_list,
    points,
    target_masks,
    teacher_logits,
    teacher_offsets,
    fpn_features,
    beta,
    alpha,
):
    """Route low-confidence anchors using exact semantics and current-crop evidence."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError("unsupported VLM routing mode: {}".format(routing_mode))
    if routing_mode != "semantic_only" and validity_model is None:
        raise ValueError("online structural routing requires a validity model")

    routes = []
    for level, (point_grid, level_mask, logits, offsets, features) in enumerate(
        zip(points, target_masks, teacher_logits, teacher_offsets, fpn_features)
    ):
        probabilities = logits.detach().sigmoid()
        teacher_score, teacher_class = probabilities.max(dim=-1)
        low_confidence = (
            level_mask.bool()
            & (teacher_score > float(beta))
            & (teacher_score < float(alpha))
        )
        validity_features = anchor_validity_features(
            probabilities,
            offsets.detach()[..., 0],
            offsets.detach()[..., 1],
            float(point_grid[0, 3]),
            level,
        )

        if validity_model is None:
            structure_score = torch.ones_like(teacher_score)
            structural_pass = torch.ones_like(low_confidence)
        else:
            fpn_rows = features.detach().transpose(1, 2)
            structure_score = probe_probabilities(
                validity_features, validity_model, fpn_features=fpn_rows
            )
            structural_pass = structure_score >= float(validity_threshold)

        semantic_known = torch.zeros_like(low_confidence)
        semantic_pass = torch.zeros_like(low_confidence)
        if routing_mode != "structure_only":
            for batch_index, video in enumerate(video_list):
                metadata = video["metadata"]
                video_id = str(metadata["video_id"])
                raw_crop_start = metadata.get("crop_start", 0)
                if (
                    isinstance(raw_crop_start, bool)
                    or int(raw_crop_start) != raw_crop_start
                ):
                    raise ValueError("crop_start must be an integer feature index")
                crop_start = int(raw_crop_start)
                for local_index in torch.nonzero(
                    level_mask[batch_index].bool(), as_tuple=False
                ).flatten().tolist():
                    absolute_index = _exact_absolute_index(
                        point_grid[local_index, 0], crop_start
                    )
                    evidence = semantic_cache.get((video_id, absolute_index))
                    if evidence is None:
                        continue
                    semantic_known[batch_index, local_index] = True
                    semantic_pass[batch_index, local_index] = evidence[
                        "semantic_pass"
                    ]
        else:
            semantic_known = torch.ones_like(low_confidence)
            semantic_pass = torch.ones_like(low_confidence)

        matched = low_confidence
        if routing_mode == "semantic_only":
            action = matched & semantic_known & semantic_pass
        elif routing_mode == "structure_only":
            action = low_confidence & structural_pass
        else:
            action = matched & semantic_known & semantic_pass & structural_pass
        ignore = matched & ~action

        routes.append(
            {
                "matched": matched,
                "action": action,
                "background": torch.zeros_like(action),
                "ignore": ignore,
                "pseudo_class": teacher_class,
                "semantic_known": semantic_known,
                "semantic_pass": semantic_pass,
                "structural_pass": structural_pass,
                "structure_score": structure_score,
                "low_confidence": low_confidence,
            }
        )
    return routes
