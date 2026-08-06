import json
import math
from collections import defaultdict
from pathlib import Path

import torch


VALID_DECISIONS = {"action", "background", "ignore"}
ROUTING_SCHEMA_VERSION = "candidate-blind-exact-anchor-routing-v1"


def load_vlm_routing_cache(path, num_classes):
    """Load a GT-free offline VLM routing cache indexed by video and level."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    exact_anchor_rescue = payload.get("routing_policy") == "exact_anchor_rescue"
    if exact_anchor_rescue and payload.get("routing_schema_version") != ROUTING_SCHEMA_VERSION:
        raise ValueError("unsupported exact-anchor VLM routing schema")
    strict_fields = {
        "video_id",
        "fpn_level",
        "anchor_index",
        "global_anchor_index",
        "absolute_center_time",
        "teacher_top1_class",
        "decision",
    }
    if exact_anchor_rescue and not strict_fields.issubset(
        set(payload.get("bank_key_fields") or []) | {"anchor_index", "decision"}
    ):
        raise ValueError("exact-anchor VLM cache is missing strict bank key metadata")
    by_video_level = defaultdict(list)
    seen = set()
    seen_routes = set()
    for entry in entries:
        if exact_anchor_rescue and not strict_fields.issubset(entry):
            raise ValueError("exact-anchor VLM cache entry is missing a strict key")
        video_id = str(entry["video_id"])
        level = int(entry["fpn_level"])
        anchor_index = int(entry["anchor_index"])
        decision = entry["decision"]
        class_id = entry.get("class_id")
        key = (
            video_id,
            level,
            int(entry["global_anchor_index"]),
            float(entry["absolute_center_time"]),
            int(entry["teacher_top1_class"]),
        ) if exact_anchor_rescue else (video_id, level, anchor_index)
        if key in seen:
            raise ValueError("duplicate VLM cache entry: {}".format(key))
        if decision not in VALID_DECISIONS:
            raise ValueError("invalid VLM cache decision: {}".format(decision))
        if decision == "action":
            class_id = int(class_id)
            if class_id < 0 or class_id >= num_classes:
                raise ValueError("invalid VLM action class: {}".format(class_id))
            if exact_anchor_rescue and class_id != int(entry["teacher_top1_class"]):
                raise ValueError("exact-anchor rescue class differs from teacher top-1")
        elif class_id is not None:
            raise ValueError("non-action VLM entry cannot carry class_id")
        route_key = (video_id, level, anchor_index)
        if route_key in seen_routes:
            raise ValueError("multiple VLM cache entries route the same anchor: {}".format(route_key))
        seen.add(key)
        seen_routes.add(route_key)
        by_video_level[(video_id, level)].append(
            {
                "anchor_index": anchor_index,
                "decision": decision,
                "class_id": class_id,
            }
        )
    return dict(by_video_level), len(entries)


def build_vlm_routing(cache, video_list, points, target_masks):
    """Project full-video cached anchors onto the current random training crop."""
    routes = []
    for level, (level_points, level_mask) in enumerate(zip(points, target_masks)):
        batch_size, length = level_mask.shape
        matched = torch.zeros_like(level_mask, dtype=torch.bool)
        action = torch.zeros_like(level_mask, dtype=torch.bool)
        background = torch.zeros_like(level_mask, dtype=torch.bool)
        ignore = torch.zeros_like(level_mask, dtype=torch.bool)
        pseudo_class = torch.zeros_like(level_mask, dtype=torch.long)
        stride = int(round(float(level_points[0, 3].item())))

        for batch_index, video in enumerate(video_list):
            metadata = video["metadata"]
            video_id = str(metadata["video_id"])
            crop_start = int(metadata.get("crop_start", 0))
            for entry in cache.get((video_id, level), []):
                absolute_point = entry["anchor_index"] * stride
                local_position = (absolute_point - crop_start) / float(stride)
                # Python's round-to-even can collapse consecutive anchors when
                # crop_start is exactly half an FPN stride from the full grid.
                local_index = int(math.floor(local_position + 0.5))
                if local_index < 0 or local_index >= length:
                    continue
                projected_point = crop_start + float(
                    level_points[local_index, 0].item()
                )
                if abs(projected_point - absolute_point) > stride / 2.0 + 1e-6:
                    continue
                if not bool(level_mask[batch_index, local_index]):
                    continue
                if matched[batch_index, local_index]:
                    raise ValueError(
                        "multiple VLM entries project to {} level {} index {}".format(
                            video_id, level, local_index
                        )
                    )
                matched[batch_index, local_index] = True
                decision = entry["decision"]
                if decision == "action":
                    action[batch_index, local_index] = True
                    pseudo_class[batch_index, local_index] = entry["class_id"]
                elif decision == "background":
                    background[batch_index, local_index] = True
                else:
                    ignore[batch_index, local_index] = True

        routes.append(
            {
                "matched": matched,
                "action": action,
                "background": background,
                "ignore": ignore,
                "pseudo_class": pseudo_class,
            }
        )
    return routes
