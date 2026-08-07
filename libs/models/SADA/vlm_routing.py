"""Exact semantic-cache lookup plus crop-local online validity routing."""

import json
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .anchor_validity import anchor_validity_features, probe_probabilities


SEMANTIC_CACHE_SCHEMA_VERSION = "offline-vlm-semantic-evidence-v1"
VLM_RESUME_SCHEMA_VERSION = "frozen-routing-teacher-v2"
VLM_TRANSITION_SCHEMA_VERSION = "sada-to-vlm-continuation-v1"
ROUTING_MODES = {
    "semantic_only",
    "validity_only",
    "semantic_and_validity",
    "mid_confidence_ignore",
}
SEMANTIC_LABELS = {
    "TARGET_ACTION",
    "OTHER_ACTIVITY",
    "TRUE_BACKGROUND",
    "UNCERTAIN",
}

VLM_TRANSITION_FIELDS = {
    "schema_version",
    "transition_mode",
    "parent_sada_checkpoint_sha256",
    "parent_epoch",
    "student_weights",
    "student_state_sha256",
    "routing_teacher_checkpoint_sha256",
    "routing_teacher_state_sha256",
    "routing_teacher_weights",
    "completed_steps",
    "transition_sha256",
}


def _canonical_sha256(payload):
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_config(config):
    if hasattr(config, "config"):
        config = config.config
    if isinstance(config, dict):
        return {str(key): _plain_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_plain_config(value) for value in config]
    return config


def _continuation_protocol(config):
    config = _plain_config(config)
    if not isinstance(config, dict):
        raise ValueError("continuation checkpoint has no readable config")
    dataset = dict(config.get("dataset", {}))
    dataset.setdefault("crop_alignment", 1)
    dataset.pop("stage1_context", None)
    keys = (
        "dataset_name",
        "src_train_split",
        "src_val_split",
        "tgt_train_split",
        "tgt_val_split",
        "loader",
        "model",
        "opt",
        "train_cfg",
    )
    protocol = {key: config.get(key) for key in keys}
    protocol["dataset"] = dataset
    return protocol


def validate_sada_continuation_checkpoint(checkpoint, current_config):
    """Require a genuine ordinary-SADA training state for the first VLM fork."""
    if checkpoint.get("vlm_routing") is not None:
        raise ValueError(
            "SADA-to-VLM transition requires an ordinary SADA checkpoint; "
            "use --resume for a VLM checkpoint"
        )
    if checkpoint.get("stage1") is not None:
        raise ValueError("SADA-to-VLM transition cannot start from a Stage 1 checkpoint")
    required = {
        "epoch",
        "state_dict",
        "optimizer_gen_source",
        "optimizer_gen_source_alignment",
        "optimizer_disc",
        "config",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            "SADA continuation checkpoint is missing training state: {}".format(
                missing
            )
        )
    parent = _continuation_protocol(checkpoint["config"])
    current = _continuation_protocol(current_config)
    mismatches = [key for key in parent if parent[key] != current[key]]
    if mismatches:
        raise ValueError(
            "SADA continuation changes the training protocol outside routing: {}".format(
                mismatches
            )
        )
    return int(checkpoint["epoch"])


def restore_training_trajectory(
    checkpoint,
    num_iters_epoch,
    optimizer_gen_source,
    optimizer_gen_source_alignment,
    optimizer_disc,
    scheduler_gen_source,
    scheduler_gen_source_alignment,
    scheduler_disc,
):
    optimizer_gen_source.load_state_dict(checkpoint["optimizer_gen_source"])
    optimizer_gen_source_alignment.load_state_dict(
        checkpoint["optimizer_gen_source_alignment"]
    )
    optimizer_disc.load_state_dict(checkpoint["optimizer_disc"])
    completed_steps = int(checkpoint["epoch"]) * int(num_iters_epoch)
    for scheduler, optimizer in (
        (scheduler_gen_source, optimizer_gen_source),
        (scheduler_gen_source_alignment, optimizer_gen_source_alignment),
        (scheduler_disc, optimizer_disc),
    ):
        if scheduler is not None:
            scheduler.last_epoch = completed_steps
            scheduler._step_count = completed_steps + 1
            scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
    return completed_steps


def build_vlm_transition(
    parent_sada_checkpoint_sha256,
    parent_epoch,
    student_state_sha256,
    routing_teacher_checkpoint_sha256,
    routing_teacher_state_sha256,
    routing_teacher_weights,
    completed_steps,
):
    transition = {
        "schema_version": VLM_TRANSITION_SCHEMA_VERSION,
        "transition_mode": "continuation",
        "parent_sada_checkpoint_sha256": parent_sada_checkpoint_sha256,
        "parent_epoch": int(parent_epoch),
        "student_weights": "student",
        "student_state_sha256": student_state_sha256,
        "routing_teacher_checkpoint_sha256": routing_teacher_checkpoint_sha256,
        "routing_teacher_state_sha256": routing_teacher_state_sha256,
        "routing_teacher_weights": routing_teacher_weights,
        "completed_steps": int(completed_steps),
    }
    transition["transition_sha256"] = _canonical_sha256(transition)
    return transition


def validate_vlm_transition(
    checkpoint,
    expected_transition=None,
    expected_teacher_checkpoint_sha256=None,
    expected_teacher_state_sha256=None,
    expected_teacher_weights=None,
):
    transition = checkpoint.get("vlm_transition")
    if not isinstance(transition, dict):
        raise ValueError("checkpoint has no SADA-to-VLM continuation provenance")
    if set(transition) != VLM_TRANSITION_FIELDS:
        raise ValueError("checkpoint VLM transition fields differ")
    if transition.get("schema_version") != VLM_TRANSITION_SCHEMA_VERSION:
        raise ValueError("unsupported SADA-to-VLM transition schema")
    unsigned = dict(transition)
    recorded_hash = unsigned.pop("transition_sha256")
    if recorded_hash != _canonical_sha256(unsigned):
        raise ValueError("checkpoint VLM transition hash mismatch")
    if expected_transition is not None and transition != expected_transition:
        raise ValueError("checkpoint VLM transition provenance differs")
    expected_teacher = {
        "routing_teacher_checkpoint_sha256": expected_teacher_checkpoint_sha256,
        "routing_teacher_state_sha256": expected_teacher_state_sha256,
        "routing_teacher_weights": expected_teacher_weights,
    }
    mismatches = {
        key: (transition[key], value)
        for key, value in expected_teacher.items()
        if value is not None and transition[key] != value
    }
    if mismatches:
        raise ValueError(
            "checkpoint VLM transition teacher differs from this run: {}".format(
                mismatches
            )
        )
    return transition


def build_vlm_signature(
    semantic_cache_sha256,
    validity_artifact_sha256,
    routing_mode,
    beta,
    alpha,
    crop_alignment,
    routing_teacher_state_sha256,
    routing_teacher_weights,
):
    signature = {
        "schema_version": VLM_RESUME_SCHEMA_VERSION,
        "semantic_cache_sha256": semantic_cache_sha256,
        "validity_artifact_sha256": validity_artifact_sha256,
        "routing_mode": routing_mode,
        "beta": float(beta),
        "alpha": float(alpha),
        "crop_alignment": int(crop_alignment),
        "routing_teacher_state_sha256": routing_teacher_state_sha256,
        "routing_teacher_weights": routing_teacher_weights,
    }
    signature["signature_sha256"] = _canonical_sha256(signature)
    return signature


def validate_vlm_resume(checkpoint, expected_signature):
    actual = checkpoint.get("vlm_routing")
    if actual is None:
        raise ValueError("checkpoint has no frozen VLM routing signature")
    if actual != expected_signature:
        raise ValueError("checkpoint VLM routing signature differs from this run")
    return actual


def load_semantic_cache(path):
    """Load candidate-blind VLM evidence keyed only by absolute feature index."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
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
    return cache, len(cache), hashlib.sha256(raw).hexdigest()


def _exact_absolute_index(local_coordinate, crop_start):
    coordinate = float(local_coordinate)
    rounded = int(round(coordinate))
    if abs(coordinate - rounded) > 1e-6:
        raise ValueError("FPN point coordinate is not an integer feature index")
    return int(crop_start) + rounded


def _expected_center_time(metadata, absolute_index):
    expected = (
        absolute_index * float(metadata["feat_stride"])
        + 0.5 * float(metadata["feat_num_frames"])
    ) / float(metadata["fps"])
    duration = float(metadata["duration"])
    return min(max(expected, 0.0), duration)


def build_full_vlm_routing(
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
    """Build the complete target route from one routing-teacher forward."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError("unsupported VLM routing mode: {}".format(routing_mode))
    if not 0.0 <= float(beta) < float(alpha) <= 1.0:
        raise ValueError("routing thresholds must satisfy 0 <= beta < alpha <= 1")
    if routing_mode in {"validity_only", "semantic_and_validity"} and validity_model is None:
        raise ValueError("online structural routing requires a validity model")

    routes = []
    for level, (point_grid, level_mask, logits, offsets, features) in enumerate(
        zip(points, target_masks, teacher_logits, teacher_offsets, fpn_features)
    ):
        probabilities = logits.detach().sigmoid()
        teacher_score, teacher_class = probabilities.max(dim=-1)
        valid = level_mask.bool()
        high_confidence = valid & (teacher_score >= float(alpha))
        mid_confidence = (
            valid
            & (teacher_score > float(beta))
            & (teacher_score < float(alpha))
        )
        low_confidence = valid & (teacher_score <= float(beta))
        validity_features = anchor_validity_features(
            probabilities,
            offsets.detach()[..., 0],
            offsets.detach()[..., 1],
            float(point_grid[0, 3]),
            level,
        )

        if routing_mode not in {"validity_only", "semantic_and_validity"}:
            structure_score = torch.zeros_like(teacher_score)
            structural_pass = torch.zeros_like(valid)
        else:
            fpn_rows = features.detach().transpose(1, 2)
            structure_score = probe_probabilities(
                validity_features, validity_model, fpn_features=fpn_rows
            )
            structural_pass = structure_score >= float(validity_threshold)

        semantic_known = torch.zeros_like(valid)
        semantic_pass = torch.zeros_like(valid)
        if routing_mode in {"semantic_only", "semantic_and_validity"}:
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
                    mid_confidence[batch_index], as_tuple=False
                ).flatten().tolist():
                    absolute_index = _exact_absolute_index(
                        point_grid[local_index, 0], crop_start
                    )
                    evidence = semantic_cache.get((video_id, absolute_index))
                    if evidence is None:
                        continue
                    expected_time = _expected_center_time(
                        metadata, absolute_index
                    )
                    if abs(
                        evidence["absolute_center_time"] - expected_time
                    ) > 1e-5:
                        raise ValueError(
                            "semantic cache timestamp differs from current "
                            "feature configuration for {} index {}".format(
                                video_id, absolute_index
                            )
                        )
                    semantic_known[batch_index, local_index] = True
                    semantic_pass[batch_index, local_index] = evidence[
                        "semantic_pass"
                    ]
        if routing_mode == "semantic_only":
            mid_action = mid_confidence & semantic_known & semantic_pass
        elif routing_mode == "validity_only":
            mid_action = mid_confidence & structural_pass
        elif routing_mode == "semantic_and_validity":
            mid_action = (
                mid_confidence
                & semantic_known
                & semantic_pass
                & structural_pass
            )
        else:
            mid_action = torch.zeros_like(valid)
        action = high_confidence | mid_action
        ignore = mid_confidence & ~mid_action
        background = low_confidence
        partition = action.to(torch.int8) + ignore.to(torch.int8) + background.to(torch.int8)
        if not torch.equal(partition, valid.to(torch.int8)):
            raise ValueError("routing teacher did not produce a complete partition")

        routes.append(
            {
                "valid": valid,
                "matched": valid,
                "action": action,
                "background": background,
                "ignore": ignore,
                "pseudo_class": teacher_class,
                "teacher_score": teacher_score,
                "high_confidence": high_confidence,
                "mid_confidence": mid_confidence,
                "mid_action": mid_action,
                "low_confidence": low_confidence,
                "semantic_known": semantic_known,
                "semantic_pass": semantic_pass,
                "structural_pass": structural_pass,
                "structure_score": structure_score,
            }
        )
    return routes


def target_labels_from_full_routing(route, valid_mask, num_classes, dtype):
    """Materialize labels without consulting student classification outputs."""
    valid_mask = valid_mask.bool()
    if not torch.equal(route["valid"], valid_mask):
        raise ValueError("routing-teacher mask differs from student target mask")
    action = route["action"][valid_mask]
    background = route["background"][valid_mask]
    ignore = route["ignore"][valid_mask]
    if not torch.all(action.to(torch.int8) + background.to(torch.int8) + ignore.to(torch.int8) == 1):
        raise ValueError("full target routing is not mutually exclusive")
    classes = route["pseudo_class"][valid_mask]
    labels = torch.zeros(
        action.shape[0], num_classes, dtype=dtype, device=action.device
    )
    labels[action] = F.one_hot(
        classes[action], num_classes=num_classes
    ).to(dtype)
    return labels, action, background, ignore


build_online_vlm_routing = build_full_vlm_routing
