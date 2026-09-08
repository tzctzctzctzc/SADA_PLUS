"""Exact semantic-cache lookup plus crop-local online validity routing."""

import json
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .anchor_validity import anchor_validity_features, probe_probabilities


LEGACY_SEMANTIC_CACHE_SCHEMA_VERSION = "offline-vlm-semantic-evidence-v1"
SEMANTIC_CACHE_SCHEMA_VERSION = "offline-vlm-bidirectional-semantic-evidence-v2"
VLM_RESUME_SCHEMA_VERSION = "frozen-routing-teacher-v4"
VLM_RESIDUAL_RESUME_SCHEMA_VERSION = "frozen-routing-teacher-soft-residual-v1"
VLM_FORGETTING_RESUME_SCHEMA_VERSION = "frozen-routing-teacher-forgetting-v1"
BAND_ORACLE_RESUME_SCHEMA_VERSION = "frozen-routing-teacher-band-oracle-v1"
BAND_ORACLE_IGNORE_RESUME_SCHEMA_VERSION = (
    "frozen-routing-teacher-band-oracle-ignore-v1"
)
VLM_TRANSITION_SCHEMA_VERSION = "sada-to-vlm-continuation-v2"
LOW_PROMPT_ROLE = "low_rescue"
HIGH_PROMPT_ROLE = "high_veto"
PROMPT_ROLES = {LOW_PROMPT_ROLE, HIGH_PROMPT_ROLE}
CLEAR_NEGATIVE_LABELS = {"OTHER_ACTIVITY", "TRUE_BACKGROUND"}
FORGETTING_NEGATIVE_LABELS = ("TRUE_BACKGROUND",)
ROUTING_MODES = {
    "semantic_only",
    "high_veto_only",
    "bidirectional_semantic",
    "validity_only",
    "semantic_and_validity",
    "mid_confidence_ignore",
}
BINARY_BAND_ORACLE_MODES = {
    "off",
    "control",
    "high_clean",
    "mid_rescue",
    "low_rescue",
    "both",
    "foreground_rescue",
    "background_clean",
    "full_actionness",
    "high_class_correct",
    "mid_rescue_class_correct",
    "full_class_correct",
}
BINARY_BAND_NEGATIVE_ROUTES = {"background", "ignore"}
SEMANTIC_LABELS = {
    "TARGET_ACTION",
    "OTHER_ACTIVITY",
    "TRUE_BACKGROUND",
    "UNCERTAIN",
}
SEMANTIC_SELECTION_FIELDS = {
    "selection_teacher_checkpoint_sha256",
    "selection_teacher_state_sha256",
    "selection_teacher_weights",
    "selection_beta",
    "selection_alpha",
}

VLM_TRANSITION_FIELDS = {
    "schema_version",
    "transition_mode",
    "parent_sada_checkpoint_sha256",
    "parent_epoch",
    "student_weights",
    "student_state_sha256",
    "student_ema_weights",
    "student_ema_state_sha256",
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
        "ema_state_dict",
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
    current_plain = _plain_config(current_config)
    diagnostic_levels = current_plain.get("diagnostic", {}).get(
        "alignment_levels"
    )
    if diagnostic_levels is not None:
        # This explicit diagnostic changes only which already-computed
        # conditional losses are aggregated. Keep every other model field
        # under the strict continuation check.
        parent["model"] = dict(parent["model"])
        current["model"] = dict(current["model"])
        parent["model"].pop("semantic_alignment_levels", None)
        current["model"].pop("semantic_alignment_levels", None)
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


def restore_training_trajectory_with_added_parameters(
    checkpoint,
    num_iters_epoch,
    model,
    added_prefixes,
    optimizer_gen_source,
    optimizer_gen_source_alignment,
    optimizer_disc,
    scheduler_gen_source,
    scheduler_gen_source_alignment,
    scheduler_disc,
):
    """Restore a SADA trajectory while leaving newly added branch state fresh."""
    name_by_parameter = {id(parameter): name for name, parameter in model.named_parameters()}

    def load_base_state(optimizer, saved):
        current = optimizer.state_dict()
        if len(current["param_groups"]) != len(saved["param_groups"]):
            raise ValueError("optimizer group count changed during branch continuation")
        merged_state = {}
        merged_groups = []
        for live_group, current_group, saved_group in zip(
            optimizer.param_groups, current["param_groups"], saved["param_groups"]
        ):
            current_names = [name_by_parameter[id(parameter)] for parameter in live_group["params"]]
            base_names = [
                name for name in current_names
                if not name.startswith(tuple(added_prefixes))
            ]
            if len(base_names) != len(saved_group["params"]):
                raise ValueError("base optimizer parameter set changed during branch continuation")
            saved_by_name = dict(zip(base_names, saved_group["params"]))
            current_by_name = dict(zip(current_names, current_group["params"]))
            for name in base_names:
                saved_id = saved_by_name[name]
                if saved_id in saved["state"]:
                    merged_state[current_by_name[name]] = saved["state"][saved_id]
            merged_group = dict(saved_group)
            merged_group["params"] = list(current_group["params"])
            merged_groups.append(merged_group)
        optimizer.load_state_dict({"state": merged_state, "param_groups": merged_groups})

    load_base_state(optimizer_gen_source, checkpoint["optimizer_gen_source"])
    load_base_state(
        optimizer_gen_source_alignment,
        checkpoint["optimizer_gen_source_alignment"],
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
    student_ema_state_sha256,
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
        "student_ema_weights": "ema",
        "student_ema_state_sha256": student_ema_state_sha256,
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
    semantic_prompt_protocols=None,
    binary_band_oracle_mode="off",
    binary_band_negative_route="background",
    binary_band_oracle_levels=None,
    soft_residual=False,
    residual_levels=None,
    residual_budget=0.05,
    forgetting=False,
    forgetting_levels=None,
    forgetting_budget=0.01,
    forgetting_labels=None,
    other_activity_class_negative=False,
):
    if other_activity_class_negative and not forgetting:
        raise ValueError("OTHER_ACTIVITY class-negative requires forgetting")
    if binary_band_oracle_mode not in BINARY_BAND_ORACLE_MODES:
        raise ValueError("unknown binary band Oracle mode")
    if binary_band_negative_route not in BINARY_BAND_NEGATIVE_ROUTES:
        raise ValueError("unknown binary band Oracle negative route")
    if soft_residual:
        if binary_band_oracle_mode != "off":
            raise ValueError("soft residual cannot be combined with band Oracle")
        if routing_mode != "semantic_only":
            raise ValueError("soft residual requires semantic_only routing")
        if list(residual_levels or [0]) != [0]:
            raise ValueError("soft residual is preregistered for L0 only")
        if not 0.0 < float(residual_budget) <= 1.0:
            raise ValueError("soft residual budget must be in (0, 1]")
    if forgetting:
        if binary_band_oracle_mode != "off":
            raise ValueError("forgetting cannot be combined with band Oracle")
        if soft_residual:
            raise ValueError("forgetting cannot be combined with soft residual")
        if routing_mode != "high_veto_only":
            raise ValueError("forgetting requires high_veto_only routing")
        if not 0.0 < float(forgetting_budget) <= 0.05:
            raise ValueError("forgetting budget must be in (0, 0.05]")
        if list(forgetting_levels or [0, 1, 2]) != sorted(
            set(forgetting_levels or [0, 1, 2])
        ):
            raise ValueError("forgetting levels must be sorted and unique")
        if any(level not in (0, 1, 2) for level in (forgetting_levels or [0, 1, 2])):
            raise ValueError("forgetting is preregistered for L0-L2 only")
        if tuple(forgetting_labels or FORGETTING_NEGATIVE_LABELS) != FORGETTING_NEGATIVE_LABELS:
            raise ValueError("forgetting only supports TRUE_BACKGROUND labels")
    signature = {
        "schema_version": (
            VLM_FORGETTING_RESUME_SCHEMA_VERSION
            if forgetting
            else (
                VLM_RESIDUAL_RESUME_SCHEMA_VERSION
                if soft_residual
                else (
                    VLM_RESUME_SCHEMA_VERSION
                    if binary_band_oracle_mode == "off"
                    else (
                        BAND_ORACLE_RESUME_SCHEMA_VERSION
                        if binary_band_negative_route == "background"
                        else BAND_ORACLE_IGNORE_RESUME_SCHEMA_VERSION
                    )
                )
            )
        ),
        "semantic_cache_sha256": semantic_cache_sha256,
        "validity_artifact_sha256": validity_artifact_sha256,
        "routing_mode": routing_mode,
        "beta": float(beta),
        "alpha": float(alpha),
        "crop_alignment": int(crop_alignment),
        "routing_teacher_state_sha256": routing_teacher_state_sha256,
        "routing_teacher_weights": routing_teacher_weights,
        "semantic_prompt_protocols": semantic_prompt_protocols or {},
        "evaluation_weights": "student_ema",
    }
    if binary_band_oracle_mode != "off":
        signature["binary_band_oracle_mode"] = binary_band_oracle_mode
        signature["binary_band_negative_route"] = binary_band_negative_route
        signature["binary_band_oracle_levels"] = (
            None
            if binary_band_oracle_levels is None
            else list(binary_band_oracle_levels)
        )
    if soft_residual:
        signature["soft_residual"] = True
        signature["residual_levels"] = list(residual_levels or [0])
        signature["residual_budget"] = float(residual_budget)
    if forgetting:
        signature["forgetting"] = True
        signature["forgetting_levels"] = list(forgetting_levels or [0, 1, 2])
        signature["forgetting_budget"] = float(forgetting_budget)
        signature["forgetting_labels"] = list(
            forgetting_labels or FORGETTING_NEGATIVE_LABELS
        )
        if other_activity_class_negative:
            signature["other_activity_class_negative"] = True
    signature["signature_sha256"] = _canonical_sha256(signature)
    return signature


def validate_vlm_resume(checkpoint, expected_signature):
    actual = checkpoint.get("vlm_routing")
    if actual is None:
        raise ValueError("checkpoint has no frozen VLM routing signature")
    if actual != expected_signature:
        raise ValueError("checkpoint VLM routing signature differs from this run")
    return actual


def semantic_selection_provenance(payload):
    provenance = payload.get("selection_teacher_provenance")
    if not isinstance(provenance, dict) or set(provenance) != SEMANTIC_SELECTION_FIELDS:
        raise ValueError("semantic cache selection teacher provenance fields differ")
    for field in (
        "selection_teacher_checkpoint_sha256",
        "selection_teacher_state_sha256",
    ):
        value = provenance[field]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError("semantic cache has an invalid selection teacher hash")
    if provenance["selection_teacher_weights"] not in {"student", "ema"}:
        raise ValueError("semantic cache has invalid selection teacher weights")
    beta = float(provenance["selection_beta"])
    alpha = float(provenance["selection_alpha"])
    if not 0.0 <= beta < alpha <= 1.0:
        raise ValueError("semantic cache has invalid selection thresholds")
    return {**provenance, "selection_beta": beta, "selection_alpha": alpha}


def validate_semantic_selection_provenance(
    payload,
    expected_teacher_provenance,
    expected_beta,
    expected_alpha,
):
    actual = semantic_selection_provenance(payload)
    expected = {
        "selection_teacher_checkpoint_sha256": expected_teacher_provenance[
            "routing_teacher_checkpoint_sha256"
        ],
        "selection_teacher_state_sha256": expected_teacher_provenance[
            "routing_teacher_state_sha256"
        ],
        "selection_teacher_weights": expected_teacher_provenance[
            "routing_teacher_weights"
        ],
        "selection_beta": float(expected_beta),
        "selection_alpha": float(expected_alpha),
    }
    if actual != expected:
        raise ValueError("semantic cache selection teacher differs from routing teacher")
    return actual


def semantic_prompt_protocols(payload):
    schema = payload.get("semantic_cache_schema_version")
    if schema == LEGACY_SEMANTIC_CACHE_SCHEMA_VERSION:
        prompt_sha256 = payload.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
            raise ValueError("legacy semantic cache has no valid prompt hash")
        return {
            LOW_PROMPT_ROLE: {
                "prompt_version": payload.get("prompt_version"),
                "prompt_sha256": prompt_sha256,
                "model": payload.get("model"),
                "wire_api": payload.get("wire_api"),
            }
        }
    protocols = payload.get("prompt_protocols")
    if not isinstance(protocols, dict) or not protocols:
        raise ValueError("bidirectional semantic cache has no prompt protocols")
    if not set(protocols).issubset(PROMPT_ROLES):
        raise ValueError("semantic cache has an invalid prompt role")
    normalized = {}
    for role, protocol in protocols.items():
        if not isinstance(protocol, dict):
            raise ValueError("semantic prompt protocol must be an object")
        prompt_sha256 = protocol.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or len(prompt_sha256) != 64:
            raise ValueError("semantic prompt protocol has an invalid hash")
        normalized[role] = dict(protocol)
    return normalized


def load_semantic_cache(path):
    """Load role-isolated candidate-blind VLM evidence at exact absolute indices."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    schema = payload.get("semantic_cache_schema_version")
    if schema not in {
        LEGACY_SEMANTIC_CACHE_SCHEMA_VERSION,
        SEMANTIC_CACHE_SCHEMA_VERSION,
    }:
        raise ValueError("unsupported VLM semantic cache schema")
    legacy = schema == LEGACY_SEMANTIC_CACHE_SCHEMA_VERSION
    expected_source = (
        "offline_vlm_candidate_blind_semantics"
        if legacy
        else "offline_vlm_candidate_blind_bidirectional_semantics"
    )
    if payload.get("source") != expected_source:
        raise ValueError("VLM cache is not candidate-blind semantic evidence")
    expected_keys = (
        ["video_id", "absolute_feature_index"]
        if legacy
        else ["prompt_role", "video_id", "absolute_feature_index"]
    )
    if payload.get("key_fields") != expected_keys:
        raise ValueError("VLM semantic cache must use the exact role/index key")
    semantic_selection_provenance(payload)
    protocols = semantic_prompt_protocols(payload)

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
        role = LOW_PROMPT_ROLE if legacy else entry.get("prompt_role")
        if role not in protocols:
            raise ValueError("semantic cache entry has no matching prompt protocol")
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
        key = (role, video_id, absolute_index)
        if key in cache:
            raise ValueError("duplicate VLM semantic cache entry: {}".format(key))
        cache[key] = {
            "prompt_role": role,
            "semantic_pass": semantic_pass,
            "semantic_label": semantic_label,
            "absolute_center_time": absolute_time,
        }
    return cache, len(cache), hashlib.sha256(raw).hexdigest(), payload


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


# Legacy manifests serialized timestamps after float32 tensor arithmetic.  The
# exact anchor identity is the integer key above; this tolerance only absorbs
# sub-millisecond timestamp round-off while still rejecting feature-config drift.
SEMANTIC_TIMESTAMP_TOLERANCE_SEC = 1e-3


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
    high_veto_labels=None,
):
    """Build the complete target route from one routing-teacher forward."""
    if routing_mode not in ROUTING_MODES:
        raise ValueError("unsupported VLM routing mode: {}".format(routing_mode))
    if not 0.0 <= float(beta) < float(alpha) <= 1.0:
        raise ValueError("routing thresholds must satisfy 0 <= beta < alpha <= 1")
    if routing_mode in {"validity_only", "semantic_and_validity"} and validity_model is None:
        raise ValueError("online structural routing requires a validity model")
    high_veto_labels = set(high_veto_labels or CLEAR_NEGATIVE_LABELS)
    if not high_veto_labels.issubset(SEMANTIC_LABELS):
        raise ValueError("high-veto labels must be semantic labels")

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
        if routing_mode not in {"validity_only", "semantic_and_validity"}:
            structure_score = torch.zeros_like(teacher_score)
            structural_pass = torch.zeros_like(valid)
        else:
            validity_features = anchor_validity_features(
                probabilities,
                offsets.detach()[..., 0],
                offsets.detach()[..., 1],
                float(point_grid[0, 3]),
                level,
            )
            fpn_rows = features.detach().transpose(1, 2)
            structure_score = probe_probabilities(
                validity_features, validity_model, fpn_features=fpn_rows
            )
            structural_pass = structure_score >= float(validity_threshold)

        semantic_known = torch.zeros_like(valid)
        semantic_pass = torch.zeros_like(valid)
        high_semantic_known = torch.zeros_like(valid)
        high_semantic_veto = torch.zeros_like(valid)
        high_forgetting_suspect = torch.zeros_like(valid)
        high_other_activity_suspect = torch.zeros_like(valid)
        if routing_mode in {
            "semantic_only",
            "high_veto_only",
            "semantic_and_validity",
            "bidirectional_semantic",
        }:
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
                bands = []
                if routing_mode in {
                    "semantic_only",
                    "semantic_and_validity",
                    "bidirectional_semantic",
                }:
                    bands.append(
                        (
                            LOW_PROMPT_ROLE,
                            mid_confidence[batch_index],
                            semantic_known,
                        )
                    )
                if routing_mode in {"high_veto_only", "bidirectional_semantic"}:
                    bands.append(
                        (
                            HIGH_PROMPT_ROLE,
                            high_confidence[batch_index],
                            high_semantic_known,
                        )
                    )
                for role, band, known_mask in bands:
                    for local_index in torch.nonzero(
                        band, as_tuple=False
                    ).flatten().tolist():
                        absolute_index = _exact_absolute_index(
                            point_grid[local_index, 0], crop_start
                        )
                        evidence = semantic_cache.get(
                            (role, video_id, absolute_index)
                        )
                        if evidence is None and role == LOW_PROMPT_ROLE:
                            # Compatibility for in-memory legacy tests/callers.
                            evidence = semantic_cache.get((video_id, absolute_index))
                        if evidence is None:
                            continue
                        expected_time = _expected_center_time(
                            metadata, absolute_index
                        )
                        if abs(
                            evidence["absolute_center_time"] - expected_time
                        ) > SEMANTIC_TIMESTAMP_TOLERANCE_SEC:
                            raise ValueError(
                                "semantic cache timestamp differs from current "
                                "feature configuration for {} role {} index {}".format(
                                    video_id, role, absolute_index
                                )
                            )
                        known_mask[batch_index, local_index] = True
                        if role == LOW_PROMPT_ROLE:
                            semantic_pass[batch_index, local_index] = evidence[
                                "semantic_pass"
                            ]
                        else:
                            high_semantic_veto[batch_index, local_index] = (
                                evidence["semantic_label"] in high_veto_labels
                            )
                            high_forgetting_suspect[batch_index, local_index] = (
                                evidence["semantic_label"] == "TRUE_BACKGROUND"
                            )
                            high_other_activity_suspect[batch_index, local_index] = (
                                evidence["semantic_label"] == "OTHER_ACTIVITY"
                            )
        if routing_mode in {"semantic_only", "bidirectional_semantic"}:
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
        high_veto = high_confidence & high_semantic_known & high_semantic_veto
        high_action = high_confidence & ~high_veto
        action = high_action | mid_action
        ignore = (mid_confidence & ~mid_action) | high_veto
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
                "high_action": high_action,
                "high_veto": high_veto,
                "high_semantic_known": high_semantic_known,
                "high_semantic_veto": high_semantic_veto,
                "high_forgetting_suspect": high_forgetting_suspect,
                "high_other_activity_suspect": high_other_activity_suspect,
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
    if "action_labels" in route:
        if route["action_labels"].shape != (*valid_mask.shape, num_classes):
            raise ValueError("Oracle action labels have incompatible shape")
        labels = route["action_labels"][valid_mask].to(dtype).clone()
        labels[~action] = 0
    else:
        classes = route["pseudo_class"][valid_mask]
        labels = torch.zeros(
            action.shape[0], num_classes, dtype=dtype, device=action.device
        )
        labels[action] = F.one_hot(
            classes[action], num_classes=num_classes
        ).to(dtype)
    return labels, action, background, ignore


def apply_binary_band_oracle(
    route,
    gt_labels,
    mode,
    negative_route="background",
):
    """Apply a GT action/background diagnostic to frozen-teacher score bands."""
    if mode not in BINARY_BAND_ORACLE_MODES - {"off"}:
        raise ValueError("unknown binary band Oracle mode: {}".format(mode))
    if negative_route not in BINARY_BAND_NEGATIVE_ROUTES:
        raise ValueError(
            "unknown binary band Oracle negative route: {}".format(
                negative_route
            )
        )
    if gt_labels.shape[:-1] != route["valid"].shape:
        raise ValueError("binary band Oracle GT shape differs from routing mask")

    valid = route["valid"].bool()
    high = route["high_confidence"].bool()
    mid = route["mid_confidence"].bool()
    low = route["low_confidence"].bool()
    if not torch.equal(
        high.to(torch.int8) + mid.to(torch.int8) + low.to(torch.int8),
        valid.to(torch.int8),
    ):
        raise ValueError("frozen-teacher score bands do not partition valid anchors")

    gt_foreground = gt_labels.sum(dim=-1) > 0
    action = high.clone()
    background = low.clone()
    ignore = mid.clone()

    if mode in {"high_clean", "both"}:
        high_background = high & ~gt_foreground
        action &= ~high_background
        if negative_route == "background":
            background |= high_background
        else:
            ignore |= high_background

    if mode in {"mid_rescue", "mid_rescue_class_correct", "both"}:
        mid_foreground = mid & gt_foreground
        mid_background = mid & ~gt_foreground
        action |= mid_foreground
        ignore &= ~mid_foreground
        if negative_route == "background":
            background |= mid_background
            ignore &= ~mid_background

    if mode == "low_rescue":
        low_foreground = low & gt_foreground
        action |= low_foreground
        background &= ~low_foreground

    if mode == "foreground_rescue":
        rescued = (mid | low) & gt_foreground
        action |= rescued
        background &= ~rescued
        ignore &= ~rescued

    if mode == "background_clean":
        action = high & gt_foreground
        background = valid & ~gt_foreground
        ignore = valid & gt_foreground & ~high

    if mode in {"full_actionness", "full_class_correct"}:
        action = valid & gt_foreground
        background = valid & ~gt_foreground
        ignore = torch.zeros_like(valid)

    action_labels = None
    if mode in {
        "high_class_correct",
        "mid_rescue_class_correct",
        "full_class_correct",
    }:
        pseudo_class = route["pseudo_class"]
        action_labels = torch.zeros_like(gt_labels)
        action_labels[action] = F.one_hot(
            pseudo_class[action], num_classes=gt_labels.shape[-1]
        ).to(gt_labels.dtype)
        if mode == "full_class_correct":
            corrected = action
        elif mode == "mid_rescue_class_correct":
            corrected = mid & gt_foreground
        else:
            corrected = high & gt_foreground
        action_labels[corrected] = gt_labels[corrected]

    partition = (
        action.to(torch.int8)
        + background.to(torch.int8)
        + ignore.to(torch.int8)
    )
    if not torch.equal(partition, valid.to(torch.int8)):
        raise ValueError("binary band Oracle did not produce a complete partition")

    result = dict(route)
    result.update({
        "action": action,
        "background": background,
        "ignore": ignore,
        "high_action": high & action,
        "high_veto": high & ~action,
        "mid_action": mid & action,
        "band_oracle_gt_foreground": valid & gt_foreground,
        "band_oracle_high_background": high & ~gt_foreground,
        "band_oracle_mid_foreground": mid & gt_foreground,
        "band_oracle_mid_background": mid & ~gt_foreground,
    })
    if action_labels is not None:
        result["action_labels"] = action_labels
    return result


build_online_vlm_routing = build_full_vlm_routing
