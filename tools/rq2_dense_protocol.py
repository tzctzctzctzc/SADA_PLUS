"""Shared temporal grid and protocol checks for full-video RQ2 features."""

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np


SCHEMA = "rq2-dense-vlm-feature-v1"
FEATURE_FPS = 30.0
FEAT_STRIDE = 16
RECEPTIVE_FIELD_FRAMES = 32

NEUTRAL_MULTIMODAL_PROMPT = (
    "Observe the chronological video frames and form a neutral representation "
    "of the visible content. Do not assign an action category and do not predict "
    "timestamps."
)
TIMEPLE_NEUTRAL_PROMPT = (
    "Observe the chronological video frames and form a neutral interval "
    "representation of the visible content without assigning an action category."
)
NEUTRAL_MULTIMODAL_PROMPT_SHA256 = hashlib.sha256(
    NEUTRAL_MULTIMODAL_PROMPT.encode("utf-8")
).hexdigest()
TIMEPLE_NEUTRAL_PROMPT_SHA256 = hashlib.sha256(
    TIMEPLE_NEUTRAL_PROMPT.encode("utf-8")
).hexdigest()

MODEL_SPECS = {
    "egovideo_hod": {
        "status": "smoke_verified",
        "formal_eligible": True,
        "output_dim": 512,
        "input_frames": 8,
        "spatial_size": 336,
        "interface": "official_joint_video_cls",
    },
    "internvideo2_clip_1b": {
        "status": "smoke_verified",
        "formal_eligible": True,
        "output_dim": 768,
        "input_frames": 8,
        "spatial_size": 224,
        "interface": "official_video_clip_projector",
        "execution_backend": "official_torch_unfused_ops",
    },
    "qwen3_vl_8b": {
        "status": "smoke_verified",
        "formal_eligible": True,
        "output_dim": 4096,
        "input_frames": 8,
        "spatial_size": 448,
        "interface": "neutral_prompt_multimodal_last_token",
        "prompt_sha256": NEUTRAL_MULTIMODAL_PROMPT_SHA256,
        "input_modality": "native_video_frames",
    },
    "timelens2_8b": {
        "status": "smoke_verified",
        "formal_eligible": True,
        "output_dim": 4096,
        "input_frames": 8,
        "spatial_size": 448,
        "interface": "neutral_prompt_multimodal_last_token",
        "prompt_sha256": NEUTRAL_MULTIMODAL_PROMPT_SHA256,
        "input_modality": "native_video_frames",
    },
    "timeple_8b": {
        "status": "smoke_verified",
        "formal_eligible": True,
        "output_dim": 1024,
        "input_frames": 8,
        "spatial_size": 448,
        "interface": "timespan_hidden_interface_adapter_decoder_trunk",
        "prompt_sha256": TIMEPLE_NEUTRAL_PROMPT_SHA256,
        "input_modality": "native_video_frames",
    },
    "qwen3_vl_8b_vision_tower": {
        "status": "ablation_only",
        "formal_eligible": False,
        "output_dim": 4096,
        "input_frames": 8,
        "spatial_size": 448,
        "interface": "vision_tower_spatial_pool_then_temporal_mean",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text(path):
    text = Path(path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def protocol_payload(model_name, checkpoint_hashes, provenance=None):
    spec = MODEL_SPECS[model_name]
    return {
        "schema": SCHEMA,
        "model_name": model_name,
        "model_spec": spec,
        "feature_fps": FEATURE_FPS,
        "feat_stride": FEAT_STRIDE,
        "receptive_field_frames": RECEPTIVE_FIELD_FRAMES,
        "temporal_grid": "slowfast-compatible-left-edge-v1",
        "sampling": "deterministic-uniform-within-receptive-field",
        "normalization": "l2",
        "checkpoint_sha256": checkpoint_hashes,
        "uses_annotations": False,
        "provenance": provenance or {},
    }


def protocol_sha256(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def video_set_sha256(video_ids):
    encoded = "".join(video_id + "\n" for video_id in video_ids).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def temporal_length(frame_count, source_fps):
    if frame_count <= 0 or source_fps <= 0:
        raise ValueError("frame count and source FPS must be positive")
    duration = frame_count / float(source_fps)
    return max(1, int(np.floor(duration * FEATURE_FPS / FEAT_STRIDE)))


def reference_temporal_length_is_compatible(calculated, reference):
    """Allow the anchor encoder's single trailing partial-stride token."""
    return reference in (calculated, calculated + 1)


def sampled_source_indices(token_index, input_frames, source_fps, frame_count):
    if token_index < 0 or input_frames <= 0:
        raise ValueError("token index must be non-negative and input frames positive")
    start = token_index * FEAT_STRIDE
    virtual = np.linspace(
        start,
        start + RECEPTIVE_FIELD_FRAMES - 1,
        input_frames,
        dtype=np.float64,
    )
    source = np.rint(virtual * float(source_fps) / FEATURE_FPS).astype(np.int64)
    return np.clip(source, 0, frame_count - 1)


def expected_centers_seconds(length):
    return (
        np.arange(length, dtype=np.float64) * FEAT_STRIDE
        + 0.5 * RECEPTIVE_FIELD_FRAMES
    ) / FEATURE_FPS


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def validate_feature_file(path, model_name, expected_length, expected_protocol):
    spec = MODEL_SPECS[model_name]
    try:
        with np.load(path) as payload:
            features = payload["feats"]
            protocol = str(payload["protocol_sha256"].item())
            valid = (
                features.shape == (expected_length, spec["output_dim"])
                and np.isfinite(features).all()
                and np.linalg.norm(features, axis=1).min() > 0
                and (
                    expected_length == 1
                    or float(features.astype(np.float32).std(axis=0).mean()) > 1e-6
                )
                and protocol == expected_protocol
            )
    except (OSError, KeyError, ValueError):
        return False
    return bool(valid)


def read_protocol(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.pop("protocol_sha256", None)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported RQ2 feature protocol schema")
    if payload.get("model_name") not in MODEL_SPECS:
        raise ValueError("unknown RQ2 feature model")
    if signature != protocol_sha256(payload):
        raise ValueError("RQ2 feature protocol hash mismatch")
    payload["protocol_sha256"] = signature
    payload["path"] = str(path)
    return payload


def validate_training_protocol(path, feature_folder, input_dim):
    payload = read_protocol(path)
    spec = MODEL_SPECS[payload["model_name"]]
    if not spec["formal_eligible"]:
        raise ValueError(
            "feature interface is not eligible for the formal RQ2 table: {}".format(
                spec["interface"]
            )
        )
    expected_dim = spec["output_dim"]
    if int(input_dim) != expected_dim:
        raise ValueError(
            "feature input_dim {} does not match protocol {}".format(
                input_dim, expected_dim
            )
        )
    protocol_parent = Path(path).resolve().parent
    if Path(feature_folder).resolve() != protocol_parent:
        raise ValueError("feature folder must be the protocol.json parent")
    video_index = protocol_parent / "video_ids.txt"
    if not video_index.is_file():
        raise ValueError("formal RQ2 features require video_ids.txt")
    video_ids = video_index.read_text(encoding="ascii").splitlines()
    if video_ids != sorted(set(video_ids)):
        raise ValueError("RQ2 video index must contain sorted unique IDs")
    if payload.get("video_selection") != "shift_registry_and_annotation":
        raise ValueError("formal RQ2 features must be selected from the shift registry")
    selection = payload.get("video_selection_provenance")
    if not isinstance(selection, dict):
        raise ValueError("formal RQ2 features lack video-selection provenance")
    if not re.fullmatch(r"[0-9a-f]{64}", str(selection.get("registry_sha256", ""))):
        raise ValueError("invalid RQ2 shift-registry hash")
    configs = selection.get("configs")
    if not isinstance(configs, dict) or list(configs) != [
        "S1", "S2", "S3", "S4", "S5", "S6"
    ]:
        raise ValueError("RQ2 selection provenance must bind S1--S6")
    for shift_id, config in configs.items():
        if (
            not isinstance(config, dict)
            or not isinstance(config.get("path"), str)
            or not config["path"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(config.get("sha256", "")))
        ):
            raise ValueError("invalid {} config provenance".format(shift_id))
    annotation = selection.get("annotation")
    if (
        not isinstance(annotation, dict)
        or not isinstance(annotation.get("path"), str)
        or not annotation["path"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(annotation.get("sha256", "")))
    ):
        raise ValueError("invalid RQ2 annotation provenance")
    if payload.get("video_set_count") != len(video_ids):
        raise ValueError("RQ2 video count does not match protocol")
    if payload.get("video_set_sha256") != video_set_sha256(video_ids):
        raise ValueError("RQ2 video-set hash does not match protocol")
    return payload
