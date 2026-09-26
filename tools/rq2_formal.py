"""Entry-point checks for the frozen full-video RQ2 SO/UDA experiment."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from tools.audit_rq2_vlm_benchmark import audit_configs, audit_features
from tools.rq2_dense_protocol import sha256_file, sha256_normalized_text

FROZEN_VIDEO_COUNT = 626
FROZEN_VIDEO_SET_SHA256 = "8308e5e6cd2076fa68247aaf71dc44d6a9ed87ae4a116b3c4971e02598cc54a2"
RUN_SIGNATURE_SCHEMA = "rq2-formal-run-v1"
TRAINING_BUDGET_PROTOCOL = "max-source-target-batches-per-epoch-v2"
SOURCE_OPTIMIZATION_PROTOCOL = "sada-matched-two-phase-adamw-v1"
RNG_PROTOCOL = "independent-source-target-generators-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40,64}$")


def build_epoch_budget(cfg):
    # Read only split membership from the already frozen annotation file.
    # SO never loads target features or uses target action annotations.
    with Path(cfg["dataset"]["json_file"]).open(encoding="utf-8") as stream:
        database = json.load(stream)["database"]
    batch_size = int(cfg["loader"]["batch_size"])
    counts = {
        domain: sum(video["subset"].lower() in cfg[key] for video in database.values())
        for domain, key in (("source", "src_train_split"), ("target", "tgt_train_split"))
    }
    source_batches = counts["source"] // batch_size
    target_batches = counts["target"] // batch_size
    if min(source_batches, target_batches) < 1:
        raise ValueError("formal RQ2 needs nonempty source and target batches")
    return {
        "source_videos": counts["source"], "target_videos": counts["target"],
        "source_batches": source_batches, "target_batches": target_batches,
        "iterations_per_epoch": max(source_batches, target_batches),
        "batch_size": batch_size, "drop_last": True,
    }


def build_target_manifest(annotation_path, target_splits, output_path):
    with Path(annotation_path).open(encoding="utf-8") as stream:
        database = json.load(stream)["database"]
    splits = set(target_splits)
    selected = {
        video_id: {key: video[key] for key in ("subset", "fps", "duration") if key in video}
        for video_id, video in sorted(database.items())
        if video["subset"].lower() in splits
    }
    if not selected:
        raise ValueError("target manifest has no videos for the selected splits")
    payload = {
        "schema": "rq2-unlabeled-target-v1",
        "source_annotation_sha256": sha256_file(annotation_path),
        "target_splits": sorted(splits),
        "database": selected,
    }
    Path(output_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return len(selected)


def check_target_manifest(path, target_splits, annotation_path=None):
    with Path(path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema") != "rq2-unlabeled-target-v1":
        raise ValueError("UDA requires a generated RQ2 unlabeled target manifest")
    database = payload.get("database", {})
    if payload.get("target_splits") != sorted(set(target_splits)) or not database or any(
        set(video) - {"subset", "fps", "duration"}
        or video.get("subset", "").lower() not in set(target_splits)
        for video in database.values()
    ):
        raise ValueError("target manifest contains labels, unknown fields, or incorrect splits")
    if annotation_path and payload.get("source_annotation_sha256") != sha256_file(annotation_path):
        raise ValueError("target manifest was generated from different annotations")
    if annotation_path:
        with Path(annotation_path).open(encoding="utf-8") as stream:
            annotated = json.load(stream)["database"]
        splits = set(target_splits)
        expected = {
            video_id: {
                key: video[key]
                for key in ("subset", "fps", "duration")
                if key in video
            }
            for video_id, video in sorted(annotated.items())
            if video["subset"].lower() in splits
        }
        if database != expected:
            raise ValueError(
                "target manifest is incomplete or its unlabeled metadata was changed"
            )
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path), "videos": len(database)}


def check_formal_config(cfg, regime, manifest_path=None):
    if regime not in ("so", "uda"):
        raise ValueError("RQ2 regime must be so or uda")
    if cfg["dataset_name"] != "epic" or cfg["dataset"].get("action_subsampling") != "10_majority_classes":
        raise ValueError("formal RQ2 requires EPIC ten-class TAL")
    supervised = cfg.get("diagnostic", {}).get("supervised_upper_bound", "off")
    if supervised != ("source_only" if regime == "so" else "off"):
        raise ValueError("RQ2 regime and supervised training mode disagree")
    diag = cfg.get("diagnostic", {})
    forbidden = ("oracle_target_task", "oracle_target_regression", "oracle_target_distillation",
                 "oracle_routing", "oracle_forgetting", "binary_band_oracle")
    if any(diag.get(key, "off") != "off" for key in forbidden):
        raise ValueError("formal RQ2 forbids target GT and oracle diagnostics")
    if any(cfg.get(key, {}).get("enabled", False) for key in
           ("stage1", "vlm_routing", "vlm_target_adapter", "vlm_relation", "validity_head")):
        raise ValueError("formal RQ2 forbids extra diagnostic branches")
    if cfg.get("proposal_semantic", {}).get("mode", "off") != "off":
        raise ValueError("formal RQ2 forbids proposal semantic diagnostics")
    if cfg.get("transport_alignment", {}).get("mode", "off") != "off":
        raise ValueError("formal RQ2 forbids transport diagnostics")
    if cfg.get("gradient_harmonization", {}).get("mode", "off") != "off":
        raise ValueError("formal RQ2 forbids gradient harmonization diagnostics")
    if diag.get("no_conditional_update", False) or diag.get("selective_alignment", "off") != "off":
        raise ValueError("formal RQ2 requires the original SADA alignment objective")
    required_diagnostics = {
        "stop_after_epoch": None,
        "alignment_levels": None,
        "target_alignment": "both",
        "class_support": False,
        "action_alignment_levels": None,
        "background_alignment_levels": None,
        "conditional_paired_support": False,
        "conditional_paired_fixed_denominator": False,
    }
    for key, expected in required_diagnostics.items():
        if diag.get(key, expected) != expected:
            raise ValueError(
                "formal RQ2 requires diagnostic.{}={!r}".format(key, expected)
            )
    known_diagnostics = {
        "oracle_routing", "oracle_forgetting", "oracle_forgetting_budget",
        "binary_band_oracle", "binary_band_negative_route",
        "binary_band_oracle_levels", "oracle_target_regression",
        "oracle_target_regression_weight", "target_regression_levels",
        "oracle_target_distillation", "oracle_target_distillation_budget",
        "target_distillation_levels", "oracle_target_task",
        "supervised_upper_bound", "stop_after_epoch", "no_conditional_update",
        "alignment_levels", "target_alignment", "class_support",
        "action_alignment_levels", "background_alignment_levels",
        "selective_alignment", "conditional_paired_support",
        "conditional_paired_fixed_denominator",
    }
    unknown = sorted(set(diag) - known_diagnostics)
    if unknown:
        raise ValueError("formal RQ2 has unknown diagnostic fields: {}".format(unknown))
    train_cfg = cfg.get("train_cfg", {})
    if train_cfg.get("evaluate_student", False):
        raise ValueError("formal RQ2 selects and evaluates EMA weights")
    if train_cfg.get("test_target_val_only", False):
        raise ValueError("formal RQ2 forbids diagnostic target-only evaluation")
    if regime == "so" and manifest_path:
        raise ValueError("source-only must not receive a target manifest")
    if regime == "uda":
        if not manifest_path:
            raise ValueError("UDA requires --rq2-target-manifest")
        return check_target_manifest(
            manifest_path, cfg["tgt_train_split"], cfg["dataset"]["json_file"]
        )
    return None


def check_formal_features(cfg, protocol_path, reference_root):
    from tools.rq2_dense_protocol import validate_training_protocol

    if not protocol_path:
        raise ValueError("VLM RQ2 requires --feature-protocol")
    protocol = validate_training_protocol(
        protocol_path, cfg["dataset"]["feat_folder"], cfg["dataset"]["input_dim"]
    )
    _check_feature_provenance(protocol)
    if (protocol["video_set_count"] != FROZEN_VIDEO_COUNT
            or protocol["video_set_sha256"] != FROZEN_VIDEO_SET_SHA256):
        raise ValueError("formal RQ2 requires the complete frozen 626-video set")
    result = audit_features(cfg["dataset"]["feat_folder"], reference_root)
    if result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("feature audit and training protocol disagree")
    return protocol


def _check_feature_provenance(protocol):
    provenance = protocol.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("formal RQ2 feature provenance must be a mapping")
    if not _SHA256.fullmatch(str(provenance.get("extractor_sha256", ""))):
        raise ValueError("formal RQ2 feature protocol lacks extractor SHA256")
    if not isinstance(provenance.get("batch_tokens"), int) or provenance["batch_tokens"] <= 0:
        raise ValueError("formal RQ2 feature protocol lacks a valid batch_tokens")
    for key in ("device_map", "attention", "dtype"):
        if not isinstance(provenance.get(key), str) or not provenance[key]:
            raise ValueError("formal RQ2 feature protocol lacks {} provenance".format(key))
    packages = provenance.get("packages")
    if (
        not isinstance(packages, dict)
        or not packages
        or any(not isinstance(value, str) or not value for value in packages.values())
    ):
        raise ValueError("formal RQ2 feature package provenance is incomplete")
    for key in ("repository", "external_repository"):
        repository = provenance.get(key)
        if not isinstance(repository, dict):
            raise ValueError("formal RQ2 feature protocol lacks {} provenance".format(key))
        revision = repository.get("revision")
        diff_hash = repository.get("tracked_diff_sha256")
        if (
            not isinstance(revision, str)
            or not _GIT_REVISION.fullmatch(revision)
            or not _SHA256.fullmatch(str(diff_hash or ""))
        ):
            raise ValueError("formal RQ2 {} is not reproducibly identified".format(key))
    checkpoints = protocol.get("checkpoint_sha256")
    if (
        not isinstance(checkpoints, dict)
        or not checkpoints
        or any(
            not isinstance(name, str)
            or not name
            or not _SHA256.fullmatch(str(digest))
            for name, digest in checkpoints.items()
        )
    ):
        raise ValueError("formal RQ2 checkpoint provenance is incomplete")


def _runtime_config_payload(cfg):
    payload = deepcopy(getattr(cfg, "config", cfg))
    for key in ("unique_id", "name", "gpu", "config_path"):
        payload.pop(key, None)
    formal = payload.get("rq2_formal")
    if isinstance(formal, dict):
        formal.pop("run_signature", None)
    payload.get("dataset", {}).pop("stage1_context", None)
    payload.get("train_cfg", {}).pop("head_empty_cls", None)
    benchmark = payload.get("benchmark_audit")
    if isinstance(benchmark, dict):
        benchmark.pop("training_initialization", None)
        benchmark.pop("initialization_inputs", None)
    return payload


def build_formal_run_signature(cfg):
    payload = _runtime_config_payload(cfg)
    formal = payload.get("rq2_formal", {})
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("ascii")
    feature = payload.get("rq2_feature_protocol") or {}
    manifest = formal.get("target_manifest") or {}
    return {
        "schema": RUN_SIGNATURE_SCHEMA,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "regime": formal.get("regime"),
        "shift": formal.get("shift"),
        "seed": int(payload.get("seed", payload.get("init_rand_seed", -1))),
        "feature_protocol_sha256": feature.get("protocol_sha256"),
        "target_manifest_sha256": (
            manifest.get("sha256") if isinstance(manifest, dict) else None
        ),
        "training_budget": formal.get("training_budget"),
        "rng_protocol": formal.get("rng_protocol"),
    }


def validate_formal_runtime(cfg, expected_signature=None):
    formal = cfg.get("rq2_formal", {})
    if formal.get("training_budget") != TRAINING_BUDGET_PROTOCOL:
        raise ValueError("formal RQ2 training budget protocol differs")
    if formal.get("source_optimization") != SOURCE_OPTIMIZATION_PROTOCOL:
        raise ValueError("formal RQ2 source optimization protocol differs")
    if formal.get("epoch_budget") != build_epoch_budget(cfg):
        raise ValueError("formal RQ2 epoch budget differs from frozen video counts")
    if formal.get("rng_protocol") != RNG_PROTOCOL:
        raise ValueError("formal RQ2 RNG protocol differs")
    manifest = formal.get("target_manifest")
    manifest_path = manifest.get("path") if isinstance(manifest, dict) else manifest
    checked = check_formal_config(cfg, formal.get("regime"), manifest_path)
    if checked and isinstance(manifest, dict):
        if checked["sha256"] != manifest.get("sha256") or checked["videos"] != manifest.get("videos"):
            raise ValueError("formal RQ2 target manifest changed after audit")
    actual = build_formal_run_signature(cfg)
    expected = expected_signature or formal.get("run_signature")
    if expected != actual:
        raise ValueError("formal RQ2 runtime configuration changed after audit")
    return actual


def validate_formal_resume(checkpoint, current_signature):
    saved = checkpoint.get("rq2_run_signature") if isinstance(checkpoint, dict) else None
    if saved != current_signature:
        raise ValueError("formal RQ2 resume checkpoint has a different run signature")


def check_slowfast_features(feature_root, registry_path):
    from tools.extract_rq2_dense_vlm_features import required_video_ids_from_registry

    video_ids = required_video_ids_from_registry(registry_path)
    if len(video_ids) != FROZEN_VIDEO_COUNT:
        raise ValueError("SlowFast feature set is not the frozen 626-video set")
    missing = []
    invalid = []
    for video_id in video_ids:
        path = Path(feature_root) / (video_id + ".npz")
        if not path.is_file():
            missing.append(video_id)
            continue
        try:
            with np.load(path) as payload:
                feats = payload["feats"]
                if feats.ndim != 2 or feats.shape[1] != 2304 or not np.isfinite(feats).all():
                    invalid.append(video_id)
        except (OSError, KeyError, ValueError):
            invalid.append(video_id)
    if missing or invalid:
        raise RuntimeError(
            "SlowFast feature audit failed: missing={} invalid={}".format(
                missing[:20], invalid[:20]
            )
        )
    return {"videos": len(video_ids), "video_set_sha256": FROZEN_VIDEO_SET_SHA256}


def _compare_frozen_values(frozen, candidate, prefix=()):
    allowed = {
        ("init_rand_seed",), ("dataset", "feat_folder"), ("dataset", "input_dim")
    }
    failures = []
    for key, value in frozen.items():
        path = prefix + (str(key),)
        if path in allowed:
            continue
        actual = candidate.get(key) if isinstance(candidate, dict) else None
        if isinstance(value, dict):
            failures.extend(_compare_frozen_values(value, actual or {}, path))
        elif actual != value:
            failures.append("{}: {!r} != {!r}".format(".".join(path), actual, value))
    generated = {
        ("dataset",): {"stage1_context", "crop_alignment"},
        ("model",): {"input_dim", "num_classes", "test_cfg", "max_seq_len"},
        ("train_cfg",): {"head_empty_cls", "evaluate_student", "test_target_val_only"},
    }
    behavior_roots = {"dataset", "model", "opt", "loader", "train_cfg", "test_cfg"}
    if prefix and prefix[0] in behavior_roots:
        allowed_extras = generated.get(prefix, set())
        extras = set(candidate or {}) - set(frozen) - allowed_extras
        failures.extend(
            "{}: unexpected runtime field".format(".".join(prefix + (str(key),)))
            for key in sorted(extras)
        )
    if prefix == ("dataset",) and "crop_alignment" not in frozen:
        if candidate.get("crop_alignment", 1) != 1:
            failures.append("dataset.crop_alignment must retain the default value 1")
    return failures


def check_frozen_runtime_values(frozen, candidate):
    failures = _compare_frozen_values(frozen, candidate)
    if failures:
        raise ValueError("runtime config differs from frozen shift:\n" + "\n".join(failures))


def check_registry_config(config_path, registry_path, shift_id, candidate_config=None):
    with Path(registry_path).open(encoding="utf-8") as stream:
        registry = yaml.safe_load(stream)
    entry = registry["shifts"][shift_id]
    base = (Path(registry_path).resolve().parent / registry["repository_root"]).resolve()
    frozen_path = base / entry["config"]
    if sha256_normalized_text(frozen_path) != entry["config_sha256"]:
        raise ValueError("frozen shift configuration changed")
    with frozen_path.open(encoding="utf-8") as stream:
        frozen = yaml.safe_load(stream)
    with Path(config_path).open(encoding="utf-8") as stream:
        candidate = yaml.safe_load(stream)
    for key in ("src_train_split", "src_val_split", "tgt_train_split", "tgt_val_split"):
        if frozen[key] != candidate[key]:
            raise ValueError("{} differs from frozen shift {}".format(key, shift_id))
    audit_configs([frozen_path, config_path])
    if candidate_config is not None:
        check_frozen_runtime_values(frozen, candidate_config)
    return entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    count = build_target_manifest(
        config["dataset"]["json_file"], config["tgt_train_split"], args.output
    )
    print("wrote {} unlabeled target videos to {}".format(count, args.output))
