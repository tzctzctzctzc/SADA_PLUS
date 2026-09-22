"""Entry-point checks for the frozen full-video RQ2 SO/UDA experiment."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tools.audit_rq2_vlm_benchmark import audit_configs, audit_features
from tools.rq2_dense_protocol import sha256_file, sha256_normalized_text

FROZEN_VIDEO_COUNT = 626
FROZEN_VIDEO_SET_SHA256 = "8308e5e6cd2076fa68247aaf71dc44d6a9ed87ae4a116b3c4971e02598cc54a2"


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
    provenance = protocol.get("provenance", {})
    if not provenance.get("extractor_sha256") or not provenance.get("packages"):
        raise ValueError("formal RQ2 feature protocol lacks extraction provenance")
    if (protocol["video_set_count"] != FROZEN_VIDEO_COUNT
            or protocol["video_set_sha256"] != FROZEN_VIDEO_SET_SHA256):
        raise ValueError("formal RQ2 requires the complete frozen 626-video set")
    result = audit_features(cfg["dataset"]["feat_folder"], reference_root)
    if result["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("feature audit and training protocol disagree")
    return protocol


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
