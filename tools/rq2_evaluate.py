"""Offline, source-selected full-video TAL evaluation for RQ2."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from libs.datasets import make_data_loader, make_dataset
from libs.evaluation import ANETdetection
from tools.rq2_dense_protocol import sha256_file, sha256_normalized_text
from tools.rq2_formal import (
    check_formal_config, check_formal_features, check_frozen_runtime_values,
    check_slowfast_features,
)


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def checkpoint_config_for_audit(saved_config, frozen_config):
    formal = saved_config.get("rq2_formal")
    if formal:
        manifest = formal.get("target_manifest")
        manifest_path = manifest.get("path") if isinstance(manifest, dict) else manifest
        checked_manifest = check_formal_config(
            saved_config, formal.get("regime"), manifest_path
        )
        if checked_manifest and (
            not isinstance(manifest, dict)
            or checked_manifest["sha256"] != manifest.get("sha256")
            or checked_manifest["videos"] != manifest.get("videos")
        ):
            raise ValueError("checkpoint target manifest provenance differs")
        return saved_config, "formal"
    check_formal_config(saved_config, "so")
    if saved_config.get("train_cfg", {}).get("test_freq") != -1:
        raise ValueError(
            "legacy source-only checkpoint did not disable periodic target evaluation"
        )
    normalized = deepcopy(saved_config)
    normalized["train_cfg"]["test_freq"] = frozen_config["train_cfg"]["test_freq"]
    return normalized, "legacy-source-only"


def selected_checkpoint_state(checkpoint, config):
    use_ema = config["train_cfg"].get("ema_active", False)
    use_student = config["train_cfg"].get("evaluate_student", False)
    key = "ema_state_dict" if use_ema and not use_student else "state_dict"
    if not isinstance(checkpoint, dict) or key not in checkpoint:
        raise ValueError("checkpoint lacks selected {} weights".format(key))
    return key, checkpoint[key]


def evaluate(cfg, ckpt_path, config_path, output_dir, device, feature_protocol=None,
             reference_root="./data/ek100_slowfast/features",
             registry_path="configs/rq2/shift_registry.yaml", shift_id=None):
    if cfg["dataset_name"] != "epic" or cfg["dataset"].get("action_subsampling") != "10_majority_classes":
        raise ValueError("formal RQ2 evaluation requires EPIC ten-class TAL")
    if int(cfg["dataset"]["input_dim"]) != 2304:
        protocol = check_formal_features(cfg, feature_protocol, reference_root)
        feature_hash = protocol["protocol_sha256"]
    else:
        if feature_protocol:
            raise ValueError("SlowFast anchor has no VLM feature protocol")
        feature_hash = None
        check_slowfast_features(cfg["dataset"]["feat_folder"], registry_path)

    checkpoint = load_checkpoint(ckpt_path)
    if Path(ckpt_path).parent.name != "best_ckpt":
        raise ValueError("formal RQ2 evaluates only the source-selected best checkpoint")
    expected_name = "checkpoint-epoch{}.pt".format(int(checkpoint.get("epoch", -1)))
    if Path(ckpt_path).name != expected_name:
        raise ValueError("checkpoint filename and stored epoch disagree")
    saved_config = checkpoint.get("config")
    if saved_config is not None:
        saved_config = getattr(saved_config, "config", saved_config)
        with Path(config_path).open(encoding="utf-8") as stream:
            frozen_config = yaml.safe_load(stream)
        audited_config, checkpoint_schema = checkpoint_config_for_audit(
            saved_config, frozen_config
        )
        check_frozen_runtime_values(frozen_config, audited_config)
        for key in ("src_val_split", "tgt_val_split"):
            if list(saved_config[key]) != list(cfg[key]):
                raise ValueError("evaluation {} differs from checkpoint".format(key))
        if saved_config["dataset"]["input_dim"] != cfg["dataset"]["input_dim"]:
            raise ValueError("evaluation input dimension differs from checkpoint")
        saved_protocol = saved_config.get("rq2_feature_protocol")
        if saved_protocol and saved_protocol["protocol_sha256"] != feature_hash:
            raise ValueError("evaluation feature protocol differs from checkpoint")
        saved_rq2 = saved_config.get("rq2_formal")
        if saved_rq2 and saved_rq2.get("shift") != shift_id:
            raise ValueError("checkpoint RQ2 shift differs from evaluation shift")
        seed = int(saved_config.get("seed", saved_config.get("init_rand_seed", -1)))
        if seed not in (6, 7, 8):
            raise ValueError("checkpoint seed is outside the frozen RQ2 seed set")
    else:
        raise ValueError("formal RQ2 checkpoint has no saved configuration")
    state_key, state = selected_checkpoint_state(checkpoint, cfg)
    from libs.models.SADA import make_meta_arch

    from libs.utils.parse_config import ConfigParser

    model_cfg = ConfigParser(cfg, save=False)
    model = make_meta_arch(cfg["model_name"], model_cfg, device=device).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / (domain + "_val.json") for domain in ("source", "target")]
    if any(path.exists() for path in output_paths):
        raise FileExistsError("RQ2 evaluation output already exists; use a new directory")

    metadata = {
        "checkpoint": str(Path(ckpt_path).resolve()),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_state_key": state_key,
        "checkpoint_schema": checkpoint_schema,
        "seed": seed,
        "shift": shift_id,
        "config_sha256": sha256_normalized_text(config_path),
        "feature_protocol_sha256": feature_hash,
        "annotation_sha256": sha256_file(cfg["dataset"]["json_file"]),
    }
    summaries = {}
    for name, split_key, output_path in zip(
        ("source", "target"), ("src_val_split", "tgt_val_split"), output_paths
    ):
        dataset = make_dataset(cfg["dataset_name"], False, cfg[split_key], **cfg["dataset"])
        loader = make_data_loader(dataset, False, None, 1, cfg["loader"]["num_workers"])
        evaluator = ANETdetection(
            cfg["dataset"]["json_file"], cfg[split_key],
            action_subsampling=cfg["dataset"]["action_subsampling"],
        )
        predictions = []
        with torch.no_grad():
            for batch in loader:
                for video in batch:
                    for key, value in video.items():
                        if isinstance(value, torch.Tensor):
                            video[key] = value.to(device)
                source_output, target_output = model(
                    video_list_src=batch, video_list_tgt=batch, mode="test"
                )
                for video in (source_output if name == "source" else target_output):
                    for segment, label, score in zip(
                        video["segments"], video["labels"], video["scores"]
                    ):
                        predictions.append({
                            "video-id": video["video_id"],
                            "t-start": float(segment[0]),
                            "t-end": float(segment[1]),
                            "label": int(label),
                            "score": float(score),
                        })
        results = {
            key: np.asarray([row[key] for row in predictions])
            for key in ("video-id", "t-start", "t-end", "label", "score")
        }
        maps, average, class_maps, *_ = evaluator.evaluate(results, verbose=False)
        summary = {
            **metadata,
            "domain": name,
            "split": list(cfg[split_key]),
            "videos": len(dataset),
            "map_0.1_to_0.5": float(average),
            "map_by_tiou": {
                str(float(tiou)): float(value)
                for tiou, value in zip(evaluator.tiou_thresholds, maps)
            },
            "ap_by_class": {
                str(int(evaluator.idx_to_activity_index[index])): float(value)
                for index, value in enumerate(class_maps)
            },
            "predictions": predictions,
        }
        output_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
        summaries[name] = {key: value for key, value in summary.items() if key != "predictions"}
    (output_dir / "manifest.json").write_text(json.dumps({
        **metadata,
        "outputs": {path.name: sha256_file(path) for path in output_paths},
    }, sort_keys=True), encoding="utf-8")
    return summaries
