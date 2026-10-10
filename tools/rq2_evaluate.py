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
    build_formal_run_signature, check_formal_config, check_formal_features,
    check_frozen_runtime_values, check_slowfast_features,
    validate_formal_resume,
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
        if formal.get('data_provenance'):
            from tools.rq2_formal import build_data_provenance
            if formal['data_provenance'] != build_data_provenance(saved_config):
                raise ValueError('checkpoint annotation or split membership provenance differs')
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


def selected_checkpoint_state(checkpoint, config=None, require_provenance=False):
    selection = checkpoint.get("selection_provenance") if isinstance(checkpoint, dict) else None
    if selection is not None:
        required = {"split", "metric", "metric_value", "selected_epoch", "state_key"}
        if not isinstance(selection, dict) or set(selection) != required:
            raise ValueError("checkpoint has incomplete selection provenance")
        if selection["split"] != "source_validation":
            raise ValueError("checkpoint was not selected on source validation")
        if selection["metric"] != "validation_loss":
            raise ValueError("checkpoint used an unexpected selection metric")
        if int(selection["selected_epoch"]) != int(checkpoint.get("epoch", -1)):
            raise ValueError("best checkpoint epoch differs from selected epoch")
        key = selection["state_key"]
        if require_provenance and key != "ema_state_dict":
            raise ValueError("formal RQ2 checkpoint must select EMA weights")
    elif require_provenance:
        raise ValueError("formal RQ2 checkpoint lacks selection provenance")
    else:
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
        if checkpoint_schema == "formal":
            saved_signature = saved_config.get("rq2_formal", {}).get("run_signature")
            if saved_signature != build_formal_run_signature(saved_config):
                raise ValueError("checkpoint saved configuration violates its run signature")
            validate_formal_resume(checkpoint, saved_signature)
        from tools.rq3_protocol import FULL_PARTICIPATION_REVISION
        if saved_config.get('rq3_auxiliary', {}).get('schema') == FULL_PARTICIPATION_REVISION:
            from tools.rq3_full_participation import validate_runtime
            validate_runtime(saved_config, 'uda')
        if saved_config.get('rq3_auxiliary', {}).get('acceleration') is not None:
            from tools.rq3_accelerated import evaluation_audit_config
            audited_config = evaluation_audit_config(audited_config)
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
    state_key, state = selected_checkpoint_state(
        checkpoint, cfg, require_provenance=(checkpoint_schema == "formal")
    )
    from libs.models.SADA import make_meta_arch

    from libs.utils.parse_config import ConfigParser

    if saved_config.get("rq3_auxiliary"):
        # 推理时保留buffer/配置身份以strict加载；forward不使用教师或辅助损失。
        cfg["rq3_auxiliary"] = deepcopy(saved_config["rq3_auxiliary"])
    model_cfg = ConfigParser(cfg, save=False)
    from tools.rq3_gpu import guard_before_cuda
    guard_before_cuda()
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
        "selection_provenance": checkpoint.get("selection_provenance"),
        "seed": seed,
        "shift": shift_id,
        "config_sha256": sha256_normalized_text(config_path),
        "feature_protocol_sha256": feature_hash,
        "annotation_sha256": sha256_file(cfg["dataset"]["json_file"]),
        "training_protocol": {
            key: saved_config.get("rq2_formal", {}).get(key)
            for key in ("training_budget", "source_optimization", "epoch_budget", "rng_protocol", "data_provenance")
        },
    }
    summaries = {}
    if saved_config.get("rq3_auxiliary"):
        metadata["rq3_auxiliary"] = saved_config["rq3_auxiliary"]
    if saved_config.get('rq4_boundary'):
        # 仅保存训练身份；推理不加载边界教师或附加任何预测。
        metadata['rq4_boundary'] = saved_config['rq4_boundary']
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


def _load_evaluation(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "domain", "split", "seed", "shift", "config_sha256",
        "feature_protocol_sha256", "annotation_sha256", "map_0.1_to_0.5",
        "map_by_tiou", "ap_by_class", "predictions", "checkpoint_sha256",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("evaluation result lacks fields: {}".format(missing))
    return path, payload


def _prediction_frame(predictions):
    import pandas as pd

    return pd.DataFrame(
        predictions,
        columns=("video-id", "t-start", "t-end", "label", "score"),
    )


def _bootstrap_map(evaluator, predictions, sampled_video_ids):
    import pandas as pd

    ground_truth_parts = []
    prediction_parts = []
    prediction_frame = _prediction_frame(predictions)
    for draw, video_id in enumerate(sampled_video_ids):
        replacement_id = "{}:{}".format(draw, video_id)
        gt = evaluator.ground_truth[
            evaluator.ground_truth["video-id"] == video_id
        ].copy()
        gt["video-id"] = replacement_id
        ground_truth_parts.append(gt)
        pred = prediction_frame[prediction_frame["video-id"] == video_id].copy()
        pred["video-id"] = replacement_id
        prediction_parts.append(pred)
    sampled_evaluator = deepcopy(evaluator)
    sampled_evaluator.num_workers = 1
    sampled_evaluator.ground_truth = pd.concat(ground_truth_parts, ignore_index=True)
    sampled_predictions = pd.concat(prediction_parts, ignore_index=True)
    results = {
        key: sampled_predictions[key].to_numpy()
        for key in ("video-id", "t-start", "t-end", "label", "score")
    }
    return float(sampled_evaluator.evaluate(results, verbose=False)[1])


def _paired_bootstrap_ci(
    so_payload, uda_payload, annotation_path, repeats, seed
):
    evaluator = ANETdetection(
        annotation_path,
        so_payload["split"],
        num_workers=1,
        action_subsampling="10_majority_classes",
    )
    video_ids = sorted(set(evaluator.ground_truth["video-id"]))
    if not video_ids:
        raise ValueError("bootstrap split has no ground-truth videos")
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        sampled = rng.choice(video_ids, size=len(video_ids), replace=True).tolist()
        differences.append(
            _bootstrap_map(evaluator, uda_payload["predictions"], sampled)
            - _bootstrap_map(evaluator, so_payload["predictions"], sampled)
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "unit": "video_id",
        "repeats": int(repeats),
        "seed": int(seed),
        "videos": len(video_ids),
        "ci_95": [float(low), float(high)],
    }


def summarize_so_uda(
    so_source_path,
    so_target_path,
    uda_source_path,
    uda_target_path,
    annotation_path,
    output_path,
    bootstrap_repeats=2000,
    bootstrap_seed=20260922,
):
    loaded = [
        _load_evaluation(path)
        for path in (
            so_source_path, so_target_path, uda_source_path, uda_target_path
        )
    ]
    paths = [item[0] for item in loaded]
    so_source, so_target, uda_source, uda_target = [item[1] for item in loaded]
    if [item["domain"] for item in (so_source, so_target, uda_source, uda_target)] != [
        "source", "target", "source", "target"
    ]:
        raise ValueError("SO/UDA result domains are not source,target pairs")
    comparison_keys = (
        "seed", "shift", "config_sha256", "feature_protocol_sha256",
        "annotation_sha256",
    )
    for key in comparison_keys:
        values = {json.dumps(item[key], sort_keys=True) for item in (
            so_source, so_target, uda_source, uda_target
        )}
        if len(values) != 1:
            raise ValueError("SO/UDA result {} differs".format(key))
    training_protocols = {
        json.dumps(item.get("training_protocol"), sort_keys=True)
        for item in (so_source, so_target, uda_source, uda_target)
    }
    if len(training_protocols) != 1:
        raise ValueError("SO/UDA training protocols differ; do not mix old and new runs")
    for source, target in ((so_source, so_target), (uda_source, uda_target)):
        if source["checkpoint_sha256"] != target["checkpoint_sha256"]:
            raise ValueError("source and target results use different checkpoints")
    s_so = float(so_source["map_0.1_to_0.5"])
    t_so = float(so_target["map_0.1_to_0.5"])
    s_uda = float(uda_source["map_0.1_to_0.5"])
    t_uda = float(uda_target["map_0.1_to_0.5"])
    gap_so = t_so - s_so
    gap_uda = t_uda - s_uda
    if s_so == 0 or s_uda == 0:
        raise ValueError("retention is undefined when source mAP is zero")
    metrics = {
        "S_SO": s_so,
        "T_SO": t_so,
        "S_UDA": s_uda,
        "T_UDA": t_uda,
        "Gap_SO": gap_so,
        "Retention_SO": 100.0 * t_so / s_so,
        "Gap_UDA": gap_uda,
        "Retention_UDA": 100.0 * t_uda / s_uda,
        "Gain_target": t_uda - t_so,
        "Change_source": s_uda - s_so,
        "Recovery_gap": gap_uda - gap_so,
    }
    result = {
        "schema": "rq2-so-uda-pair-v1",
        "seed": so_source["seed"],
        "shift": so_source["shift"],
        "feature_protocol_sha256": so_source["feature_protocol_sha256"],
        "training_protocol": so_source.get("training_protocol"),
        "inputs": {
            label: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for label, path in zip(
                ("SO_source", "SO_target", "UDA_source", "UDA_target"), paths
            )
        },
        "metrics": metrics,
        "map_by_tiou": {
            "SO_source": so_source["map_by_tiou"],
            "SO_target": so_target["map_by_tiou"],
            "UDA_source": uda_source["map_by_tiou"],
            "UDA_target": uda_target["map_by_tiou"],
        },
        "ap_by_class": {
            "SO_source": so_source["ap_by_class"],
            "SO_target": so_target["ap_by_class"],
            "UDA_source": uda_source["ap_by_class"],
            "UDA_target": uda_target["ap_by_class"],
        },
        "bootstrap": {
            "Gain_target": _paired_bootstrap_ci(
                so_target, uda_target, annotation_path,
                bootstrap_repeats, bootstrap_seed,
            ),
            "Change_source": _paired_bootstrap_ci(
                so_source, uda_source, annotation_path,
                bootstrap_repeats, bootstrap_seed + 1,
            ),
        },
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("RQ2 pair summary already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def _aggregate_nested(records, field):
    outer_keys = records[0][field].keys()
    return {
        outer: {
            inner: _mean_std([
                float(record[field][outer][inner]) for record in records
            ])
            for inner in records[0][field][outer]
        }
        for outer in outer_keys
    }


def aggregate_pair_summaries(pair_paths, output_path, require_complete=True):
    paths = [Path(path) for path in pair_paths]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not records or any(record.get("schema") != "rq2-so-uda-pair-v1" for record in records):
        raise ValueError("aggregate inputs must be RQ2 SO/UDA pair summaries")
    protocol_hashes = {
        record.get("feature_protocol_sha256") for record in records
    }
    if len(protocol_hashes) != 1:
        raise ValueError("aggregate inputs use different feature protocols")
    # Counts differ by shift; the budget and optimization policy must not.
    training_policies = {
        json.dumps({
            "annotation_sha256": ((record.get("training_protocol") or {}).get("data_provenance") or {}).get("annotation_sha256"),
            **{
            key: (record.get("training_protocol") or {}).get(key)
            for key in ("training_budget", "source_optimization", "rng_protocol")
            },
        }, sort_keys=True)
        for record in records
    }
    if len(training_policies) != 1:
        raise ValueError("aggregate inputs use different training protocols")
    keys = [(record["shift"], int(record["seed"])) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("aggregate inputs repeat a shift/seed pair")
    expected_shifts = {"S1", "S2", "S3", "S4", "S5", "S6"}
    if require_complete and set(shift for shift, _ in keys) != expected_shifts:
        raise ValueError("formal aggregate requires all six shifts")
    grouped = {}
    for record in records:
        grouped.setdefault(record["shift"], []).append(record)
    if require_complete:
        for shift, shift_records in grouped.items():
            if {int(record["seed"]) for record in shift_records} != {6, 7, 8}:
                raise ValueError("{} does not contain seeds 6/7/8".format(shift))
    per_shift = {}
    metric_names = list(records[0]["metrics"])
    for shift, shift_records in sorted(grouped.items()):
        per_shift[shift] = {
            "seeds": sorted(int(record["seed"]) for record in shift_records),
            "metrics": {
                metric: _mean_std([
                    float(record["metrics"][metric]) for record in shift_records
                ])
                for metric in metric_names
            },
            "map_by_tiou": _aggregate_nested(shift_records, "map_by_tiou"),
            "ap_by_class": _aggregate_nested(shift_records, "ap_by_class"),
        }
    macro = {
        metric: _mean_std([
            per_shift[shift]["metrics"][metric]["mean"] for shift in per_shift
        ])
        for metric in metric_names
    }
    result = {
        "schema": "rq2-so-uda-aggregate-v1",
        "feature_protocol_sha256": next(iter(protocol_hashes)),
        "training_policy": json.loads(next(iter(training_policies))),
        "inputs": {
            str(path.resolve()): sha256_file(path) for path in paths
        },
        "per_shift": per_shift,
        "macro_over_shifts": macro,
        "worst_shift": {
            "T_SO": min(per_shift, key=lambda shift: per_shift[shift]["metrics"]["T_SO"]["mean"]),
            "T_UDA": min(per_shift, key=lambda shift: per_shift[shift]["metrics"]["T_UDA"]["mean"]),
        },
    }
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("RQ2 aggregate summary already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize formal RQ2 results")
    subparsers = parser.add_subparsers(dest="command", required=True)
    pair = subparsers.add_parser("pair")
    pair.add_argument("--so-source", required=True)
    pair.add_argument("--so-target", required=True)
    pair.add_argument("--uda-source", required=True)
    pair.add_argument("--uda-target", required=True)
    pair.add_argument("--annotations", required=True)
    pair.add_argument("--output", required=True)
    pair.add_argument("--bootstrap-repeats", type=int, default=2000)
    pair.add_argument("--bootstrap-seed", type=int, default=20260922)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--pairs", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--allow-partial", action="store_true")
    cli = parser.parse_args()
    if cli.command == "pair":
        summarize_so_uda(
            cli.so_source, cli.so_target, cli.uda_source, cli.uda_target,
            cli.annotations, cli.output, cli.bootstrap_repeats,
            cli.bootstrap_seed,
        )
    else:
        aggregate_pair_summaries(
            cli.pairs, cli.output, require_complete=not cli.allow_partial
        )
