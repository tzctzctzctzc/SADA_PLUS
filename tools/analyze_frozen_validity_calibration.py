"""Diagnose level and video calibration for a frozen-SADA Validity Head."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.anchor_validity import (
    protocol_sha256,
    video_robust_high_precision_threshold,
)
from libs.models.SADA.stage1_calibration import state_dict_sha256
from libs.utils.utils import _update_config, read_yaml
from tools.evaluate_integrated_validity_head import calibrate, collect_split
from tools.train_frozen_sada_validity_head import base_state, load_base_checkpoint


SCHEMA_VERSION = "frozen-validity-calibration-diagnostics-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-selected-per-video", type=int, default=10)
    return parser.parse_args()


def clone_by_ids(dataset, video_ids):
    expected = set(video_ids)
    result = copy.copy(dataset)
    result.data_list = tuple(
        item for item in dataset.data_list if str(item["id"]) in expected
    )
    actual = {str(item["id"]) for item in result.data_list}
    if actual != expected:
        raise ValueError("dataset video IDs differ: missing={}".format(expected - actual))
    return result


def collect_records(model, dataset, artifact, device, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    return collect_split(
        model,
        dataset,
        artifact["alpha"],
        artifact["beta"],
        device,
        artifact["training_protocol"]["evaluation_crops_per_video"],
        return_groups=True,
    )


def decision_metrics(decisions, labels):
    labels = labels.bool()
    true_positive = decisions & labels
    return {
        "anchors": len(labels),
        "valid": int(labels.sum()),
        "selected": int(decisions.sum()),
        "precision": float(true_positive.sum() / decisions.sum().clamp_min(1)),
        "recall": float(true_positive.sum() / labels.sum().clamp_min(1)),
    }


def grouped_metrics(decisions, labels, groups):
    levels = groups["levels"]
    video_ids = groups["video_ids"]
    per_level = {}
    for level in sorted(levels.unique().tolist()):
        mask = levels == level
        per_level[str(level)] = decision_metrics(decisions[mask], labels[mask])
    per_video = {}
    for video_id in sorted(set(video_ids)):
        mask = torch.tensor([value == video_id for value in video_ids])
        per_video[video_id] = decision_metrics(decisions[mask], labels[mask])
    return {"overall": decision_metrics(decisions, labels),
            "per_level": per_level, "per_video": per_video}


def score_diagnostics(scores, labels, groups):
    output = {}
    for level in sorted(groups["levels"].unique().tolist()):
        mask = groups["levels"] == level
        level_scores = scores[mask]
        level_labels = labels[mask].bool()
        row = {"anchors": int(mask.sum()), "valid": int(level_labels.sum())}
        for name, subset in (
            ("positive", level_scores[level_labels]),
            ("negative", level_scores[~level_labels]),
        ):
            if len(subset):
                row[name + "_quantiles"] = [
                    float(value)
                    for value in torch.quantile(
                        subset, torch.tensor([0.5, 0.9, 0.95, 0.99])
                    )
                ]
        output[str(level)] = row
    return output


def global_decisions(scores, threshold, groups):
    del groups
    return scores >= threshold


def level_decisions(scores, thresholds, groups):
    decisions = torch.zeros(len(scores), dtype=torch.bool)
    for level, threshold in thresholds.items():
        mask = groups["levels"] == int(level)
        decisions[mask] = scores[mask] >= threshold
    return decisions


def calibrate_level_thresholds(scores, labels, groups, minimum_precision):
    thresholds = {}
    calibration = {}
    requirement_met = True
    for level in sorted(groups["levels"].unique().tolist()):
        mask = groups["levels"] == level
        met, result = calibrate(scores[mask], labels[mask], minimum_precision)
        thresholds[str(level)] = result["threshold"]
        calibration[str(level)] = {"requirement_met": met, **result}
        requirement_met = requirement_met and met
    return requirement_met, thresholds, calibration


def combine_records(first, second):
    return (
        torch.cat((first[0], second[0])),
        torch.cat((first[1], second[1])),
        {
            "levels": torch.cat((first[3]["levels"], second[3]["levels"])),
            "video_ids": first[3]["video_ids"] + second[3]["video_ids"],
        },
    )


def evaluate_policy(policy, records):
    scores, labels, _, groups = records
    if policy["type"] == "level_thresholds":
        decisions = level_decisions(scores, policy["thresholds"], groups)
    else:
        decisions = global_decisions(scores, policy["threshold"], groups)
    return grouped_metrics(decisions, labels, groups)


def main():
    args = parse_args()
    if args.minimum_selected_per_video <= 0:
        raise ValueError("minimum selected per video must be positive")
    artifact = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    if artifact.get("schema_version") != "frozen-sada-validity-head-v1":
        raise ValueError("unsupported frozen-head artifact")

    cfg = read_yaml(args.config)
    head_design = artifact.get("head_design", {
        "use_level_embedding": True,
        "per_level_output_heads": False,
        "level_balanced_loss": False,
    })
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["crop_alignment"] = artifact["training_protocol"][
        "crop_alignment"
    ]
    cfg["validity_head"] = {
        "enabled": True,
        "loss_weight": 1.0,
        "hidden_dim": 128,
        "fusion_dim": 64,
        "level_embedding_dim": 8,
        "use_level_embedding": head_design["use_level_embedding"],
        "per_level_output_heads": head_design["per_level_output_heads"],
        "level_balanced_loss": head_design["level_balanced_loss"],
        "detach_fpn": True,
        "detach_geometry": True,
    }
    _update_config(cfg)
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    load_base_checkpoint(
        model, artifact["checkpoint"], artifact["weights"], args.device
    )
    head_checkpoint = torch.load(artifact["head_checkpoint"], map_location="cpu")
    model.validity_head.load_state_dict(head_checkpoint["validity_head_state_dict"])
    if state_dict_sha256(base_state(model)) != artifact["base_state_sha256"]:
        raise ValueError("frozen SADA base state differs from artifact")
    if state_dict_sha256(model.validity_head.state_dict()) != artifact[
        "trained_head_sha256"
    ]:
        raise ValueError("trained Validity Head state differs from artifact")
    model.eval()

    source_train = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
    )
    partition = artifact["partition"]
    calibration_dataset = clone_by_ids(
        source_train, partition["calibration_video_ids"]
    )
    train_dataset = clone_by_ids(source_train, partition["train_video_ids"])
    internal_dataset = clone_by_ids(source_train, partition["report_video_ids"])
    external_dataset = clone_by_ids(
        source_val, partition["external_report_video_ids"]
    )
    evaluation_seed = artifact["training_protocol"]["evaluation_seed"]
    train = collect_records(
        model, train_dataset, artifact, args.device, evaluation_seed
    )
    calibration = collect_records(
        model, calibration_dataset, artifact, args.device, evaluation_seed + 1
    )
    internal = collect_records(
        model, internal_dataset, artifact, args.device, evaluation_seed + 2
    )
    external = collect_records(
        model, external_dataset, artifact, args.device, evaluation_seed + 3
    )
    combined_scores, combined_labels, combined_groups = combine_records(
        internal, external
    )
    combined = (combined_scores, combined_labels, {}, combined_groups)

    minimum_precision = artifact["minimum_precision"]
    level_met, level_thresholds, level_calibration = calibrate_level_thresholds(
        calibration[0], calibration[1], calibration[3], minimum_precision
    )
    robust_met, robust_calibration = video_robust_high_precision_threshold(
        calibration[0],
        calibration[1],
        calibration[3]["video_ids"],
        minimum_precision,
        args.minimum_selected_per_video,
    )
    policies = {
        "pooled_global": {
            "type": "global_threshold",
            "requirement_met": artifact["calibration_requirement_met"],
            "threshold": artifact["threshold"],
        },
        "pooled_per_level": {
            "type": "level_thresholds",
            "requirement_met": level_met,
            "thresholds": level_thresholds,
            "calibration_by_level": level_calibration,
        },
        "video_robust_global": {
            "type": "global_threshold",
            "requirement_met": robust_met,
            "minimum_selected_per_video": args.minimum_selected_per_video,
            **robust_calibration,
        },
    }
    for policy in policies.values():
        policy["metrics"] = {
            "train": evaluate_policy(policy, train),
            "calibration": evaluate_policy(policy, calibration),
            "internal_report": evaluate_policy(policy, internal),
            "external_report": evaluate_policy(policy, external),
            "combined_report": evaluate_policy(policy, combined),
        }

    baseline = policies["pooled_global"]["metrics"]
    for split in ("internal_report", "external_report", "combined_report"):
        expected = artifact[split]
        actual = baseline[split]["overall"]
        for metric in ("anchors", "valid", "selected"):
            if actual[metric] != expected[metric]:
                raise ValueError("baseline {} {} does not reproduce".format(split, metric))
        for metric in ("precision", "recall"):
            if abs(actual[metric] - expected[metric]) > 1e-7:
                raise ValueError("baseline {} {} does not reproduce".format(split, metric))

    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact": args.artifact,
        "artifact_sha256": protocol_sha256(artifact),
        "target_gt_usage": "forbidden",
        "vlm_usage": "none",
        "minimum_precision": minimum_precision,
        "score_diagnostics_by_level": {
            "train": score_diagnostics(train[0], train[1], train[3]),
            "calibration": score_diagnostics(
                calibration[0], calibration[1], calibration[3]
            ),
            "combined_report": score_diagnostics(
                combined[0], combined[1], combined[3]
            ),
        },
        "policies": policies,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "policies": {
            name: {
                "requirement_met": row["requirement_met"],
                "calibration": row["metrics"]["calibration"]["overall"],
                "combined_report": row["metrics"]["combined_report"]["overall"],
            }
            for name, row in policies.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
