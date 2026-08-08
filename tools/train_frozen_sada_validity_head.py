"""Train only the Validity Head on a completely frozen SADA model."""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.datasets import make_data_loader, make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.anchor_validity import (
    balanced_probability_loss,
    freeze_sada_except_validity_head,
    partition_frozen_head_source_videos,
    protocol_sha256,
    source_validity_loss,
)
from libs.models.SADA.stage1_calibration import state_dict_sha256
from libs.models.SADA.vlm_routing import file_sha256
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import move_video
from tools.evaluate_integrated_validity_head import (
    calibrate,
    collect_split,
    split_metrics,
)


SCHEMA_VERSION = "frozen-sada-validity-head-v1"
HEAD_VARIANTS = {
    "A": {
        "description": "shared head with level embedding and global loss",
        "use_level_embedding": True,
        "per_level_output_heads": False,
        "level_balanced_loss": False,
    },
    "B": {
        "description": "shared head without explicit level embedding",
        "use_level_embedding": False,
        "per_level_output_heads": False,
        "level_balanced_loss": False,
    },
    "C": {
        "description": "shared head with level embedding and level-balanced loss",
        "use_level_embedding": True,
        "per_level_output_heads": False,
        "level_balanced_loss": True,
    },
    "D": {
        "description": "shared projection with level-balanced per-level outputs",
        "use_level_embedding": False,
        "per_level_output_heads": True,
        "level_balanced_loss": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-crops-per-video", type=int, default=4)
    parser.add_argument("--selection-crops-per-video", type=int, default=4)
    parser.add_argument("--evaluation-crops-per-video", type=int, default=8)
    parser.add_argument("--selection-frequency", type=int, default=2)
    parser.add_argument("--selection-videos", type=int, default=6)
    parser.add_argument("--calibration-videos", type=int, default=7)
    parser.add_argument("--report-videos", type=int, default=7)
    parser.add_argument("--partition-seed", type=int, default=20260809)
    parser.add_argument("--evaluation-seed", type=int, default=20260810)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--crop-alignment", type=int, default=1)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--head-variant", choices=tuple(HEAD_VARIANTS), default="A"
    )
    return parser.parse_args()


def clone_dataset(dataset, rows):
    result = copy.copy(dataset)
    result.data_list = tuple(rows)
    return result


def repeated_training_dataset(dataset, rows, crops_per_video):
    return clone_dataset(
        dataset,
        tuple(row for row in rows for _ in range(crops_per_video)),
    )


def load_base_checkpoint(model, checkpoint_path, weights, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    incompatible = model.load_state_dict(checkpoint[state_key], strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("validity_head.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise ValueError(
            "frozen-head initialization mismatch: missing={} unexpected={}".format(
                invalid_missing, incompatible.unexpected_keys
            )
        )
    if not incompatible.missing_keys:
        raise ValueError("base checkpoint already contains a Validity Head")
    return checkpoint


def base_state(model):
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("validity_head.")
    }


def head_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.validity_head.state_dict().items()
    }


def frozen_source_batch(model, raw_videos, device):
    videos = [move_video(video, device) for video in raw_videos]
    with torch.no_grad():
        inputs, masks = model.preprocessing(videos, mode="validation")
        backbone, backbone_masks, _ = model.backbone_source(inputs, masks)
        fpn, fpn_masks = model.neck_source(backbone, backbone_masks)
        offsets, _ = model.reg_head(fpn, fpn_masks)
        points = model.point_generator(fpn)
        gt_cls, _ = model.label_points(
            points,
            [video["segments"] for video in videos],
            [video["labels"] for video in videos],
        )
        cohorts = model._source_validity_cohorts(
            points,
            [video["segments"] for video in videos],
            fpn_masks,
        )
        labels = []
        start = 0
        for point_grid in points:
            end = start + len(point_grid)
            labels.append(
                torch.stack(
                    [row[start:end].sum(dim=-1) > 0 for row in gt_cls], dim=0
                )
            )
            start = end
    logits = model.validity_head(fpn, offsets, fpn_masks, model.fpn_strides)
    return source_validity_loss(
        logits,
        labels,
        cohorts,
        fpn_masks,
        level_balanced=model.validity_head_level_balanced_loss,
    )


def selection_loss(model, dataset, alpha, args):
    random.seed(args.evaluation_seed)
    torch.manual_seed(args.evaluation_seed)
    model.eval()
    scores, labels, stats, groups = collect_split(
        model,
        dataset,
        alpha,
        args.beta,
        args.device,
        args.selection_crops_per_video,
        return_groups=True,
    )
    levels = groups["levels"] if model.validity_head_level_balanced_loss else None
    return balanced_probability_loss(scores, labels, levels), stats


def collect_operating_report(model, dataset, alpha, args, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model.eval()
    return collect_split(
        model,
        dataset,
        alpha,
        args.beta,
        args.device,
        args.evaluation_crops_per_video,
    )


def validate_args(args):
    positive = (
        args.epochs,
        args.learning_rate,
        args.batch_size,
        args.num_workers,
        args.train_crops_per_video,
        args.selection_crops_per_video,
        args.evaluation_crops_per_video,
        args.selection_frequency,
        args.selection_videos,
        args.calibration_videos,
        args.report_videos,
        args.crop_alignment,
        args.clip_grad_norm,
    )
    if min(positive) <= 0:
        raise ValueError("frozen-head counts and optimization values must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight decay must be non-negative")


def main():
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    head_design = dict(HEAD_VARIANTS[args.head_variant])
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["crop_alignment"] = args.crop_alignment
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
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    source_train = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    source_external_report = make_dataset(
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
    )
    (
        train_rows,
        selection_rows,
        calibration_rows,
        report_rows,
        partition,
    ) = partition_frozen_head_source_videos(
        tuple(source_train.data_list),
        args.selection_videos,
        args.calibration_videos,
        args.report_videos,
        args.partition_seed,
    )
    external_ids = {str(item["id"]) for item in source_external_report.data_list}
    partition_ids = {
        video_id
        for key, values in partition.items()
        if key.endswith("_video_ids")
        for video_id in values
    }
    if partition_ids & external_ids:
        raise ValueError("source train and external report video IDs overlap")
    partition["external_report_video_ids"] = sorted(external_ids)

    train_dataset = repeated_training_dataset(
        source_train, train_rows, args.train_crops_per_video
    )
    selection_dataset = clone_dataset(source_train, selection_rows)
    calibration_dataset = clone_dataset(source_train, calibration_rows)
    internal_report_dataset = clone_dataset(source_train, report_rows)
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_data_loader(
        train_dataset,
        True,
        loader_generator,
        args.batch_size,
        args.num_workers,
    )

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = load_base_checkpoint(
        model, args.checkpoint, args.weights, args.device
    )
    freeze_stats = freeze_sada_except_validity_head(model)
    model.eval()
    model.validity_head.train()
    base_sha256_before = state_dict_sha256(base_state(model))
    initial_head_sha256 = state_dict_sha256(head_state(model))
    optimizer = torch.optim.AdamW(
        model.validity_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best = None
    history = []
    total_supervised_anchors = 0
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.validity_head.train()
        epoch_loss_sum = 0.0
        epoch_anchors = 0
        epoch_positive = 0
        epoch_negative = 0
        epoch_by_level = [
            {"count": 0, "positive": 0, "negative": 0}
            for _ in range(model.n_levels)
        ]
        for raw_videos in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, stats = frozen_source_batch(model, raw_videos, args.device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.validity_head.parameters(), args.clip_grad_norm
            )
            optimizer.step()
            count = int(stats["count"])
            epoch_loss_sum += float(loss.detach()) * count
            epoch_anchors += count
            epoch_positive += int(stats["positive"])
            epoch_negative += int(stats["negative"])
            for total, current in zip(
                epoch_by_level, stats["per_level"]
            ):
                for key in ("count", "positive", "negative"):
                    total[key] += int(current[key])
        if epoch_anchors == 0:
            raise RuntimeError("frozen-head epoch produced no supervised anchors")
        total_supervised_anchors += epoch_anchors
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss_sum / epoch_anchors,
            "train_anchors": epoch_anchors,
            "train_positive": epoch_positive,
            "train_negative": epoch_negative,
            "train_by_level": epoch_by_level,
        }
        if epoch % args.selection_frequency == 0 or epoch == args.epochs:
            current_selection_loss, selection_stats = selection_loss(
                model, selection_dataset, alpha, args
            )
            record["selection_loss"] = current_selection_loss
            record["selection_operating_cohort_count"] = selection_stats[
                "operating_cohort_count"
            ]
            candidate = (
                current_selection_loss,
                epoch,
                head_state(model),
            )
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    if best is None:
        raise RuntimeError("frozen-head training produced no selection checkpoint")
    best_selection_loss, best_epoch, best_head_state = best
    model.validity_head.load_state_dict(best_head_state)
    model.eval()
    base_sha256_after = state_dict_sha256(base_state(model))
    if base_sha256_after != base_sha256_before:
        raise RuntimeError("frozen SADA parameters changed during Head-only training")
    final_head_sha256 = state_dict_sha256(head_state(model))
    if final_head_sha256 == initial_head_sha256:
        raise RuntimeError("Validity Head did not change during training")

    calibration_scores, calibration_labels, calibration_stats = (
        collect_operating_report(
            model, calibration_dataset, alpha, args, args.evaluation_seed + 1
        )
    )
    internal_scores, internal_labels, internal_stats = collect_operating_report(
        model,
        internal_report_dataset,
        alpha,
        args,
        args.evaluation_seed + 2,
    )
    external_scores, external_labels, external_stats = collect_operating_report(
        model,
        source_external_report,
        alpha,
        args,
        args.evaluation_seed + 3,
    )
    requirement_met, calibration = calibrate(
        calibration_scores, calibration_labels, args.minimum_precision
    )
    threshold = calibration["threshold"]
    combined_scores = torch.cat((internal_scores, external_scores))
    combined_labels = torch.cat((internal_labels, external_labels))

    training_protocol = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "train_crops_per_video": args.train_crops_per_video,
        "selection_crops_per_video": args.selection_crops_per_video,
        "evaluation_crops_per_video": args.evaluation_crops_per_video,
        "selection_frequency": args.selection_frequency,
        "seed": args.seed,
        "evaluation_seed": args.evaluation_seed,
        "crop_alignment": args.crop_alignment,
        "clip_grad_norm": args.clip_grad_norm,
        "optimizer": "AdamW",
        "frozen_sada": True,
        "head_variant": args.head_variant,
        "head_design": head_design,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "supervision": "source_gt_official_label_points_conditioned_on_target_event",
        "target_gt_usage": "forbidden",
        "vlm_usage": "none",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "base_state_sha256": base_sha256_after,
        "initial_head_sha256": initial_head_sha256,
        "trained_head_sha256": final_head_sha256,
        "freeze_stats": freeze_stats,
        "partition": partition,
        "partition_sha256": protocol_sha256(partition),
        "training_protocol": training_protocol,
        "training_protocol_sha256": protocol_sha256(training_protocol),
        "total_supervised_anchors": total_supervised_anchors,
        "head_variant": args.head_variant,
        "head_design": head_design,
        "head_design_sha256": protocol_sha256(head_design),
        "best_epoch": best_epoch,
        "best_selection_loss": best_selection_loss,
        "history": history,
        "operating_cohort": "source_event_center_and_beta_lt_teacher_score_lt_alpha",
        "beta": args.beta,
        "alpha": alpha,
        "minimum_precision": args.minimum_precision,
        "calibration_requirement_met": requirement_met,
        "threshold": threshold,
        "calibration": {
            **calibration_stats,
            **split_metrics(calibration_scores, calibration_labels, threshold),
        },
        "internal_report": {
            **internal_stats,
            **split_metrics(internal_scores, internal_labels, threshold),
        },
        "external_report": {
            **external_stats,
            **split_metrics(external_scores, external_labels, threshold),
        },
        "combined_report": split_metrics(
            combined_scores, combined_labels, threshold
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_output = output.with_suffix(".pt")
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "validity_head_state_dict": head_state(model),
            "base_state_sha256": base_sha256_after,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "checkpoint_epoch": payload["checkpoint_epoch"],
            "weights": args.weights,
            "partition_sha256": payload["partition_sha256"],
            "training_protocol_sha256": payload["training_protocol_sha256"],
            "threshold": threshold,
            "best_epoch": best_epoch,
        },
        checkpoint_output,
    )
    payload["head_checkpoint"] = str(checkpoint_output)
    payload["head_checkpoint_sha256"] = file_sha256(checkpoint_output)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "head_checkpoint": str(checkpoint_output),
                "best_epoch": best_epoch,
                "calibration_requirement_met": requirement_met,
                "calibration": payload["calibration"],
                "internal_report": payload["internal_report"],
                "external_report": payload["external_report"],
                "combined_report": payload["combined_report"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
