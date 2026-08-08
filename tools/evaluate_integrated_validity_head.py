"""Calibrate and report an integrated SADA Validity Head on source val."""

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
    high_precision_threshold,
    partition_source_validation_videos,
    protocol_sha256,
)
from libs.models.SADA.stage1_calibration import model_state_sha256
from libs.models.SADA.vlm_routing import file_sha256
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import move_video


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--crop-alignment", type=int, default=1)
    parser.add_argument("--val-crops-per-video", type=int, default=4)
    parser.add_argument("--source-calibration-fraction", type=float, default=0.6)
    parser.add_argument("--validation-split-seed", type=int, default=20260809)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def center_inside_target_event(points, segments):
    if segments.numel() == 0:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    centers = points[:, 0, None]
    return ((centers > segments[None, :, 0]) & (centers < segments[None, :, 1])).any(1)


def split_metrics(scores, labels, threshold):
    selected = scores >= threshold
    true_positive = selected & labels.bool()
    return {
        "anchors": len(labels),
        "valid": int(labels.sum()),
        "selected": int(selected.sum()),
        "precision": float(true_positive.sum() / selected.sum().clamp_min(1)),
        "recall": float(true_positive.sum() / labels.sum().clamp_min(1)),
    }


def best_observed_threshold(scores, labels):
    labels = labels.bool()
    best = None
    positives = int(labels.sum())
    for threshold in torch.unique(scores).sort(descending=True).values:
        selected = scores >= threshold
        true_positive = int((selected & labels).sum())
        count = int(selected.sum())
        precision = true_positive / max(count, 1)
        recall = true_positive / max(positives, 1)
        candidate = (precision, count, float(threshold), recall)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    precision, count, threshold, recall = best
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "selected": count,
    }


def calibrate(scores, labels, minimum_precision):
    try:
        result = high_precision_threshold(scores, labels, minimum_precision)
        return True, result
    except ValueError:
        return False, best_observed_threshold(scores, labels)


def collect_split(
    model,
    dataset,
    alpha,
    beta,
    device,
    crops_per_video,
    return_groups=False,
):
    scores = []
    labels = []
    level_rows = []
    video_rows = []
    sampled_crops = 0
    duplicate_draws = 0
    valid_anchors = 0
    event_center_anchors = 0
    mid_confidence_anchors = 0
    crops_by_video = {}
    with torch.inference_mode():
        for index in range(len(dataset)):
            seen_crops = set()
            attempts = 0
            max_attempts = max(crops_per_video * 4, crops_per_video)
            while len(seen_crops) < crops_per_video and attempts < max_attempts:
                attempts += 1
                raw_video = dataset[index]
                metadata = raw_video["metadata"]
                video_id = str(metadata["video_id"])
                crop_key = (
                    int(metadata.get("crop_start", 0)),
                    int(metadata.get("crop_feat_len", raw_video["feats"].shape[-1])),
                )
                if crop_key in seen_crops:
                    duplicate_draws += 1
                    continue
                seen_crops.add(crop_key)
                sampled_crops += 1
                video = move_video(raw_video, device)
                inputs, masks = model.preprocessing([video], mode="eval")
                backbone, backbone_masks, _ = model.backbone_source(inputs, masks)
                fpn, fpn_masks = model.neck_source(backbone, backbone_masks)
                logits, _ = model.cls_head(fpn, fpn_masks)
                offsets, _ = model.reg_head(fpn, fpn_masks)
                validity_logits = model.validity_head(
                    fpn, offsets, fpn_masks, model.fpn_strides
                )
                points = model.point_generator(fpn)
                gt_cls, _ = model.label_points(
                    points, [video["segments"]], [video["labels"]]
                )
                gt_cls = gt_cls[0].sum(-1) > 0
                concat_start = 0
                for level, (
                    point_grid,
                    level_logits,
                    level_mask,
                    level_validity,
                ) in enumerate(zip(points, logits, fpn_masks, validity_logits)):
                    length = point_grid.shape[0]
                    probabilities = level_logits[0].transpose(0, 1).sigmoid()
                    top_score = probabilities.max(dim=-1).values
                    valid = level_mask[0, 0].bool()
                    event_center = center_inside_target_event(
                        point_grid, video["segments"]
                    )
                    mid_confidence = valid & (top_score > beta) & (top_score < alpha)
                    cohort = mid_confidence & event_center
                    level_labels = gt_cls[concat_start:concat_start + length]
                    valid_anchors += int(valid.sum())
                    event_center_anchors += int((valid & event_center).sum())
                    mid_confidence_anchors += int(mid_confidence.sum())
                    if cohort.any():
                        scores.append(level_validity[0, cohort].sigmoid().cpu())
                        labels.append(level_labels[cohort].cpu())
                        selected_count = int(cohort.sum())
                        level_rows.append(
                            torch.full(
                                (selected_count,), level, dtype=torch.long
                            )
                        )
                        video_rows.extend([video_id] * selected_count)
                    concat_start += length
            crops_by_video[video_id] = len(seen_crops)
    if not scores:
        raise RuntimeError("source split produced no event-center mid-confidence anchors")
    result = (torch.cat(scores), torch.cat(labels), {
        "videos": len(dataset),
        "requested_crops_per_video": crops_per_video,
        "sampled_crops": sampled_crops,
        "duplicate_crop_draws": duplicate_draws,
        "minimum_unique_crops_per_video": min(crops_by_video.values()),
        "maximum_unique_crops_per_video": max(crops_by_video.values()),
        "valid_anchor_count": valid_anchors,
        "event_center_anchor_count": event_center_anchors,
        "mid_confidence_anchor_count": mid_confidence_anchors,
        "operating_cohort_count": sum(len(row) for row in labels),
    })
    if not return_groups:
        return result
    return (*result, {
        "levels": torch.cat(level_rows),
        "video_ids": tuple(video_rows),
    })


def main():
    args = parse_args()
    if args.crop_alignment <= 0 or args.val_crops_per_video <= 0:
        raise ValueError("crop alignment and crops per video must be positive")
    if not 0.0 < args.source_calibration_fraction < 1.0:
        raise ValueError("source calibration fraction must be in (0, 1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["crop_alignment"] = args.crop_alignment
    cfg["validity_head"] = {
        "enabled": True,
        "loss_weight": 1.0,
        "hidden_dim": 128,
        "fusion_dim": 64,
        "level_embedding_dim": 8,
        "detach_fpn": False,
        "detach_geometry": True,
    }
    _update_config(cfg)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    source_val = make_dataset(
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
    )
    calibration_items, report_items, partition = partition_source_validation_videos(
        tuple(source_val.data_list),
        args.source_calibration_fraction,
        args.validation_split_seed,
    )
    source_calibration = copy.copy(source_val)
    source_calibration.data_list = calibration_items
    source_report = copy.copy(source_val)
    source_report.data_list = report_items

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    state = checkpoint.get(state_key)
    if state is None:
        raise ValueError("checkpoint has no {}".format(state_key))
    if not any(key.startswith("validity_head.") for key in state):
        raise ValueError("checkpoint has no integrated validity head parameters")
    model.load_state_dict(state)
    model.eval()

    calibration_scores, calibration_labels, calibration_stats = collect_split(
        model,
        source_calibration,
        alpha,
        args.beta,
        args.device,
        args.val_crops_per_video,
    )
    report_scores, report_labels, report_stats = collect_split(
        model,
        source_report,
        alpha,
        args.beta,
        args.device,
        args.val_crops_per_video,
    )
    requirement_met, calibration = calibrate(
        calibration_scores, calibration_labels, args.minimum_precision
    )
    threshold = calibration["threshold"]
    payload = {
        "schema_version": "integrated-validity-head-calibration-report-v1",
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "state_sha256": model_state_sha256(model),
        "supervision": "source_gt_official_label_points_conditioned_on_target_event",
        "target_gt_usage": "forbidden",
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
        "source_report": {
            **report_stats,
            **split_metrics(report_scores, report_labels, threshold),
        },
        "validation_partition": partition,
        "validation_partition_sha256": protocol_sha256(partition),
        "sampling": {
            "seed": args.seed,
            "crop_alignment": args.crop_alignment,
            "val_crops_per_video": args.val_crops_per_video,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "checkpoint_epoch": payload["checkpoint_epoch"],
        "calibration_requirement_met": requirement_met,
        "threshold": threshold,
        "calibration": payload["calibration"],
        "source_report": payload["source_report"],
    }, indent=2))


if __name__ == "__main__":
    main()
