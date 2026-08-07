"""Train and calibrate a source-only anchor-validity model on current crops."""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.anchor_validity import (
    FEATURE_GROUPS,
    assignment_protocol_from_model,
    anchor_validity_features,
    cohort_protocol_from_config,
    fit_fpn_mlp_probe,
    fit_gbdt_probe,
    fit_logistic_probe,
    fit_mlp_probe,
    high_precision_threshold,
    probe_probabilities,
    protocol_sha256,
    sampling_protocol_from_config,
    serialize_probe,
)
from libs.utils.utils import _update_config, read_yaml
from libs.models.SADA.stage1_calibration import model_state_sha256
from libs.models.SADA.vlm_routing import file_sha256
from tools.audit_original_sada_pseudolabels import move_video
from tools.export_vlm_training_candidates import decode_teacher_offsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument(
        "--model-type",
        choices=("logistic", "mlp", "gbdt", "fpn_mlp"),
        default="logistic",
    )
    parser.add_argument(
        "--feature-group", choices=tuple(FEATURE_GROUPS), default="reliability_full"
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--gbdt-estimators", type=int, default=100)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-max-depth", type=int, default=3)
    parser.add_argument("--crop-alignment", type=int, default=1)
    parser.add_argument("--max-train-anchors", type=int, default=200000)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--train-crops-per-video", type=int, default=1)
    parser.add_argument("--val-crops-per-video", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def center_inside_target_event(points, segments):
    if segments.numel() == 0:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    centers = points[:, 0, None]
    return ((centers > segments[None, :, 0]) & (centers < segments[None, :, 1])).any(1)


def collect_split(
    model,
    dataset,
    alpha,
    beta,
    device,
    max_videos,
    crops_per_video,
):
    feature_rows = []
    fpn_rows = []
    label_rows = []
    semantic_rows = 0
    low_confidence_rows = 0
    sampled_crops = 0
    duplicate_draws = 0
    crops_by_video = {}
    video_count = min(len(dataset), max_videos or len(dataset))
    with torch.inference_mode():
        for index in range(video_count):
            seen_crops = set()
            attempts = 0
            max_attempts = max(crops_per_video * 4, crops_per_video)
            while len(seen_crops) < crops_per_video and attempts < max_attempts:
                attempts += 1
                raw_video = dataset[index]
                metadata = raw_video["metadata"]
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
                decoded = decode_teacher_offsets(
                    model, fpn, fpn_masks, domain="source"
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
                    level_fpn,
                ) in enumerate(zip(points, logits, fpn_masks, fpn)):
                    length = point_grid.shape[0]
                    probabilities = level_logits[0].transpose(0, 1).sigmoid()
                    top2 = probabilities.topk(k=2, dim=-1).values
                    valid = level_mask[0, 0].bool()
                    low_confidence = (
                        valid & (top2[:, 0] > beta) & (top2[:, 0] < alpha)
                    )
                    semantic = center_inside_target_event(
                        point_grid, video["segments"]
                    )
                    cohort = low_confidence & semantic
                    semantic_rows += int((valid & semantic).sum())
                    low_confidence_rows += int(low_confidence.sum())
                    if cohort.any():
                        top1 = probabilities.argmax(-1)
                        row_indices = torch.arange(length, device=device)
                        class_offsets = decoded[level][top1, row_indices]
                        features = anchor_validity_features(
                            probabilities,
                            class_offsets[:, 0],
                            class_offsets[:, 1],
                            float(point_grid[0, 3]),
                            level,
                        )
                        labels = gt_cls[concat_start:concat_start + length]
                        feature_rows.append(features[cohort].cpu())
                        fpn_rows.append(
                            level_fpn[0].transpose(0, 1)[cohort].cpu()
                        )
                        label_rows.append(labels[cohort].cpu())
                    concat_start += length
            video_id = str(raw_video["metadata"]["video_id"])
            crops_by_video[video_id] = len(seen_crops)
    if not feature_rows:
        raise RuntimeError("source split produced no semantic low-confidence anchors")
    return (
        torch.cat(feature_rows),
        torch.cat(fpn_rows),
        torch.cat(label_rows),
        {
            "videos": video_count,
            "requested_crops_per_video": crops_per_video,
            "sampled_crops": sampled_crops,
            "duplicate_crop_draws": duplicate_draws,
            "minimum_unique_crops_per_video": min(crops_by_video.values()),
            "maximum_unique_crops_per_video": max(crops_by_video.values()),
            "semantic_anchor_count": semantic_rows,
            "low_confidence_anchor_count": low_confidence_rows,
            "model_cohort_count": sum(len(row) for row in label_rows),
        },
    )


def bounded_sample(features, fpn_features, labels, maximum, seed):
    if maximum <= 0 or len(labels) <= maximum:
        return features, fpn_features, labels
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(labels), generator=generator)[:maximum]
    return features[selected], fpn_features[selected], labels[selected]


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


def fit_model(args, features, fpn_features, labels):
    if args.model_type == "logistic":
        return fit_logistic_probe(
            features, labels, args.max_iter, args.l2, args.feature_group
        )
    if args.model_type == "mlp":
        return fit_mlp_probe(
            features,
            labels,
            args.feature_group,
            args.epochs,
            args.learning_rate,
            args.l2,
            args.seed,
        )
    if args.model_type == "gbdt":
        return fit_gbdt_probe(
            features,
            labels,
            args.feature_group,
            args.gbdt_estimators,
            args.gbdt_learning_rate,
            args.gbdt_max_depth,
            args.seed,
        )
    return fit_fpn_mlp_probe(
        features,
        fpn_features,
        labels,
        args.epochs,
        args.learning_rate,
        args.l2,
        args.seed,
    )


def main():
    args = parse_args()
    if args.crop_alignment <= 0:
        raise ValueError("--crop-alignment must be positive")
    if args.train_crops_per_video <= 0 or args.val_crops_per_video <= 0:
        raise ValueError("crops per video must be positive")
    if args.model_type == "fpn_mlp" and args.feature_group != "structure_pure":
        raise ValueError("fpn_mlp uses FPN + structural scalars; select structure_pure")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["dataset"]["crop_alignment"] = args.crop_alignment
    _update_config(cfg)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    source_train = make_dataset(
        cfg["dataset_name"], True, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], True, cfg["src_val_split"], **cfg["dataset"]
    )
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    teacher_provenance = {
        "routing_teacher_checkpoint_sha256": file_sha256(args.checkpoint),
        "routing_teacher_state_sha256": model_state_sha256(model),
        "routing_teacher_weights": args.weights,
    }
    assignment_protocol = assignment_protocol_from_model(model)
    cohort_protocol = cohort_protocol_from_config(cfg, model, args.beta, alpha)
    sampling_protocol = sampling_protocol_from_config(
        cfg,
        args.seed,
        args.max_train_anchors,
        args.max_videos,
        args.train_crops_per_video,
        args.val_crops_per_video,
    )
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = "unknown"

    train_x, train_fpn, train_y, train_stats = collect_split(
        model,
        source_train,
        alpha,
        args.beta,
        args.device,
        args.max_videos,
        args.train_crops_per_video,
    )
    val_x, val_fpn, val_y, val_stats = collect_split(
        model,
        source_val,
        alpha,
        args.beta,
        args.device,
        args.max_videos,
        args.val_crops_per_video,
    )
    train_x, train_fpn, train_y = bounded_sample(
        train_x, train_fpn, train_y, args.max_train_anchors, args.seed
    )
    probe = fit_model(args, train_x, train_fpn, train_y)
    train_scores = probe_probabilities(train_x, probe, train_fpn)
    val_scores = probe_probabilities(val_x, probe, val_fpn)
    calibration = high_precision_threshold(val_scores, val_y, args.minimum_precision)
    threshold = calibration["threshold"]
    payload = serialize_probe(
        probe,
        threshold,
        {
            "supervision": "source_gt_official_label_points_conditioned_on_target_event",
            "target_gt_usage": "forbidden",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "weights": args.weights,
            "source_train_split": list(cfg["src_train_split"]),
            "source_val_split": list(cfg["src_val_split"]),
            "crop_training": True,
            "crop_alignment": args.crop_alignment,
            "beta": args.beta,
            "alpha": alpha,
            "minimum_precision": args.minimum_precision,
            "teacher_provenance": teacher_provenance,
            "teacher_provenance_sha256": protocol_sha256(teacher_provenance),
            "assignment_protocol": assignment_protocol,
            "assignment_protocol_sha256": protocol_sha256(
                assignment_protocol
            ),
            "cohort_protocol": cohort_protocol,
            "cohort_protocol_sha256": protocol_sha256(cohort_protocol),
            "sampling_protocol": sampling_protocol,
            "sampling_protocol_sha256": protocol_sha256(sampling_protocol),
            "git_commit": git_commit,
            "calibration": calibration,
            "source_train": {
                **train_stats,
                **split_metrics(train_scores, train_y, threshold),
            },
            "source_val": {
                **val_stats,
                **split_metrics(val_scores, val_y, threshold),
            },
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **payload["source_val"]}, indent=2))


if __name__ == "__main__":
    main()
