"""Train and calibrate an anchor-validity probe using source GT only."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.anchor_validity import (
    anchor_validity_features,
    fit_logistic_probe,
    high_precision_threshold,
    probe_probabilities,
    serialize_probe,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import move_video
from tools.export_vlm_training_candidates import decode_teacher_offsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max-train-anchors", type=int, default=200000)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260808)
    return parser.parse_args()


def center_inside_target_event(points, segments):
    if segments.numel() == 0:
        return torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
    centers = points[:, 0, None]
    return ((centers > segments[None, :, 0]) & (centers < segments[None, :, 1])).any(1)


def collect_split(model, dataset, alpha, beta, device, max_videos):
    feature_rows = []
    label_rows = []
    semantic_rows = 0
    low_confidence_rows = 0
    video_count = min(len(dataset), max_videos or len(dataset))
    with torch.inference_mode():
        for index in range(video_count):
            raw_video = dataset[index]
            video = move_video(raw_video, device)
            inputs, masks = model.preprocessing([video], mode="eval")
            backbone, backbone_masks, _ = model.backbone_source(inputs, masks)
            fpn, fpn_masks = model.neck_source(backbone, backbone_masks)
            logits, _ = model.cls_head(fpn, fpn_masks)
            decoded, actionness = decode_teacher_offsets(
                model, fpn, fpn_masks, domain="source"
            )
            points = model.point_generator(fpn)
            gt_cls, _ = model.label_points(
                points,
                [video["segments"]],
                [video["labels"]],
            )
            gt_cls = gt_cls[0].sum(-1) > 0
            concat_start = 0
            for level, (point_grid, level_logits, level_mask) in enumerate(
                zip(points, logits, fpn_masks)
            ):
                length = point_grid.shape[0]
                probabilities = level_logits[0].transpose(0, 1).sigmoid()
                top2 = probabilities.topk(k=2, dim=-1).values
                valid = level_mask[0, 0].bool()
                low_confidence = (
                    valid & (top2[:, 0] > beta) & (top2[:, 0] < alpha)
                )
                semantic = center_inside_target_event(point_grid, video["segments"])
                cohort = low_confidence & semantic
                semantic_rows += int((valid & semantic).sum())
                low_confidence_rows += int(low_confidence.sum())
                if cohort.any():
                    top1 = probabilities.argmax(-1)
                    row_indices = torch.arange(length, device=device)
                    class_offsets = decoded[level][top1, row_indices]
                    level_actionness = actionness[level][0, 0]
                    features = anchor_validity_features(
                        probabilities,
                        level_actionness,
                        class_offsets[:, 0],
                        class_offsets[:, 1],
                        float(point_grid[0, 3]),
                        level,
                    )
                    labels = gt_cls[concat_start:concat_start + length]
                    feature_rows.append(features[cohort].cpu())
                    label_rows.append(labels[cohort].cpu())
                concat_start += length
    if not feature_rows:
        raise RuntimeError("source split produced no semantic low-confidence anchors")
    return (
        torch.cat(feature_rows),
        torch.cat(label_rows),
        {
            "videos": video_count,
            "semantic_anchor_count": semantic_rows,
            "low_confidence_anchor_count": low_confidence_rows,
            "probe_cohort_count": sum(len(row) for row in label_rows),
        },
    )


def bounded_sample(features, labels, maximum, seed):
    if maximum <= 0 or len(labels) <= maximum:
        return features, labels
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(labels), generator=generator)[:maximum]
    return features[selected], labels[selected]


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


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    source_train = make_dataset(
        cfg["dataset_name"], False, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    train_x, train_y, train_stats = collect_split(
        model, source_train, alpha, args.beta, args.device, args.max_videos
    )
    val_x, val_y, val_stats = collect_split(
        model, source_val, alpha, args.beta, args.device, args.max_videos
    )
    train_x, train_y = bounded_sample(
        train_x, train_y, args.max_train_anchors, args.seed
    )
    probe = fit_logistic_probe(train_x, train_y, args.max_iter, args.l2)
    train_scores = probe_probabilities(train_x, probe)
    val_scores = probe_probabilities(val_x, probe)
    calibration = high_precision_threshold(
        val_scores, val_y, args.minimum_precision
    )
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
            "beta": args.beta,
            "alpha": alpha,
            "minimum_precision": args.minimum_precision,
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
