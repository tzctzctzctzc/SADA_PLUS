"""Audit the exact Stage 2 EMA proposal and anchor-reassignment pipeline."""

import argparse
import collections
import json
import random
from pathlib import Path

import torch

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def pairwise_iou(segments_a, segments_b):
    if segments_a.numel() == 0 or segments_b.numel() == 0:
        return segments_a.new_zeros((segments_a.shape[0], segments_b.shape[0]))
    left = torch.maximum(segments_a[:, None, 0], segments_b[None, :, 0])
    right = torch.minimum(segments_a[:, None, 1], segments_b[None, :, 1])
    intersection = (right - left).clamp_min(0)
    union = (
        (segments_a[:, None, 1] - segments_a[:, None, 0]).clamp_min(0)
        + (segments_b[None, :, 1] - segments_b[None, :, 0]).clamp_min(0)
        - intersection
    )
    return intersection / union.clamp_min(1e-6)


def aligned_iou(prediction, target):
    return pairwise_iou(prediction, target).diagonal()


def add_iou_summary(result, prefix, values):
    result[prefix + "_n"] = len(values)
    result[prefix + "_mean"] = sum(values) / len(values) if values else None
    result[prefix + "_median"] = (
        float(torch.tensor(values).median().item()) if values else None
    )
    for threshold in (0.1, 0.3, 0.5, 0.7):
        result[prefix + "_ge_{:.1f}".format(threshold)] = (
            sum(value >= threshold for value in values) / len(values)
            if values else None
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-source", choices=("ema", "student"), default="ema")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    args = parser.parse_args()

    levels = [int(level.strip()) for level in args.levels.split(",") if level.strip()]
    if not levels or len(levels) != len(set(levels)) or any(level < 0 or level > 5 for level in levels):
        raise ValueError("levels must be unique FPN levels from 0 to 5")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {"enabled": True, "start_epoch": 5, "weight": 1.0, "levels": levels}
    _update_config(cfg)
    dataset = make_dataset(cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"])
    annotations = json.loads(Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8"))["database"]
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.teacher_source == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    proposal = collections.Counter()
    anchor = collections.Counter()
    proposal_ious, anchor_ious = [], []
    level_positive = collections.Counter()
    videos = 0
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, args.device) for video in raw_videos]
            pseudo_targets = model.generate_stage2_pseudo_targets(batch)

            inputs, masks = model.preprocessing(batch, mode="train")
            backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
            points = model.point_generator(fpn_feats)
            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, batch):
                video_id = raw_video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id], video["metadata"], video["feats"].shape[-1],
                    cfg, args.device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)
                videos += 1

            gt_cls, gt_offsets = model.label_points(points, gt_segments, gt_labels)
            pseudo_cls, pseudo_offsets = model.label_points(
                points,
                [item["segments"] for item in pseudo_targets],
                [item["labels"] for item in pseudo_targets],
            )
            valid = torch.cat([mask.squeeze(1) for mask in fpn_masks], dim=1)
            enabled_levels = torch.cat([
                torch.full(
                    (len(batch), point.shape[0]), level in levels,
                    dtype=torch.bool, device=args.device,
                )
                for level, point in enumerate(points)
            ], dim=1)
            centers = torch.cat([point[:, 0] for point in points])
            strides = torch.cat([point[:, 3] for point in points])

            for index, (targets, segments, labels) in enumerate(
                    zip(pseudo_targets, gt_segments, gt_labels)):
                proposals = targets["segments"]
                classes = targets["labels"]
                proposal["n"] += int(proposals.shape[0])
                if proposals.numel() and segments.numel():
                    ious = pairwise_iou(proposals, segments)
                    best_iou, best_index = ious.max(dim=1)
                    proposal_ious.extend(best_iou.cpu().tolist())
                    proposal["matched_gt_iou_0.5_n"] += int((best_iou >= 0.5).sum())
                    class_correct = classes == labels[best_index]
                    proposal["best_match_class_correct_n"] += int(
                        class_correct.sum().item()
                    )
                    proposal["matched_gt_iou_0.5_class_correct_n"] += int(
                        (class_correct & (best_iou >= 0.5)).sum().item()
                    )
                    matches_per_gt = (ious >= 0.5).sum(dim=0)
                    matched_gt = matches_per_gt > 0
                    proposal["matched_gt_n"] += int(matched_gt.sum())
                    proposal["proposals_for_matched_gt_n"] += int(
                        matches_per_gt[matched_gt].sum()
                    )

                pseudo_positive = (
                    valid[index] & enabled_levels[index]
                    & (pseudo_cls[index].sum(dim=-1) > 0)
                )
                gt_foreground = gt_cls[index].sum(dim=-1) > 0
                anchor["pseudo_positive_n"] += int(pseudo_positive.sum())
                anchor["pseudo_positive_gt_foreground_n"] += int(
                    (pseudo_positive & gt_foreground).sum()
                )
                overlapping = pseudo_positive & gt_foreground
                if overlapping.any():
                    pseudo_segments = torch.stack((
                        centers - pseudo_offsets[index][:, 0] * strides,
                        centers + pseudo_offsets[index][:, 1] * strides,
                    ), dim=-1)
                    target_segments = torch.stack((
                        centers - gt_offsets[index][:, 0] * strides,
                        centers + gt_offsets[index][:, 1] * strides,
                    ), dim=-1)
                    anchor_ious.extend(aligned_iou(
                        pseudo_segments[overlapping], target_segments[overlapping]
                    ).cpu().tolist())

                offset = 0
                for level, point in enumerate(points):
                    next_offset = offset + point.shape[0]
                    level_positive[level] += int(
                        pseudo_positive[offset:next_offset].sum()
                    )
                    offset = next_offset

    result = {
        "audit": "exact_stage2_ema_proposal_and_reassignment",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "teacher_source": args.teacher_source,
        "videos": videos,
        "method": {
            "proposal": "EMA eval -> alpha -> crop clip -> hard class-agnostic NMS -> alpha",
            "anchor": "label_points reassignment on student-equivalent FPN grids",
            "target_gt_use": "post-prediction offline metric only",
            "levels": levels,
        },
        "proposal": dict(proposal),
        "anchor": dict(anchor),
        "anchor_positive_amplification": (
            anchor["pseudo_positive_n"] / proposal["n"] if proposal["n"] else None
        ),
        "proposal_matched_gt_iou_0.5_rate": (
            proposal["matched_gt_iou_0.5_n"] / proposal["n"] if proposal["n"] else None
        ),
        "proposal_best_match_class_rate": (
            proposal["best_match_class_correct_n"] / proposal["n"]
            if proposal["n"] else None
        ),
        "proposal_class_correct_at_iou_0.5_rate": (
            proposal["matched_gt_iou_0.5_class_correct_n"]
            / proposal["matched_gt_iou_0.5_n"]
            if proposal["matched_gt_iou_0.5_n"] else None
        ),
        "mean_proposals_per_matched_gt": (
            proposal["proposals_for_matched_gt_n"] / proposal["matched_gt_n"]
            if proposal["matched_gt_n"] else None
        ),
        "anchor_gt_foreground_rate": (
            anchor["pseudo_positive_gt_foreground_n"] / anchor["pseudo_positive_n"]
            if anchor["pseudo_positive_n"] else None
        ),
        "anchor_positive_by_level": {str(level): level_positive[level] for level in range(len(points))},
    }
    add_iou_summary(result, "proposal_best_match_iou", proposal_ious)
    add_iou_summary(result, "anchor_reassigned_iou", anchor_ious)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
