"""Causal audit for the exact Stage 2 proposal-to-anchor supervision path."""

import argparse
import collections
import json
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


def decode_segments(offsets, centers, strides):
    return torch.stack((
        centers - offsets[..., 0] * strides,
        centers + offsets[..., 1] * strides,
    ), dim=-1)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def parse_checkpoints(values):
    checkpoints = []
    for value in values:
        if "=" not in value:
            raise ValueError("checkpoint must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path:
            raise ValueError("checkpoint must use NAME=PATH")
        checkpoints.append((name, path))
    return checkpoints


def new_anchor_stats():
    return collections.Counter({
        "all_gt_positive": 0,
        "pseudo_positive": 0,
        "pseudo_gt_positive": 0,
        "good_boundary": 0,
    })


def add_anchor_stats(stats, pseudo_positive, gt_positive, good_boundary):
    stats["all_gt_positive"] += int(gt_positive.sum())
    stats["pseudo_positive"] += int(pseudo_positive.sum())
    stats["pseudo_gt_positive"] += int((pseudo_positive & gt_positive).sum())
    stats["good_boundary"] += int(good_boundary.sum())


def summarize_anchor(stats):
    return {
        **dict(stats),
        "gt_anchor_coverage": ratio(
            stats["pseudo_gt_positive"], stats["all_gt_positive"]
        ),
        "pseudo_anchor_foreground_precision": ratio(
            stats["pseudo_gt_positive"], stats["pseudo_positive"]
        ),
        "pseudo_anchor_good_boundary_precision": ratio(
            stats["good_boundary"], stats["pseudo_positive"]
        ),
    }


def new_delta_stats():
    return {"teacher_wins": 0, "student_wins": 0, "ties": 0,
            "delta_iou": [], "offset_l1": [], "near_same": 0}


def add_delta_stats(stats, teacher_segments, student_segments, gt_segments,
                    mask, tie_eps, near_same_l1):
    if not mask.any():
        return
    teacher_iou = aligned_iou(teacher_segments[mask], gt_segments[mask])
    student_iou = aligned_iou(student_segments[mask], gt_segments[mask])
    delta = teacher_iou - student_iou
    offset_l1 = (teacher_segments[mask] - student_segments[mask]).abs().mean(-1)
    stats["teacher_wins"] += int((delta > tie_eps).sum())
    stats["student_wins"] += int((delta < -tie_eps).sum())
    stats["ties"] += int((delta.abs() <= tie_eps).sum())
    stats["near_same"] += int((offset_l1 <= near_same_l1).sum())
    stats["delta_iou"].extend(delta.cpu().tolist())
    stats["offset_l1"].extend(offset_l1.cpu().tolist())


def summarize_delta(stats):
    count = len(stats["delta_iou"])
    result = {
        "n": count,
        "teacher_win_rate": ratio(stats["teacher_wins"], count),
        "student_win_rate": ratio(stats["student_wins"], count),
        "tie_rate": ratio(stats["ties"], count),
        "near_same_prediction_rate": ratio(stats["near_same"], count),
    }
    for name in ("delta_iou", "offset_l1"):
        values = stats[name]
        result["{}_mean".format(name)] = sum(values) / len(values) if values else None
        result["{}_median".format(name)] = (
            float(torch.tensor(values).median()) if values else None
        )
    return result


def score_bin(scores):
    for lower, upper in ((0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        if lower <= scores < upper:
            return "{:.1f}-{:.1f}".format(lower, min(upper, 1.0))
    return "out_of_range"


def load_models(cfg, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    teacher = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    student.load_state_dict(checkpoint["state_dict"])
    teacher.load_state_dict(checkpoint["ema_state_dict"])
    student.eval()
    teacher.eval()
    return checkpoint, student, teacher


def audit_checkpoint(name, checkpoint_path, cfg, dataset, annotations, device,
                     batch_size, levels, manifest, tie_eps, near_same_l1):
    checkpoint, student, teacher = load_models(cfg, checkpoint_path, device)
    summary = collections.Counter()
    post_anchor = new_anchor_stats()
    per_level_anchor = {str(level): new_anchor_stats() for level in levels}
    per_score_anchor = collections.defaultdict(new_anchor_stats)
    delta = new_delta_stats()
    per_level_delta = {str(level): new_delta_stats() for level in levels}
    per_score_delta = collections.defaultdict(new_delta_stats)
    proposal = collections.Counter()
    proposal_anchor = collections.Counter()

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_videos]
            post_nms, pre_nms = teacher.generate_stage2_pseudo_targets(
                batch, return_pre_nms=True
            )

            inputs, masks = student.preprocessing(batch, mode="train")
            backbone_feats, backbone_masks, _ = student.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = student.neck_target(backbone_feats, backbone_masks)
            student_offsets, _ = student.reg_head_target(fpn_feats, fpn_masks)
            student_offsets = torch.cat([
                offsets.permute(0, 2, 1) for offsets in student_offsets
            ], dim=1)
            points = student.point_generator(fpn_feats)
            valid = torch.cat([mask.squeeze(1) for mask in fpn_masks], dim=1)
            enabled = torch.cat([
                torch.full(
                    (len(batch), point.shape[0]), level in levels,
                    dtype=torch.bool, device=device,
                )
                for level, point in enumerate(points)
            ], dim=1)
            centers = torch.cat([point[:, 0] for point in points])
            strides = torch.cat([point[:, 3] for point in points])

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, batch):
                video_id = raw_video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id], video["metadata"],
                    video["feats"].shape[-1], cfg, device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)

            gt_cls, gt_offsets = student.label_points(points, gt_segments, gt_labels)
            pseudo_cls, pseudo_offsets, assignments = student.label_points(
                points,
                [item["segments"] for item in post_nms],
                [item["labels"] for item in post_nms],
                return_assignments=True,
            )

            summary["batches"] += 1
            for index, (raw_video, gt, post, pre) in enumerate(zip(
                    raw_videos, gt_segments, post_nms, pre_nms)):
                summary["videos"] += 1
                proposal["pre_nms_n"] += int(pre["segments"].shape[0])
                proposal["post_nms_n"] += int(post["segments"].shape[0])
                if post["segments"].numel():
                    summary["videos_with_proposal"] += 1
                if any(item["scores"].numel() for item in post_nms):
                    summary["batches_with_proposal"] += int(index == 0)

                proposal["all_gt_segments"] += int(gt.shape[0])
                for prefix, targets in (("pre", pre), ("post", post)):
                    if gt.numel() and targets["segments"].numel():
                        matches = pairwise_iou(targets["segments"], gt) >= 0.5
                        proposal["{}_matched_gt_segments".format(prefix)] += int(
                            matches.any(dim=0).sum()
                        )

                pseudo_positive = (
                    valid[index] & enabled[index]
                    & (pseudo_cls[index].sum(dim=-1) > 0)
                )
                gt_positive = gt_cls[index].sum(dim=-1) > 0
                pseudo_segments = decode_segments(
                    pseudo_offsets[index], centers, strides
                )
                gt_anchor_segments = decode_segments(
                    gt_offsets[index], centers, strides
                )
                student_segments = decode_segments(
                    student_offsets[index], centers, strides
                )
                overlap = pseudo_positive & gt_positive
                good_boundary = torch.zeros_like(pseudo_positive)
                if overlap.any():
                    good_boundary[overlap] = aligned_iou(
                        pseudo_segments[overlap], gt_anchor_segments[overlap]
                    ) >= 0.5
                add_anchor_stats(
                    post_anchor, pseudo_positive, gt_positive, good_boundary
                )
                add_delta_stats(
                    delta, pseudo_segments, student_segments, gt_anchor_segments,
                    overlap, tie_eps, near_same_l1,
                )

                offset = 0
                for level, point in enumerate(points):
                    next_offset = offset + point.shape[0]
                    level_mask = torch.zeros_like(pseudo_positive)
                    level_mask[offset:next_offset] = True
                    if level in levels:
                        key = str(level)
                        add_anchor_stats(
                            per_level_anchor[key],
                            pseudo_positive & level_mask,
                            gt_positive & level_mask,
                            good_boundary & level_mask,
                        )
                        add_delta_stats(
                            per_level_delta[key], pseudo_segments, student_segments,
                            gt_anchor_segments, overlap & level_mask,
                            tie_eps, near_same_l1,
                        )
                    offset = next_offset

                assignment = assignments[index]
                for proposal_index, score in enumerate(post["scores"]):
                    assigned = pseudo_positive & (assignment == proposal_index)
                    anchor_count = int(assigned.sum())
                    proposal_anchor["assigned_anchors"] += anchor_count
                    if gt.numel():
                        matched = bool((pairwise_iou(
                            post["segments"][proposal_index:proposal_index + 1], gt
                        ).max() >= 0.5).item())
                    else:
                        matched = False
                    group = "true" if matched else "false"
                    proposal_anchor["{}_proposals".format(group)] += 1
                    proposal_anchor["{}_assigned_anchors".format(group)] += anchor_count

                    bin_name = score_bin(float(score))
                    score_mask = assigned
                    add_anchor_stats(
                        per_score_anchor[bin_name], score_mask,
                        gt_positive & score_mask, good_boundary & score_mask,
                    )
                    add_delta_stats(
                        per_score_delta[bin_name], pseudo_segments, student_segments,
                        gt_anchor_segments, overlap & score_mask,
                        tie_eps, near_same_l1,
                    )

    proposal["post_nms_gt_segment_recall"] = ratio(
        proposal["post_matched_gt_segments"], proposal["all_gt_segments"]
    )
    proposal["pre_nms_gt_segment_recall"] = ratio(
        proposal["pre_matched_gt_segments"], proposal["all_gt_segments"]
    )
    proposal_anchor["all_anchor_amplification"] = ratio(
        proposal_anchor["assigned_anchors"], proposal["post_nms_n"]
    )
    for group in ("true", "false"):
        proposal_anchor["{}_anchor_amplification".format(group)] = ratio(
            proposal_anchor["{}_assigned_anchors".format(group)],
            proposal_anchor["{}_proposals".format(group)],
        )
    return {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "proposal": dict(proposal),
        "proposal_to_anchor": dict(proposal_anchor),
        "activity": {
            "videos": summary["videos"],
            "batches": summary["batches"],
            "video_with_proposal_rate": ratio(
                summary["videos_with_proposal"], summary["videos"]
            ),
            "batch_with_proposal_rate": ratio(
                summary["batches_with_proposal"], summary["batches"]
            ),
            "proposals_per_batch": ratio(proposal["post_nms_n"], summary["batches"]),
            "pseudo_anchors_per_batch": ratio(
                post_anchor["pseudo_positive"], summary["batches"]
            ),
        },
        "anchor": summarize_anchor(post_anchor),
        "anchor_by_level": {
            key: summarize_anchor(value) for key, value in per_level_anchor.items()
        },
        "anchor_by_proposal_score": {
            key: summarize_anchor(value) for key, value in per_score_anchor.items()
        },
        "teacher_vs_student": summarize_delta(delta),
        "teacher_vs_student_by_level": {
            key: summarize_delta(value) for key, value in per_level_delta.items()
        },
        "teacher_vs_student_by_proposal_score": {
            key: summarize_delta(value) for key, value in per_score_delta.items()
        },
        "manifest_video_ids": manifest,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--near-same-l1", type=float, default=1e-2)
    args = parser.parse_args()

    levels = [int(value) for value in args.levels.split(",") if value]
    if not levels or len(levels) != len(set(levels)) or any(
            level < 0 or level > 5 for level in levels):
        raise ValueError("levels must be unique values from 0 to 5")
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {"enabled": True, "start_epoch": 5, "weight": 1.0, "levels": levels}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    manifest = [dataset[index]["metadata"]["video_id"] for index in range(len(dataset))]
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    results = {}
    for name, checkpoint_path in parse_checkpoints(args.checkpoint):
        results[name] = audit_checkpoint(
            name, checkpoint_path, cfg, dataset, annotations, args.device,
            batch_size, levels, manifest, args.tie_eps, args.near_same_l1,
        )
    output = {
        "audit": "stage2_causal",
        "method": "EMA proposal -> hard class-agnostic NMS -> label_points",
        "levels": levels,
        "batch_size": batch_size,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
