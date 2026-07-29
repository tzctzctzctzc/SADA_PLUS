"""Target-GT offline audit of NPL-style boundary reliability."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage2_utils import matched_budget_indices
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video
from tools.audit_stage2_causal import (
    aligned_iou,
    decode_segments,
    pairwise_iou,
    parse_checkpoints,
    ratio,
)


def reset_audit_rng(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_numbers(value, cast):
    return [cast(item) for item in value.split(",") if item]


def pearson(values_x, values_y):
    if len(values_x) < 2:
        return None
    x = torch.tensor(values_x, dtype=torch.float64)
    y = torch.tensor(values_y, dtype=torch.float64)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[finite], y[finite]
    if x.numel() < 2 or x.std(unbiased=False) == 0 or y.std(unbiased=False) == 0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (
        x.std(unbiased=False) * y.std(unbiased=False)
    ))


def rank_values(values):
    values = torch.tensor(values, dtype=torch.float64)
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    ranks[order] = torch.arange(values.numel(), dtype=values.dtype)
    return ranks.tolist()


def summarize_selection(
        records, selected_indices, total_gt_segments, total_gt_anchors,
        tie_eps, near_same_l1):
    selected = [records[index] for index in selected_indices]
    proposal_count = len(selected)
    foreground_count = sum(record["proposal_foreground"] for record in selected)
    class_correct = sum(record["proposal_class_correct"] for record in selected)
    matched_gt = {
        key
        for record in selected
        for key in record["matched_gt_keys"]
    }
    assigned_anchors = sum(record["assigned_anchors"] for record in selected)
    foreground_anchors = sum(
        record["foreground_anchors"] for record in selected
    )
    good_boundary_anchors = sum(
        record["good_boundary_anchors"] for record in selected
    )
    deltas = [
        value for record in selected for value in record["delta_iou"]
    ]
    offset_l1 = [
        value for record in selected for value in record["offset_l1"]
    ]
    teacher_wins = sum(value > tie_eps for value in deltas)
    student_wins = sum(value < -tie_eps for value in deltas)
    ties = len(deltas) - teacher_wins - student_wins
    level_counts = {}
    for record in selected:
        key = str(record["level"])
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        "proposal": {
            "n": proposal_count,
            "fpn_level_counts": level_counts,
            "foreground_precision_iou50": ratio(
                foreground_count, proposal_count
            ),
            "class_correct_precision": ratio(class_correct, proposal_count),
            "class_correct_given_foreground": ratio(
                class_correct, foreground_count
            ),
            "mean_max_gt_iou": ratio(
                sum(record["proposal_max_gt_iou"] for record in selected),
                proposal_count,
            ),
            "iou70_rate": ratio(
                sum(record["proposal_max_gt_iou"] >= 0.7 for record in selected),
                proposal_count,
            ),
            "gt_segment_recall_iou50": ratio(
                len(matched_gt), total_gt_segments
            ),
        },
        "anchor": {
            "pseudo_positive": assigned_anchors,
            "foreground_precision": ratio(
                foreground_anchors, assigned_anchors
            ),
            "good_boundary_precision_iou50": ratio(
                good_boundary_anchors, assigned_anchors
            ),
            "gt_foreground_anchor_coverage": ratio(
                foreground_anchors, total_gt_anchors
            ),
            "proposal_to_anchor_amplification": ratio(
                assigned_anchors, proposal_count
            ),
        },
        "teacher_vs_student_on_gt_foreground_anchors": {
            "n": len(deltas),
            "teacher_win_rate": ratio(teacher_wins, len(deltas)),
            "student_win_rate": ratio(student_wins, len(deltas)),
            "tie_rate": ratio(ties, len(deltas)),
            "mean_delta_iou": ratio(sum(deltas), len(deltas)),
            "mean_teacher_student_boundary_l1": ratio(
                sum(offset_l1), len(offset_l1)
            ),
            "near_same_prediction_rate": ratio(
                sum(value <= near_same_l1 for value in offset_l1),
                len(offset_l1),
            ),
        },
    }


def select_within_fpn_level(records, values, fraction, level_key="level"):
    selected = []
    levels = sorted({record[level_key] for record in records})
    for level in levels:
        global_indices = [
            index for index, record in enumerate(records)
            if record[level_key] == level
        ]
        local_values = values[global_indices]
        local_indices = matched_budget_indices(
            local_values, fraction
        ).tolist()
        selected.extend(global_indices[index] for index in local_indices)
    return selected


def summarize_rankings(
        records, radii, fractions, total_gt_segments, total_gt_anchors,
        tie_eps, near_same_l1):
    if not records:
        return {"records": 0, "correlation": {}, "matched_budget": {}}
    confidence = torch.tensor(
        [record["score"] for record in records], dtype=torch.float64
    )
    max_gt_iou = [record["proposal_max_gt_iou"] for record in records]
    result = {
        "records": len(records),
        "correlation": {
            "confidence_vs_max_gt_iou_pearson": pearson(
                confidence.tolist(), max_gt_iou
            ),
            "confidence_vs_max_gt_iou_spearman": pearson(
                rank_values(confidence.tolist()), rank_values(max_gt_iou)
            ),
        },
        "matched_budget": {},
        "matched_budget_within_fpn_level": {},
    }

    ranking_values = {"confidence": confidence}
    for radius_index, radius in enumerate(radii):
        ambiguity = torch.tensor([
            record["ambiguity"][radius_index] for record in records
        ], dtype=torch.float64)
        finite_ambiguity = torch.isfinite(ambiguity)
        npl_score = torch.zeros_like(confidence)
        npl_score[finite_ambiguity] = confidence[finite_ambiguity] / (
            ambiguity[finite_ambiguity].clamp_min(1e-6)
        )
        ranking_values["npl_radius_{}".format(radius)] = npl_score
        finite_indices = torch.where(finite_ambiguity)[0].tolist()
        finite_ambiguity_values = ambiguity[finite_ambiguity].tolist()
        finite_gt_iou = [max_gt_iou[index] for index in finite_indices]
        neighbor_counts = [
            record["neighbor_counts"][radius_index] for record in records
        ]
        result["correlation"]["radius_{}".format(radius)] = {
            "finite_rate": ratio(len(finite_indices), len(records)),
            "zero_ambiguity_rate": ratio(
                int((ambiguity == 0).sum()), len(records)
            ),
            "mean_neighbor_count": ratio(
                sum(neighbor_counts), len(neighbor_counts)
            ),
            "ambiguity_vs_max_gt_iou_pearson": pearson(
                finite_ambiguity_values, finite_gt_iou
            ),
            "ambiguity_vs_max_gt_iou_spearman": pearson(
                rank_values(finite_ambiguity_values),
                rank_values(finite_gt_iou),
            ),
            "npl_score_vs_max_gt_iou_pearson": pearson(
                npl_score.tolist(), max_gt_iou
            ),
            "npl_score_vs_max_gt_iou_spearman": pearson(
                rank_values(npl_score.tolist()), rank_values(max_gt_iou)
            ),
        }

    for fraction in fractions:
        fraction_key = "{:.2f}".format(fraction)
        result["matched_budget"][fraction_key] = {}
        result["matched_budget_within_fpn_level"][fraction_key] = {}
        expected_count = min(
            len(records), max(1, math.ceil(len(records) * fraction))
        )
        for name, values in ranking_values.items():
            indices = matched_budget_indices(values, fraction).tolist()
            if len(indices) != expected_count:
                raise RuntimeError("matched-budget selection size drifted")
            result["matched_budget"][fraction_key][name] = summarize_selection(
                records, indices, total_gt_segments, total_gt_anchors,
                tie_eps, near_same_l1,
            )
            stratified_indices = select_within_fpn_level(
                records, values, fraction
            )
            result["matched_budget_within_fpn_level"][
                fraction_key
            ][name] = summarize_selection(
                records, stratified_indices,
                total_gt_segments, total_gt_anchors,
                tie_eps, near_same_l1,
            )
    return result


def summarize_anchor_selection(
        records, selected_indices, total_gt_anchors, tie_eps, near_same_l1):
    selected = [records[index] for index in selected_indices]
    foreground = sum(record["foreground"] for record in selected)
    good_boundary = sum(record["good_boundary"] for record in selected)
    deltas = [
        record["delta_iou"] for record in selected
        if record["delta_iou"] is not None
    ]
    offset_l1 = [
        record["offset_l1"] for record in selected
        if record["offset_l1"] is not None
    ]
    teacher_wins = sum(value > tie_eps for value in deltas)
    student_wins = sum(value < -tie_eps for value in deltas)
    ties = len(deltas) - teacher_wins - student_wins
    level_counts = {}
    for record in selected:
        key = str(record["anchor_level"])
        level_counts[key] = level_counts.get(key, 0) + 1
    return {
        "pseudo_positive": len(selected),
        "student_anchor_level_counts": level_counts,
        "foreground_precision": ratio(foreground, len(selected)),
        "good_boundary_precision_iou50": ratio(
            good_boundary, len(selected)
        ),
        "gt_foreground_anchor_coverage": ratio(
            foreground, total_gt_anchors
        ),
        "good_boundary_anchor_coverage": ratio(
            good_boundary, total_gt_anchors
        ),
        "teacher_vs_student_on_gt_foreground_anchors": {
            "n": len(deltas),
            "teacher_win_rate": ratio(teacher_wins, len(deltas)),
            "student_win_rate": ratio(student_wins, len(deltas)),
            "tie_rate": ratio(ties, len(deltas)),
            "mean_delta_iou": ratio(sum(deltas), len(deltas)),
            "mean_teacher_student_boundary_l1": ratio(
                sum(offset_l1), len(offset_l1)
            ),
            "near_same_prediction_rate": ratio(
                sum(value <= near_same_l1 for value in offset_l1),
                len(offset_l1),
            ),
        },
    }


def summarize_anchor_rankings(
        records, radii, fractions, total_gt_anchors,
        tie_eps, near_same_l1):
    if not records:
        return {"records": 0, "matched_budget": {}}
    confidence = torch.tensor(
        [record["score"] for record in records], dtype=torch.float64
    )
    ranking_values = {"confidence": confidence}
    for radius_index, radius in enumerate(radii):
        ambiguity = torch.tensor([
            record["ambiguity"][radius_index] for record in records
        ], dtype=torch.float64)
        npl_score = torch.zeros_like(confidence)
        finite = torch.isfinite(ambiguity)
        npl_score[finite] = confidence[finite] / (
            ambiguity[finite].clamp_min(1e-6)
        )
        ranking_values["npl_radius_{}".format(radius)] = npl_score

    output = {
        "records": len(records),
        "matched_budget": {},
        "matched_budget_within_student_fpn_level": {},
    }
    for fraction in fractions:
        key = "{:.2f}".format(fraction)
        output["matched_budget"][key] = {}
        output["matched_budget_within_student_fpn_level"][key] = {}
        expected_count = min(
            len(records), max(1, math.ceil(len(records) * fraction))
        )
        for name, values in ranking_values.items():
            indices = matched_budget_indices(values, fraction).tolist()
            if len(indices) != expected_count:
                raise RuntimeError("anchor matched-budget size drifted")
            output["matched_budget"][key][name] = summarize_anchor_selection(
                records, indices, total_gt_anchors,
                tie_eps, near_same_l1,
            )
            stratified_indices = select_within_fpn_level(
                records, values, fraction, level_key="anchor_level"
            )
            output["matched_budget_within_student_fpn_level"][
                key
            ][name] = summarize_anchor_selection(
                records, stratified_indices, total_gt_anchors,
                tie_eps, near_same_l1,
            )
    return output


def load_models(cfg, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    teacher = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    student.load_state_dict(checkpoint["state_dict"])
    teacher.load_state_dict(checkpoint["ema_state_dict"])
    student.eval()
    teacher.eval()
    return checkpoint, student, teacher


def audit_checkpoint(
        checkpoint_path, cfg, dataset, annotations, device, batch_size,
        levels, radii, fractions, seed, tie_eps, near_same_l1):
    checkpoint, student, teacher = load_models(cfg, checkpoint_path, device)
    reset_audit_rng(seed)
    records = []
    anchor_records = []
    total_gt_segments = 0
    total_gt_anchors = 0
    videos = 0
    videos_with_proposals = 0

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_videos]
            post_nms, _ = teacher.generate_stage2_pseudo_targets(
                batch,
                return_pre_nms=True,
                return_npl_details=True,
                npl_neighbor_radii=tuple(radii),
            )

            inputs, masks = student.preprocessing(batch, mode="train")
            backbone_feats, backbone_masks, _ = student.backbone_target(
                inputs, masks
            )
            fpn_feats, fpn_masks = student.neck_target(
                backbone_feats, backbone_masks
            )
            student_offsets, _ = student.reg_head_target(
                fpn_feats, fpn_masks
            )
            student_offsets = torch.cat([
                offsets.permute(0, 2, 1) for offsets in student_offsets
            ], dim=1)
            points = student.point_generator(fpn_feats)
            valid = torch.cat([
                mask.squeeze(1) for mask in fpn_masks
            ], dim=1)
            enabled = torch.cat([
                torch.full(
                    (len(batch), point.shape[0]), level in levels,
                    dtype=torch.bool, device=device,
                )
                for level, point in enumerate(points)
            ], dim=1)
            centers = torch.cat([point[:, 0] for point in points])
            strides = torch.cat([point[:, 3] for point in points])
            anchor_levels = torch.cat([
                torch.full(
                    (point.shape[0],), level,
                    dtype=torch.long, device=device,
                )
                for level, point in enumerate(points)
            ])

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, batch):
                video_id = raw_video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id], video["metadata"],
                    video["feats"].shape[-1], cfg, device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)

            gt_cls, gt_offsets = student.label_points(
                points, gt_segments, gt_labels
            )
            pseudo_cls, pseudo_offsets, assignments = student.label_points(
                points,
                [item["segments"] for item in post_nms],
                [item["labels"] for item in post_nms],
                return_assignments=True,
            )

            for index, (gt, gt_label, post) in enumerate(zip(
                    gt_segments, gt_labels, post_nms)):
                video_key = batch_start + index
                videos += 1
                videos_with_proposals += int(post["segments"].numel() > 0)
                total_gt_segments += int(gt.shape[0])

                gt_positive = (
                    (gt_cls[index].sum(dim=-1) > 0)
                    & valid[index] & enabled[index]
                )
                total_gt_anchors += int(gt_positive.sum())
                pseudo_positive = (
                    (pseudo_cls[index].sum(dim=-1) > 0)
                    & valid[index] & enabled[index]
                )
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
                anchor_iou = torch.zeros_like(
                    centers, dtype=torch.float32
                )
                if overlap.any():
                    anchor_iou[overlap] = aligned_iou(
                        pseudo_segments[overlap],
                        gt_anchor_segments[overlap],
                    )

                proposal_iou = pairwise_iou(post["segments"], gt)
                assignment = assignments[index]
                for proposal_index in range(post["segments"].shape[0]):
                    if gt.numel():
                        max_iou, matched_index = proposal_iou[
                            proposal_index
                        ].max(dim=0)
                        max_iou = float(max_iou)
                        matched_index = int(matched_index)
                        class_correct = (
                            max_iou >= 0.5
                            and int(post["labels"][proposal_index])
                            == int(gt_label[matched_index])
                        )
                        matched_gt_keys = [
                            (video_key, gt_index)
                            for gt_index in torch.where(
                                proposal_iou[proposal_index] >= 0.5
                            )[0].tolist()
                        ]
                    else:
                        max_iou = 0.0
                        class_correct = False
                        matched_gt_keys = []

                    assigned = (
                        pseudo_positive & (assignment == proposal_index)
                    )
                    assigned_foreground = assigned & gt_positive
                    delta_iou, offset_l1 = [], []
                    delta_by_anchor = torch.full(
                        pseudo_positive.shape, float("nan"), device=device
                    )
                    l1_by_anchor = torch.full(
                        pseudo_positive.shape, float("nan"), device=device
                    )
                    if assigned_foreground.any():
                        teacher_iou = aligned_iou(
                            pseudo_segments[assigned_foreground],
                            gt_anchor_segments[assigned_foreground],
                        )
                        student_iou = aligned_iou(
                            student_segments[assigned_foreground],
                            gt_anchor_segments[assigned_foreground],
                        )
                        delta_iou = (
                            teacher_iou - student_iou
                        ).cpu().tolist()
                        offset_l1 = (
                            pseudo_segments[assigned_foreground]
                            - student_segments[assigned_foreground]
                        ).abs().mean(dim=-1).cpu().tolist()
                        delta_by_anchor[assigned_foreground] = (
                            teacher_iou - student_iou
                        )
                        l1_by_anchor[assigned_foreground] = (
                            pseudo_segments[assigned_foreground]
                            - student_segments[assigned_foreground]
                        ).abs().mean(dim=-1)

                    records.append({
                        "score": float(post["scores"][proposal_index]),
                        "level": int(post["levels"][proposal_index]),
                        "temporal_index": int(post["indices"][proposal_index]),
                        "ambiguity": post["npl_ambiguity"][
                            proposal_index
                        ].cpu().tolist(),
                        "neighbor_counts": post["npl_neighbor_counts"][
                            proposal_index
                        ].cpu().tolist(),
                        "proposal_max_gt_iou": max_iou,
                        "proposal_foreground": max_iou >= 0.5,
                        "proposal_class_correct": class_correct,
                        "matched_gt_keys": matched_gt_keys,
                        "assigned_anchors": int(assigned.sum()),
                        "foreground_anchors": int(
                            assigned_foreground.sum()
                        ),
                        "good_boundary_anchors": int(
                            (assigned & (anchor_iou >= 0.5)).sum()
                        ),
                        "delta_iou": delta_iou,
                        "offset_l1": offset_l1,
                    })
                    for anchor_index in torch.where(assigned)[0].tolist():
                        is_foreground = bool(gt_positive[anchor_index])
                        anchor_records.append({
                            "score": float(post["scores"][proposal_index]),
                            "ambiguity": post["npl_ambiguity"][
                                proposal_index
                            ].cpu().tolist(),
                            "anchor_level": int(anchor_levels[anchor_index]),
                            "foreground": is_foreground,
                            "good_boundary": bool(
                                anchor_iou[anchor_index] >= 0.5
                            ),
                            "delta_iou": (
                                float(delta_by_anchor[anchor_index])
                                if is_foreground else None
                            ),
                            "offset_l1": (
                                float(l1_by_anchor[anchor_index])
                                if is_foreground else None
                            ),
                        })

    return {
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "dataset": {
            "videos": videos,
            "videos_with_proposal": videos_with_proposals,
            "video_with_proposal_rate": ratio(
                videos_with_proposals, videos
            ),
            "gt_segments": total_gt_segments,
            "gt_foreground_anchors_fpn_0_2": total_gt_anchors,
        },
        **summarize_rankings(
            records, radii, fractions, total_gt_segments, total_gt_anchors,
            tie_eps, near_same_l1,
        ),
        "anchor_exact_budget": summarize_anchor_rankings(
            anchor_records, radii, fractions, total_gt_anchors,
            tie_eps, near_same_l1,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--neighbor-radii", default="1,2,4")
    parser.add_argument("--retain-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--tie-eps", type=float, default=1e-4)
    parser.add_argument("--near-same-l1", type=float, default=1e-2)
    args = parser.parse_args()

    levels = parse_numbers(args.levels, int)
    radii = parse_numbers(args.neighbor_radii, int)
    fractions = parse_numbers(args.retain_fractions, float)
    if not levels or len(levels) != len(set(levels)):
        raise ValueError("levels must be unique")
    if not radii or len(radii) != len(set(radii)) or any(
            radius < 1 for radius in radii):
        raise ValueError("neighbor radii must be unique positive integers")
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError("retain fractions must be in (0, 1]")

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {
        "enabled": True, "start_epoch": 5,
        "weight": 1.0, "levels": levels,
    }
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    batch_size = args.batch_size or int(
        cfg.get("loader", {}).get("batch_size", 1)
    )

    results = {}
    for name, checkpoint_path in parse_checkpoints(args.checkpoint):
        results[name] = audit_checkpoint(
            checkpoint_path, cfg, dataset, annotations, args.device,
            batch_size, levels, radii, fractions, args.seed,
            args.tie_eps, args.near_same_l1,
        )
    output = {
        "audit": "npl_style_target_gt_boundary_reliability",
        "candidate_pipeline": (
            "EMA dense predictions -> alpha -> hard class-agnostic NMS"
        ),
        "npl_ambiguity": (
            "(Var(adjacent start) + Var(adjacent end)) / proposal duration"
        ),
        "npl_ranking": "classification confidence / boundary ambiguity",
        "selection": "global exact matched budget per checkpoint",
        "levels": levels,
        "neighbor_radii": radii,
        "retain_fractions": fractions,
        "seed_for_identical_target_crops": args.seed,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
