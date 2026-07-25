"""Offline test of whether shift stability purifies SADA action pseudo-labels.

Target GT is used only after a frozen model has produced detached routing
records. The audit compares nested action gates on the same target crops and
common valid FPN anchors:

    high confidence -> class consistent -> low confidence variation
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage1_utils import (
    interpolate_probabilities,
    population_std,
    shift_features,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


RETENTION_FRACTIONS = (1.0, 0.8, 0.6, 0.4, 0.2)


def summarize_group(mask, gt_action, correct, valid_gt_actions):
    count = int(mask.sum().item())
    return {
        "n": count,
        "selection_rate": count / max(mask.numel(), 1),
        "gt_action_rate": (
            float(gt_action[mask].float().mean().item()) if count else None
        ),
        "correct_class_rate": (
            float(correct[mask].float().mean().item()) if count else None
        ),
        "selected_correct_coverage": (
            int(correct[mask].sum().item()) / max(valid_gt_actions, 1)
        ),
    }


def summarize_route(action, gt_action, correct):
    valid_gt_actions = int(gt_action.sum().item())
    background = ~action
    result = summarize_group(action, gt_action, correct, valid_gt_actions)
    result["background_n"] = int(background.sum().item())
    result["background_gt_action_rate"] = (
        float(gt_action[background].float().mean().item())
        if background.any()
        else None
    )
    return result


def build_summary(records):
    score = records["score"]
    consistent = records["consistent"]
    stability = records["stability"]
    gt_action = records["gt_action"]
    correct = records["correct"]
    alpha = records["alpha"]

    high = score >= alpha
    class_consistent = high & consistent
    class_flip = high & ~consistent
    valid_gt_actions = int(gt_action.sum().item())

    result = {
        "valid": {
            "n": int(score.numel()),
            "gt_action_rate": float(gt_action.float().mean().item()),
        },
        "high_confidence": summarize_route(high, gt_action, correct),
        "class_consistent": summarize_route(
            class_consistent, gt_action, correct
        ),
        "class_flip_rejected": summarize_group(
            class_flip, gt_action, correct, valid_gt_actions
        ),
        "stability_retention": {},
    }

    candidate_indices = torch.where(class_consistent)[0]
    candidate_stability = stability[candidate_indices]
    if not candidate_indices.numel():
        return result

    result["class_consistent_stability"] = {
        "min": float(candidate_stability.min().item()),
        "median": float(candidate_stability.median().item()),
        "mean": float(candidate_stability.mean().item()),
        "max": float(candidate_stability.max().item()),
        "exact_zero_rate": float((candidate_stability == 0).float().mean().item()),
    }

    sorted_stability, order = torch.sort(candidate_stability)
    sorted_indices = candidate_indices[order]
    for fraction in RETENTION_FRACTIONS:
        keep = max(1, int(math.ceil(fraction * sorted_indices.numel())))
        threshold = sorted_stability[keep - 1]
        reliable = class_consistent & (stability <= threshold)
        rejected = class_consistent & ~reliable
        key = "retain_{:03d}".format(int(round(fraction * 100)))
        result["stability_retention"][key] = {
            "diagnostic_tau": float(threshold.item()),
            "reliable_action": summarize_route(
                reliable, gt_action, correct
            ),
            "stability_rejected": summarize_group(
                rejected, gt_action, correct, valid_gt_actions
            ),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--delta", type=int, choices=(-2, 2), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.load(
        open(cfg["dataset"]["json_file"], encoding="utf-8")
    )["database"]
    batch_size = int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(
        args.device
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    record_lists = {
        key: []
        for key in (
            "score", "consistent", "stability", "gt_action", "correct", "level"
        )
    }
    ordinary_valid_n = 0
    common_valid_n = 0
    alpha = float(cfg["model"].get("pl_threshold", 0.7))

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            inputs, masks = model.preprocessing(videos, mode="validation")
            shifted_inputs, shifted_masks = shift_features(
                inputs, masks, args.delta
            )

            strict0 = model._stage1_strict_masks(masks)
            strict1 = model._stage1_strict_masks(shifted_masks)
            feats0, backbone_masks0, _ = model.backbone_target(inputs, masks)
            fpn0, fpn_masks0 = model.neck_target(feats0, backbone_masks0)
            logits0, _ = model.cls_head_target(fpn0, fpn_masks0)
            feats1, backbone_masks1, _ = model.backbone_target(
                shifted_inputs, shifted_masks
            )
            fpn1, fpn_masks1 = model.neck_target(feats1, backbone_masks1)
            logits1, _ = model.cls_head_target(fpn1, fpn_masks1)
            points = model.point_generator(fpn0)

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    args.device,
                )
                if not torch.equal(
                    segments, raw_video["segments"].to(args.device)
                ) or not torch.equal(
                    labels, raw_video["labels"].to(args.device)
                ):
                    raise RuntimeError(
                        "Target GT reconstruction mismatch for {}".format(video_id)
                    )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, _ = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)

            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                prob0 = logits0[level].permute(0, 2, 1).sigmoid()
                prob1 = logits1[level].permute(0, 2, 1).sigmoid()
                query = point_grid[:, 0] + args.delta
                aligned_prob1, aligned_valid = interpolate_probabilities(
                    prob1, strict1[level].squeeze(1), point_grid, query
                )
                common_valid = strict0[level].squeeze(1) & aligned_valid
                ordinary_valid = fpn_masks0[level].squeeze(1)

                score0, class0 = prob0.max(dim=-1)
                _, class1 = aligned_prob1.max(dim=-1)
                class1_score = aligned_prob1.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1)
                stability = population_std(
                    torch.stack((score0, class1_score), dim=0), dim=0
                )
                labels = gt_cls[:, start:start + length]
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, class0.unsqueeze(-1)
                ).squeeze(-1) > 0

                ordinary_valid_n += int(ordinary_valid.sum().item())
                common_valid_n += int(common_valid.sum().item())
                values = {
                    "score": score0,
                    "consistent": class0 == class1,
                    "stability": stability,
                    "gt_action": gt_action,
                    "correct": correct,
                    "level": torch.full_like(class0, level),
                }
                for key, value in values.items():
                    record_lists[key].append(value[common_valid].detach().cpu())
                start += length

    records = {key: torch.cat(value) for key, value in record_lists.items()}
    records["alpha"] = alpha
    per_level = {}
    for level in range(model.n_levels):
        selected = records["level"] == level
        level_records = {
            key: value[selected]
            for key, value in records.items()
            if key != "alpha"
        }
        level_records["alpha"] = alpha
        per_level[str(level)] = build_summary(level_records)

    result = {
        "audit": "shift_action_stability_direction_test",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "config": args.config,
        "split": "target_train",
        "seed": args.seed,
        "delta": args.delta,
        "videos": len(dataset),
        "batch_size": batch_size,
        "alpha": alpha,
        "model_mode": "eval",
        "comparison_scope": "same common strict-valid target FPN anchors",
        "ordinary_valid_n": ordinary_valid_n,
        "common_valid_n": common_valid_n,
        "common_valid_ratio": common_valid_n / max(ordinary_valid_n, 1),
        "global": build_summary(records),
        "per_level": per_level,
    }
    Path(args.output).write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["global"], indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
