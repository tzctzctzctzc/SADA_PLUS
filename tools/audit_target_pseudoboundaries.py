"""Offline boundary audit for original-SADA target pseudo-label candidates.

The model is a frozen EMA teacher in evaluation mode. Target annotations are
read only after predictions have been produced and are never passed into the
model. This measures whether confidence-selected action anchors carry usable
boundary supervision before any target regression experiment is enabled.
"""

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


def segment_iou(prediction, target):
    """Return pairwise-aligned 1-D IoU for ``N x 2`` segments."""
    left = torch.maximum(prediction[:, 0], target[:, 0])
    right = torch.minimum(prediction[:, 1], target[:, 1])
    intersection = (right - left).clamp_min(0)
    union = (
        (prediction[:, 1] - prediction[:, 0]).clamp_min(0)
        + (target[:, 1] - target[:, 0]).clamp_min(0)
        - intersection
    )
    return intersection / union.clamp_min(1e-6)


def add_counter(counter, valid, selected, gt_action, correct, iou):
    """Accumulate routing and boundary statistics for one disjoint group."""
    counter["valid_n"] += int(valid.sum().item())
    counter["valid_gt_action_n"] += int((valid & gt_action).sum().item())
    counter["selected_n"] += int(selected.sum().item())
    counter["selected_gt_action_n"] += int((selected & gt_action).sum().item())
    counter["selected_correct_class_n"] += int((selected & correct).sum().item())

    selected_foreground = selected & gt_action
    selected_iou = iou[selected_foreground]
    if selected_iou.numel():
        counter["boundary_n"] += int(selected_iou.numel())
        counter["boundary_iou_sum"] += float(selected_iou.sum().item())
        counter.setdefault("boundary_iou_values", []).extend(
            selected_iou.detach().cpu().tolist()
        )
        for threshold in (0.1, 0.3, 0.5, 0.7):
            counter["boundary_iou_ge_{:.1f}".format(threshold)] += int(
                (selected_iou >= threshold).sum().item()
            )


def summarize(counter):
    selected_n = counter["selected_n"]
    valid_gt_action_n = counter["valid_gt_action_n"]
    boundary_n = counter["boundary_n"]
    values = counter["boundary_iou_values"]
    result = {
        "valid_n": counter["valid_n"],
        "valid_gt_action_n": valid_gt_action_n,
        "selected_n": selected_n,
        "selected_gt_action_rate": (
            counter["selected_gt_action_n"] / selected_n if selected_n else None
        ),
        "selected_correct_class_rate": (
            counter["selected_correct_class_n"] / selected_n if selected_n else None
        ),
        "selected_correct_class_coverage": (
            counter["selected_correct_class_n"] / valid_gt_action_n
            if valid_gt_action_n
            else None
        ),
        "selected_foreground_boundary_n": boundary_n,
        "boundary_iou_mean": (
            counter["boundary_iou_sum"] / boundary_n if boundary_n else None
        ),
        "boundary_iou_median": (
            float(torch.tensor(values).median().item()) if values else None
        ),
    }
    for threshold in (0.1, 0.3, 0.5, 0.7):
        key = "boundary_iou_ge_{:.1f}".format(threshold)
        result[key] = counter[key] / boundary_n if boundary_n else None
    return result


def score_bin(score, alpha):
    lower = max(alpha, min(int(score * 10), 9) / 10.0)
    return "score_{:.1f}_{:.1f}".format(lower, min(lower + 0.1, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--verify-crop-targets", action="store_true")
    args = parser.parse_args()

    levels = [int(level.strip()) for level in args.levels.split(",") if level.strip()]
    if not levels or len(set(levels)) != len(levels) or any(level < 0 or level > 5 for level in levels):
        raise ValueError("levels must be unique FPN levels from 0 to 5")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], True, cfg["tgt_train_split"], **cfg["dataset"]
    )
    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if "ema_state_dict" not in checkpoint:
        raise ValueError("checkpoint has no ema_state_dict; refusing non-EMA boundary audit")
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()

    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    totals = collections.defaultdict(collections.Counter)
    score_totals = collections.defaultdict(collections.Counter)
    videos_checked = 0
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            inputs, masks = model.preprocessing(videos, mode="eval")
            backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
            logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
            offsets, _ = model.reg_head_target(fpn_feats, fpn_masks)
            points = model.point_generator(fpn_feats)

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = raw_video["metadata"]["video_id"]
                segments, labels = crop_targets(
                    annotations[video_id],
                    video["metadata"],
                    video["feats"].shape[-1],
                    cfg,
                    args.device,
                )
                if args.verify_crop_targets:
                    if (
                        segments.shape != raw_video["segments"].shape
                        or not torch.equal(segments, raw_video["segments"].to(args.device))
                        or not torch.equal(labels, raw_video["labels"].to(args.device))
                    ):
                        raise RuntimeError("reconstructed target GT differs for {}".format(video_id))
                gt_segments.append(segments)
                gt_labels.append(labels)
                videos_checked += 1
            gt_cls, gt_offsets = model.label_points(points, gt_segments, gt_labels)
            gt_cls = torch.stack(gt_cls)
            gt_offsets = torch.stack(gt_offsets)

            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                if level not in levels:
                    start += length
                    continue
                valid = fpn_masks[level].squeeze(1)
                probabilities = logits[level].transpose(1, 2).sigmoid()
                scores, pseudo_class = probabilities.max(dim=-1)
                selected = valid & (scores >= alpha)
                labels = gt_cls[:, start:start + length]
                target_offsets = gt_offsets[:, start:start + length]
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(-1, pseudo_class.unsqueeze(-1)).squeeze(-1) > 0

                stride = point_grid[:, 3].view(1, length)
                centers = point_grid[:, 0].view(1, length)
                predicted_segments = torch.stack(
                    (centers - offsets[level].transpose(1, 2)[..., 0] * stride,
                     centers + offsets[level].transpose(1, 2)[..., 1] * stride),
                    dim=-1,
                )
                target_segments = torch.stack(
                    (centers - target_offsets[..., 0] * stride,
                     centers + target_offsets[..., 1] * stride),
                    dim=-1,
                )
                iou = segment_iou(
                    predicted_segments.reshape(-1, 2), target_segments.reshape(-1, 2)
                ).reshape_as(scores)

                add_counter(totals["global"], valid, selected, gt_action, correct, iou)
                add_counter(totals["level_{}".format(level)], valid, selected, gt_action, correct, iou)
                for score, chosen, action, is_correct, item_iou in zip(
                    scores[valid].tolist(),
                    selected[valid].tolist(),
                    gt_action[valid].tolist(),
                    correct[valid].tolist(),
                    iou[valid].tolist(),
                ):
                    if chosen:
                        counter = score_totals[score_bin(score, alpha)]
                        counter["selected_n"] += 1
                        counter["selected_gt_action_n"] += int(action)
                        counter["selected_correct_class_n"] += int(is_correct)
                        if action:
                            counter["boundary_n"] += 1
                            counter["boundary_iou_sum"] += item_iou
                            counter.setdefault("boundary_iou_values", []).append(item_iou)
                            for threshold in (0.1, 0.3, 0.5, 0.7):
                                counter["boundary_iou_ge_{:.1f}".format(threshold)] += int(item_iou >= threshold)
                start += length

    result = {
        "audit": "original_sada_ema_target_pseudoboundaries",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "seed": args.seed,
        "videos": videos_checked,
        "batch_size": batch_size,
        "method": {
            "teacher": "checkpoint.ema_state_dict",
            "model_mode": "eval",
            "levels": levels,
            "candidate_rule": "valid and max(sigmoid(class_logits)) >= pl_threshold",
            "target_gt_use": "post-prediction offline metric only",
            "pl_threshold": alpha,
        },
        "groups": {name: summarize(counter) for name, counter in totals.items()},
        "selected_score_bins": {name: summarize(counter) for name, counter in score_totals.items()},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["groups"]["global"], indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
