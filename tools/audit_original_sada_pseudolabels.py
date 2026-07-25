"""Offline target-GT audit for the original SADA pseudo-label rule.

This script never feeds target labels to training, calibration, or checkpoint
selection. It only measures the pseudo-labels produced by a frozen checkpoint.
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


def move_video(video, device):
    result = dict(video)
    result["metadata"] = dict(video["metadata"])
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and key != "full_feats":
            result[key] = value.to(device)
    return result


def kept_label(label, action_subsampling):
    if action_subsampling == "10_majority_classes":
        return label < 10
    if action_subsampling == "20_random_classes":
        return label < 20
    return True


def crop_targets(annotation, metadata, crop_length, cfg, device):
    """Rebuild labels from raw JSON on the exact sampled crop.

    ``metadata['feat_num_frames']`` is mutable after truncation, so the
    feature-window size must come from the immutable dataset configuration.
    """
    fps = float(metadata["fps"])
    stride = float(metadata["feat_stride"])
    num_frames = float(cfg["dataset"]["num_frames"])
    crop_start = float(metadata.get("crop_start", 0))
    crop_end = crop_start + float(crop_length)
    trunc_thresh = float(cfg["dataset"]["trunc_thresh"])
    action_subsampling = cfg["dataset"].get("action_subsampling", "all")

    segments, labels = [], []
    for action in annotation["annotations"]:
        label = int(action["label_id"])
        if not kept_label(label, action_subsampling):
            continue
        left = (float(action["segment"][0]) * fps - 0.5 * num_frames) / stride
        right = (float(action["segment"][1]) * fps - 0.5 * num_frames) / stride
        segments.append((left, right))
        labels.append(label)

    if not segments:
        return (
            torch.empty((0, 2), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    # Match EpicKitchensDataset: sort the complete CPU tensor before applying
    # the crop. Sorting only retained segments changes tie ordering.
    segments = torch.tensor(segments, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    order = torch.argsort(segments[:, 0])
    segments = segments[order]
    labels = labels[order]

    # truncate_feats returns early when its sampled crop spans the entire
    # feature sequence, preserving out-of-range segment endpoints.
    if crop_length == int(metadata["original_feat_len"]):
        return segments.to(device), labels.to(device)

    window = torch.tensor([crop_start, crop_end], dtype=torch.float32)
    window = window.unsqueeze(0).repeat(segments.shape[0], 1)
    left = torch.maximum(window[:, 0], segments[:, 0])
    right = torch.minimum(window[:, 1], segments[:, 1])
    intersection = (right - left).clamp(min=0)
    segment_area = torch.abs(segments[:, 1] - segments[:, 0])
    retained = intersection / segment_area >= trunc_thresh
    cropped_segments = torch.stack((left[retained], right[retained]), dim=1)
    return cropped_segments.sub(crop_start).to(device), labels[retained].to(device)


def new_counter():
    return collections.Counter()


def add_group(counter, name, mask, gt_action, correct):
    count = int(mask.sum().item())
    if not count:
        return
    counter[name + "_n"] += count
    counter[name + "_gt_action"] += int((mask & gt_action).sum().item())
    counter[name + "_correct_class"] += int((mask & correct).sum().item())


def summarize(counter):
    output = {}
    for name in ("valid", "selected_action", "background"):
        count = counter[name + "_n"]
        output[name] = {
            "n": count,
            "gt_action_rate": (
                counter[name + "_gt_action"] / count if count else None
            ),
            "correct_class_rate": (
                counter[name + "_correct_class"] / count if count else None
            ),
        }
    gt_actions = counter["valid_gt_action"]
    output["selected_correct_coverage"] = (
        counter["selected_action_correct_class"] / max(gt_actions, 1)
    )
    return output


def score_bin(score):
    lower = min(int(score * 10), 9) / 10.0
    return "score_{:.1f}_{:.1f}".format(lower, lower + 0.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--pseudo-mode",
        choices=("train", "eval"),
        default="train",
        help="Use train to reproduce the original training-time pseudo-label path.",
    )
    parser.add_argument(
        "--verify-crop-targets",
        action="store_true",
        help="Assert reconstructed raw-JSON targets equal the dataset's cropped targets.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    split = cfg["tgt_train_split"] if args.split == "train" else cfg["tgt_val_split"]
    dataset = make_dataset(
        cfg["dataset_name"], args.split == "train", split, **cfg["dataset"]
    )
    annotations = json.load(open(cfg["dataset"]["json_file"], encoding="utf-8"))["database"]

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["state_dict"])
    if args.pseudo_mode == "train":
        model.train()
    else:
        model.eval()

    threshold = float(cfg["model"].get("pl_threshold", 0.7))
    totals = collections.defaultdict(new_counter)
    score_totals = collections.defaultdict(new_counter)
    class_totals = collections.defaultdict(new_counter)
    with torch.no_grad():
        for raw_video in dataset:
            video_id = raw_video["metadata"]["video_id"]
            video = move_video(raw_video, args.device)
            inputs, masks = model.preprocessing([video], mode=args.pseudo_mode)
            inputs = model.dropout(inputs)
            backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
            logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
            points = model.point_generator(fpn_feats)

            gt_segments, gt_labels = crop_targets(
                annotations[video_id],
                video["metadata"],
                video["feats"].shape[-1],
                cfg,
                args.device,
            )
            if args.verify_crop_targets:
                expected_segments = raw_video["segments"].to(args.device)
                expected_labels = raw_video["labels"].to(args.device)
                if (
                    gt_segments.shape != expected_segments.shape
                    or not torch.allclose(gt_segments, expected_segments, atol=1e-4, rtol=1e-5)
                    or not torch.equal(gt_labels, expected_labels)
                ):
                    raise RuntimeError(
                        "Reconstructed target GT differs from dataset crop for {}".format(video_id)
                    )
            gt_cls, _ = model.label_points(points, [gt_segments], [gt_labels])
            start = 0
            for level, point_grid in enumerate(points):
                length = point_grid.shape[0]
                valid = fpn_masks[level][0].squeeze(0)
                probabilities = logits[level][0].transpose(0, 1).sigmoid()
                scores, pseudo_class = probabilities.max(dim=-1)
                selected = valid & (scores >= threshold)
                background = valid & ~selected
                labels = gt_cls[0][start:start + length]
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, pseudo_class.unsqueeze(-1)
                ).squeeze(-1) > 0

                masks_to_audit = {
                    "valid": valid,
                    "selected_action": selected,
                    "background": background,
                }
                for bucket_name in ("global", "level_{}".format(level)):
                    counter = totals[bucket_name]
                    for name, mask in masks_to_audit.items():
                        add_group(counter, name, mask, gt_action, correct)
                for score, is_selected, is_action, is_correct in zip(
                    scores[valid].tolist(),
                    selected[valid].tolist(),
                    gt_action[valid].tolist(),
                    correct[valid].tolist(),
                ):
                    if is_selected:
                        counter = score_totals[score_bin(score)]
                        counter["n"] += 1
                        counter["gt_action"] += int(is_action)
                        counter["correct_class"] += int(is_correct)
                for cls, is_action, is_correct in zip(
                    pseudo_class[selected].tolist(),
                    gt_action[selected].tolist(),
                    correct[selected].tolist(),
                ):
                    counter = class_totals[str(cls)]
                    counter["n"] += 1
                    counter["gt_action"] += int(is_action)
                    counter["correct_class"] += int(is_correct)
                start += length

    def summarize_simple(counters):
        return {
            key: {
                "n": value["n"],
                "gt_action_rate": value["gt_action"] / max(value["n"], 1),
                "correct_class_rate": value["correct_class"] / max(value["n"], 1),
            }
            for key, value in counters.items()
        }

    result = {
        "audit": "original_sada_threshold_pseudolabels",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "seed": args.seed,
        "videos": len(dataset),
        "method": {
            "score": "max(sigmoid(class_logits))",
            "action_rule": "score >= pl_threshold",
            "background_rule": "all valid anchors not selected as action",
            "pseudo_mode": args.pseudo_mode,
            "pl_threshold": threshold,
            "align_bkg": bool(cfg["model"].get("align_bkg", False)),
            "raw_num_frames": cfg["dataset"]["num_frames"],
            "action_subsampling": cfg["dataset"].get("action_subsampling", "all"),
            "input_noise": cfg["model"].get("input_noise", 0.0),
            "dropout": cfg["model"].get("dropout", 0.0),
            "droppath": cfg["model"].get("droppath", 0.0),
        },
        "groups": {name: summarize(counter) for name, counter in totals.items()},
        "selected_score_bins": summarize_simple(score_totals),
        "selected_pseudo_classes": summarize_simple(class_totals),
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["groups"]["global"], indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
