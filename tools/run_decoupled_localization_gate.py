"""Source-gated class-agnostic direct localization stream.

The parent detector and its semantic predictions stay frozen.  A temporal
convolutional head is trained from source GT to predict direct left/right
offsets plus localization quality.  Epoch and score fusion are selected on
source validation; target is instantiated only when both source gates pass.
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from libs.datasets import make_dataset
from libs.models.SADA.decoupled_localization import (
    DirectLocalizationHead,
    balanced_soft_quality_loss,
    fuse_class_and_quality,
    initialize_from_trident_regression_trunk,
)
from libs.models.SADA.losses_aux import ctr_diou_loss_1d, ctr_giou_loss_1d
from libs.models.SADA.vlm_routing import file_sha256
from tools.run_proposal_iou_quality_gate import (
    _append_predictions,
    _domain_outputs,
    _empty_predictions,
    _finalize_predictions,
    evaluator_for,
    load_config,
    load_detector,
    report_to_json,
)


PROTOCOL_VERSION = "decoupled-class-agnostic-localization-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--trident-head", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", default=(5, 10, 20))
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--quality-loss-weight", type=float, default=0.5)
    parser.add_argument("--init-trunk-from-parent", action="store_true")
    parser.add_argument("--quality-gammas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--min-source-map-gain", type=float, default=0.005)
    parser.add_argument("--min-source-high-tiou-gain", type=float, default=0.005)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-eval-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


@torch.inference_mode()
def extract_source_examples(model, dataset, device, limit=None):
    examples = []
    count = min(len(dataset), limit or len(dataset))
    positive_points = 0
    valid_points = 0
    for index in range(count):
        video, fpn, masks, _, _, _, _, points = _domain_outputs(
            model, dataset[index], "source", device
        )
        gt_classes, gt_offsets = model.label_points(
            points, [video["segments"]], [video["labels"]]
        )
        lengths = [features.shape[-1] for features in fpn]
        split_classes = torch.split(gt_classes[0], lengths)
        split_offsets = torch.split(gt_offsets[0], lengths)
        positive_points += sum(int(classes.sum(-1).gt(0).sum()) for classes in split_classes)
        valid_points += sum(int(mask.sum()) for mask in masks)
        examples.append(
            {
                "features": [value.cpu().half() for value in fpn],
                "masks": [value.cpu().bool() for value in masks],
                "positive": [value.sum(-1).gt(0).cpu() for value in split_classes],
                "offsets": [value.cpu().float() for value in split_offsets],
            }
        )
        if (index + 1) % 10 == 0 or index + 1 == count:
            print("source extraction {}/{}".format(index + 1, count), flush=True)
    return examples, positive_points, valid_points


def _example_to_device(example, device):
    return {
        "features": [value.to(device, dtype=torch.float32) for value in example["features"]],
        "masks": [value.to(device) for value in example["masks"]],
        "positive": [value.to(device) for value in example["positive"]],
        "offsets": [value.to(device) for value in example["offsets"]],
    }


def localization_loss(predicted_offsets, quality_logits, example, quality_weight):
    predicted = torch.cat([value.permute(0, 2, 1) for value in predicted_offsets], dim=1)[0]
    quality = torch.cat([value.permute(0, 2, 1) for value in quality_logits], dim=1)[0, :, 0]
    targets = torch.cat(example["offsets"], dim=0)
    positive = torch.cat(example["positive"], dim=0)
    valid = torch.cat([value[0, 0] for value in example["masks"]], dim=0)
    positive = positive & valid
    if not positive.any():
        return predicted.sum() * 0.0, predicted.sum() * 0.0, quality.sum() * 0.0
    regression = ctr_diou_loss_1d(predicted[positive], targets[positive], reduction="mean")
    soft_targets = torch.zeros_like(quality)
    soft_targets[positive] = 1.0 - ctr_giou_loss_1d(
        predicted[positive].detach(), targets[positive], reduction="none"
    )
    quality_loss = balanced_soft_quality_loss(quality, soft_targets, positive, valid)
    return regression + quality_weight * quality_loss, regression, quality_loss


def fit_heads(examples, input_dim, fpn_levels, args, parent_reg_head=None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    head = DirectLocalizationHead(
        input_dim,
        args.hidden_dim,
        fpn_levels,
        args.num_layers,
        args.kernel_size,
        with_ln=True,
    ).to(args.device)
    if args.init_trunk_from_parent:
        if parent_reg_head is None:
            raise ValueError("parent regression head is required for trunk initialization")
        initialize_from_trident_regression_trunk(head, parent_reg_head)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    snapshots = {}
    snapshot_epochs = set(args.snapshot_epochs)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(examples)).tolist()
        totals = np.zeros(3, dtype=np.float64)
        head.train()
        for index in order:
            example = _example_to_device(examples[index], args.device)
            optimizer.zero_grad(set_to_none=True)
            offsets, quality = head(example["features"], example["masks"])
            loss, regression, quality_loss = localization_loss(
                offsets, quality, example, args.quality_loss_weight
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            totals += (float(loss), float(regression), float(quality_loss))
        print(
            "epoch {:02d}: total={:.6f} reg={:.6f} quality={:.6f}".format(
                epoch, *(totals / max(len(examples), 1))
            ),
            flush=True,
        )
        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(head).cpu().eval()
    if args.epochs not in snapshots:
        snapshots[args.epochs] = copy.deepcopy(head).cpu().eval()
    return snapshots


def _raw_direct_result(model, video, points, masks, logits, offsets, quality, gamma):
    segments = []
    scores = []
    labels = []
    qualities = []
    for point_level, mask_level, logit_level, offset_level, quality_level in zip(
        points, masks, logits, offsets, quality
    ):
        class_probabilities = logit_level[0].transpose(0, 1).sigmoid()
        class_probabilities *= mask_level[0, 0].unsqueeze(-1)
        quality_logits = quality_level[0].transpose(0, 1)
        fused = fuse_class_and_quality(class_probabilities, quality_logits, gamma)
        flat = fused.flatten()
        keep = flat > model.test_pre_nms_thresh
        flat = flat[keep]
        indices = keep.nonzero(as_tuple=True)[0]
        count = min(model.test_pre_nms_topk, len(indices))
        flat, order = flat.sort(descending=True)
        flat = flat[:count]
        indices = indices[order[:count]]
        point_indices = torch.div(
            indices, model.num_class_orig, rounding_mode="floor"
        )
        class_indices = torch.fmod(indices, model.num_class_orig)
        direct_offsets = offset_level[0].transpose(0, 1)[point_indices]
        selected_points = point_level[point_indices]
        left = selected_points[:, 0] - direct_offsets[:, 0] * selected_points[:, 3]
        right = selected_points[:, 0] + direct_offsets[:, 1] * selected_points[:, 3]
        valid_duration = right - left > model.test_duration_thresh
        segments.append(torch.stack((left, right), dim=-1)[valid_duration])
        scores.append(flat[valid_duration])
        labels.append(class_indices[valid_duration])
        qualities.append(quality_logits.sigmoid()[point_indices, 0][valid_duration])
    metadata = video["metadata"]
    return {
        "video_id": metadata["video_id"],
        "fps": metadata["fps"],
        "duration": metadata["duration"],
        "video_length": metadata["video_length"],
        "feat_stride": metadata["feat_stride"],
        "feat_num_frames": metadata["feat_num_frames"],
        "use_temporal_grid_feats": metadata["use_temporal_grid_feats"],
        "segments": torch.cat(segments),
        "scores": torch.cat(scores),
        "labels": torch.cat(labels),
        "qualities": torch.cat(qualities),
    }


@torch.inference_mode()
def evaluate_heads(model, heads, dataset, domain, cfg, args):
    variants = [(0, -1.0)] + [
        (epoch, gamma) for epoch in sorted(heads) for gamma in args.quality_gammas
    ]
    predictions = {variant: _empty_predictions() for variant in variants}
    device_heads = {
        epoch: copy.deepcopy(head).to(args.device).eval() for epoch, head in heads.items()
    }
    count = min(len(dataset), args.max_eval_videos or len(dataset))
    for index in range(count):
        video, fpn, masks, logits, parent_offsets, starts, ends, points = _domain_outputs(
            model, dataset[index], domain, args.device
        )
        original_trident = model.use_trident_head
        parent_result = model.inference(
            [video],
            points,
            [value.squeeze(1) for value in masks],
            [value.permute(0, 2, 1) for value in logits],
            [value.permute(0, 2, 1) for value in parent_offsets],
            starts if original_trident else None,
            ends if original_trident else None,
        )[0]
        _append_predictions(predictions[(0, -1.0)], parent_result)
        for epoch, head in device_heads.items():
            direct_offsets, quality = head(fpn, masks)
            for gamma in args.quality_gammas:
                raw = _raw_direct_result(
                    model, video, points, masks, logits, direct_offsets, quality, gamma
                )
                result = model.postprocessing([raw])[0]
                _append_predictions(predictions[(epoch, gamma)], result)
        if (index + 1) % 10 == 0 or index + 1 == count:
            print("{} evaluation {}/{}".format(domain, index + 1, count), flush=True)
    evaluator = evaluator_for(dataset, cfg)
    return {
        variant: {
            "mAP_by_tiou": [float(value) for value in metrics[0]],
            "average_mAP": float(metrics[1]),
        }
        for variant in variants
        for metrics in [
            evaluator.evaluate(_finalize_predictions(predictions[variant]), verbose=False)
        ]
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.trident_head:
        cfg["model"]["use_trident_head"] = True
    model, checkpoint = load_detector(
        cfg, args.checkpoint, args.weights, args.device
    )
    source_train = make_dataset(
        cfg["dataset_name"], False, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    examples, positive_points, valid_points = extract_source_examples(
        model, source_train, args.device, args.max_train_videos
    )
    input_dim = examples[0]["features"][0].shape[1]
    heads = fit_heads(
        examples, input_dim, len(model.fpn_strides), args, model.reg_head
    )
    source_reports = evaluate_heads(model, heads, source_val, "source", cfg, args)
    baseline = source_reports[(0, -1.0)]
    best_variant = max(
        (variant for variant in source_reports if variant != (0, -1.0)),
        key=lambda variant: (
            source_reports[variant]["average_mAP"],
            source_reports[variant]["mAP_by_tiou"][-1],
            -variant[1],
            -variant[0],
        ),
    )
    selected = source_reports[best_variant]
    average_gain = selected["average_mAP"] - baseline["average_mAP"]
    high_gain = selected["mAP_by_tiou"][-1] - baseline["mAP_by_tiou"][-1]
    gate_pass = (
        average_gain >= args.min_source_map_gain
        and high_gain >= args.min_source_high_tiou_gain
    )
    target_reports = None
    if gate_pass:
        target_val = make_dataset(
            cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
        )
        chosen_args = copy.copy(args)
        chosen_args.quality_gammas = (best_variant[1],)
        target_reports = evaluate_heads(
            model,
            {best_variant[0]: heads[best_variant[0]]},
            target_val,
            "target",
            cfg,
            chosen_args,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "input_dim": input_dim,
            "fpn_levels": len(model.fpn_strides),
            "selected_epoch": best_variant[0],
            "selected_gamma": best_variant[1],
            "state_dict": heads[best_variant[0]].state_dict(),
        },
        output,
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "target_gt_usage_for_selection": "forbidden",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "weights": args.weights,
        "source_train_videos": len(examples),
        "source_positive_points": positive_points,
        "source_valid_points": valid_points,
        "init_trunk_from_parent": args.init_trunk_from_parent,
        "selected_epoch": best_variant[0],
        "selected_gamma": best_variant[1],
        "source_baseline": baseline,
        "source_selected": selected,
        "source_average_mAP_gain": average_gain,
        "source_high_tiou_mAP_gain": high_gain,
        "source_gate_pass": gate_pass,
        "source_gate_thresholds": {
            "average_mAP_gain": args.min_source_map_gain,
            "high_tiou_mAP_gain": args.min_source_high_tiou_gain,
        },
        "source_variants": report_to_json(source_reports),
        "target_variants": None if target_reports is None else report_to_json(target_reports),
        "output_head": str(output),
    }
    output.with_suffix(output.suffix + ".report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
