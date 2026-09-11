"""Train a source-only proposal IoU head and gate target pre-NMS reranking.

The detector is frozen.  Head training uses source-train GT, model/gamma
selection uses source-validation mAP, and target GT is read only after the
source gate passes.
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
from libs.evaluation import ANETdetection
from libs.models.SADA import make_meta_arch
from libs.models.SADA.proposal_quality import (
    PROTOCOL_VERSION,
    ProposalIoUQualityHead,
    proposal_region_context,
    proposal_scalar_features,
    quality_weighted_scores,
    same_class_temporal_iou,
    soft_iou_quality_loss,
)
from libs.models.SADA.trident_quality import boundary_peak_quality
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
    parser.add_argument("--train-topk-per-level", type=int, default=128)
    parser.add_argument("--eval-topk-per-level", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshot-epochs", type=int, nargs="+", default=(5, 10, 20))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool-samples", type=int, default=4)
    parser.add_argument("--gammas", type=float, nargs="+", default=(0.25, 0.5, 1.0, 2.0))
    parser.add_argument("--min-source-map-gain", type=float, default=0.005)
    parser.add_argument("--min-source-high-tiou-gain", type=float, default=0.005)
    parser.add_argument("--max-train-videos", type=int, default=None)
    parser.add_argument("--max-eval-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def load_config(path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        cfg = json.loads(path.read_text(encoding="utf-8"))
    else:
        cfg = read_yaml(str(path))
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    return cfg


def load_detector(cfg, checkpoint_path, weights, device):
    model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_key = "ema_state_dict" if weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, checkpoint


def _trident_windows(model, starts, ends):
    start_windows = torch.nn.functional.pad(
        starts, (model.num_bins, 0)
    ).unsqueeze(-1)
    window_size = list(start_windows.size())
    window_size[-1] = model.num_bins + 1
    window_size[-2] -= model.num_bins
    window_stride = list(start_windows.stride())
    window_stride[-2] = window_stride[-1]
    start_windows = start_windows.as_strided(
        size=window_size, stride=window_stride
    )
    end_windows = torch.nn.functional.pad(
        ends, (0, model.num_bins)
    ).unsqueeze(-1)
    end_windows = end_windows.as_strided(
        size=window_size, stride=window_stride
    )
    return start_windows, end_windows


def _domain_outputs(model, raw_video, domain, device):
    video = move_video(raw_video, device)
    inputs, masks = model.preprocessing([video], mode="eval")
    if domain == "source":
        backbone, backbone_masks, _ = model.backbone_source(inputs, masks)
        fpn, fpn_masks = model.neck_source(backbone, backbone_masks)
        logits, _ = model.cls_head(fpn, fpn_masks)
        offsets, _ = model.reg_head(fpn, fpn_masks)
    else:
        backbone, backbone_masks, _ = model.backbone_target(inputs, masks)
        fpn, fpn_masks = model.neck_target(backbone, backbone_masks)
        logits, _ = model.cls_head_target(fpn, fpn_masks)
        offsets, _ = model.reg_head_target(fpn, fpn_masks)
    points = model.point_generator(fpn)
    if model.use_trident_head:
        starts, _ = model.start_head(fpn, fpn_masks)
        ends, _ = model.end_head(fpn, fpn_masks)
    else:
        starts = [None] * len(fpn)
        ends = [None] * len(fpn)
    return video, fpn, fpn_masks, logits, offsets, starts, ends, points


def _level_candidates(
    model,
    level,
    fpn,
    mask,
    logits,
    raw_offsets,
    starts,
    ends,
    points,
    topk,
    pool_samples,
    with_targets,
    gt_segments,
    gt_labels,
):
    probabilities = logits[0].transpose(0, 1).sigmoid()
    probabilities = probabilities * mask[0, 0].unsqueeze(-1)
    flat = probabilities.flatten()
    eligible = flat > model.test_pre_nms_thresh
    flat = flat[eligible]
    flat_indices = eligible.nonzero(as_tuple=True)[0]
    count = min(int(topk), len(flat_indices))
    if count == 0:
        return None
    scores, order = flat.sort(descending=True)
    scores = scores[:count]
    flat_indices = flat_indices[order[:count]]
    point_indices = torch.div(
        flat_indices, model.num_class_orig, rounding_mode="floor"
    )
    class_indices = torch.fmod(flat_indices, model.num_class_orig)

    offsets = raw_offsets[0].transpose(0, 1)
    if model.use_trident_head:
        start_windows, end_windows = _trident_windows(
            model, starts[0], ends[0]
        )
        decoded = model.decode_offset(
            offsets, start_windows, end_windows, mode="test"
        )
        selected_offsets = decoded[class_indices, point_indices]
        boundary = boundary_peak_quality(
            offsets, start_windows, end_windows
        )[class_indices, point_indices]
    else:
        selected_offsets = offsets[point_indices]
        boundary = torch.ones_like(scores)
    selected_points = points[point_indices, 0]
    stride = points[0, 3]
    segments = torch.stack(
        (
            selected_points - selected_offsets[:, 0] * stride,
            selected_points + selected_offsets[:, 1] * stride,
        ),
        dim=-1,
    )
    duration_mask = (
        segments[:, 1] - segments[:, 0]
    ) > model.test_duration_thresh
    scores = scores[duration_mask]
    point_indices = point_indices[duration_mask]
    class_indices = class_indices[duration_mask]
    selected_points = selected_points[duration_mask]
    segments = segments[duration_mask]
    boundary = boundary[duration_mask]
    if not len(scores):
        return None

    selected_probabilities = probabilities[point_indices]
    masked_other = selected_probabilities.clone()
    masked_other[torch.arange(len(class_indices), device=scores.device), class_indices] = -1
    margins = scores - masked_other.amax(dim=-1)
    pooled = proposal_region_context(
        fpn[0], segments, float(stride), samples=pool_samples
    )
    level_indices = torch.full_like(class_indices, level)
    scalars = proposal_scalar_features(
        scores,
        margins,
        segments,
        selected_points,
        boundary,
        class_indices,
        level_indices,
        model.num_class_orig,
        len(model.fpn_strides),
    )
    result = {
        "features": torch.cat((pooled, scalars), dim=-1),
        "segments": segments,
        "scores": scores,
        "labels": class_indices,
        "boundary_quality": boundary,
    }
    if with_targets:
        result["targets"] = same_class_temporal_iou(
            segments, class_indices, gt_segments, gt_labels
        )
    return result


@torch.inference_mode()
def extract_video_candidates(
    model,
    raw_video,
    domain,
    device,
    topk,
    pool_samples,
    with_targets=False,
):
    (
        video,
        fpn,
        fpn_masks,
        logits,
        offsets,
        starts,
        ends,
        points,
    ) = _domain_outputs(model, raw_video, domain, device)
    rows = []
    for level, values in enumerate(
        zip(fpn, fpn_masks, logits, offsets, starts, ends, points)
    ):
        row = _level_candidates(
            model,
            level,
            *values,
            topk,
            pool_samples,
            with_targets,
            video.get("segments"),
            video.get("labels"),
        )
        if row is not None:
            rows.append(row)
    keys = ("features", "segments", "scores", "labels", "boundary_quality")
    result = {key: torch.cat([row[key] for row in rows]) for key in keys}
    if with_targets:
        result["targets"] = torch.cat([row["targets"] for row in rows])
    return result


def collect_training_candidates(model, dataset, args):
    features = []
    targets = []
    video_count = min(len(dataset), args.max_train_videos or len(dataset))
    for index in range(video_count):
        candidates = extract_video_candidates(
            model,
            dataset[index],
            "source",
            args.device,
            args.train_topk_per_level,
            args.pool_samples,
            with_targets=True,
        )
        features.append(candidates["features"].cpu().half())
        targets.append(candidates["targets"].cpu().float())
        if (index + 1) % 10 == 0 or index + 1 == video_count:
            print(
                "training extraction {}/{}: {} candidates".format(
                    index + 1, video_count, sum(len(item) for item in targets)
                ),
                flush=True,
            )
    return torch.cat(features), torch.cat(targets)


def fit_quality_heads(features, targets, args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    head = ProposalIoUQualityHead(
        features.shape[-1], args.hidden_dim, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    snapshots = {}
    snapshot_epochs = set(args.snapshot_epochs)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(features))
        total_loss = 0.0
        total_count = 0
        head.train()
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_features = features[indices].to(args.device, dtype=torch.float32)
            batch_targets = targets[indices].to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = soft_iou_quality_loss(head(batch_features), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(indices)
            total_count += len(indices)
        print(
            "quality epoch {:02d}: loss {:.6f}".format(
                epoch, total_loss / max(total_count, 1)
            ),
            flush=True,
        )
        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(head).cpu().eval()
    if args.epochs not in snapshots:
        snapshots[args.epochs] = copy.deepcopy(head).cpu().eval()
    return snapshots


def evaluator_for(dataset, cfg):
    attributes = dataset.get_attributes()
    return ANETdetection(
        dataset.json_file,
        dataset.split,
        tiou_thresholds=attributes["tiou_thresholds"],
        action_subsampling=cfg["dataset"].get("action_subsampling", "all"),
    )


def _empty_predictions():
    return {key: [] for key in ("video-id", "t-start", "t-end", "label", "score")}


def _append_predictions(storage, result):
    count = len(result["segments"])
    storage["video-id"].extend([result["video_id"]] * count)
    storage["t-start"].append(result["segments"][:, 0])
    storage["t-end"].append(result["segments"][:, 1])
    storage["label"].append(result["labels"])
    storage["score"].append(result["scores"])


def _finalize_predictions(storage):
    return {
        "video-id": storage["video-id"],
        "t-start": torch.cat(storage["t-start"]).numpy(),
        "t-end": torch.cat(storage["t-end"]).numpy(),
        "label": torch.cat(storage["label"]).numpy(),
        "score": torch.cat(storage["score"]).numpy(),
    }


@torch.inference_mode()
def evaluate_heads(model, heads, dataset, domain, cfg, args, topk):
    variants = [(0, 0.0)] + [
        (epoch, gamma) for epoch in sorted(heads) for gamma in args.gammas
    ]
    predictions = {variant: _empty_predictions() for variant in variants}
    device_heads = {
        epoch: copy.deepcopy(head).to(args.device).eval() for epoch, head in heads.items()
    }
    video_count = min(len(dataset), args.max_eval_videos or len(dataset))
    for index in range(video_count):
        raw_video = dataset[index]
        candidates = extract_video_candidates(
            model,
            raw_video,
            domain,
            args.device,
            topk,
            args.pool_samples,
            with_targets=False,
        )
        qualities = {
            epoch: head(candidates["features"].float()).sigmoid()
            for epoch, head in device_heads.items()
        }
        metadata = raw_video["metadata"]
        for epoch, gamma in variants:
            quality = (
                torch.ones_like(candidates["scores"])
                if epoch == 0
                else qualities[epoch]
            )
            raw_result = {
                "video_id": metadata["video_id"],
                "fps": metadata["fps"],
                "duration": metadata["duration"],
                "video_length": metadata["video_length"],
                "feat_stride": metadata["feat_stride"],
                "feat_num_frames": metadata["feat_num_frames"],
                "use_temporal_grid_feats": metadata["use_temporal_grid_feats"],
                "segments": candidates["segments"],
                "scores": quality_weighted_scores(
                    candidates["scores"], quality, gamma
                ),
                "labels": candidates["labels"],
                "qualities": quality,
            }
            result = model.postprocessing([raw_result])[0]
            _append_predictions(predictions[(epoch, gamma)], result)
        if (index + 1) % 10 == 0 or index + 1 == video_count:
            print(
                "{} evaluation {}/{}".format(domain, index + 1, video_count),
                flush=True,
            )

    evaluator = evaluator_for(dataset, cfg)
    reports = {}
    for variant in variants:
        metrics = evaluator.evaluate(
            _finalize_predictions(predictions[variant]), verbose=False
        )
        reports[variant] = {
            "mAP_by_tiou": [float(value) for value in metrics[0]],
            "average_mAP": float(metrics[1]),
        }
    return reports


def report_to_json(reports):
    return {
        "epoch{}_gamma{}".format(epoch, gamma): values
        for (epoch, gamma), values in reports.items()
    }


def main():
    args = parse_args()
    cfg = load_config(args.config)
    eval_topk = args.eval_topk_per_level or int(cfg["test_cfg"]["pre_nms_topk"])
    model, checkpoint = load_detector(
        cfg, args.checkpoint, args.weights, args.device
    )
    source_train = make_dataset(
        cfg["dataset_name"], False, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    train_features, train_targets = collect_training_candidates(
        model, source_train, args
    )
    print(
        json.dumps(
            {
                "train_candidates": len(train_targets),
                "positive_iou_fraction": float((train_targets > 0).float().mean()),
                "mean_iou": float(train_targets.mean()),
                "input_dim": train_features.shape[-1],
            },
            indent=2,
        ),
        flush=True,
    )
    heads = fit_quality_heads(train_features, train_targets, args)
    source_reports = evaluate_heads(
        model, heads, source_val, "source", cfg, args, eval_topk
    )
    baseline = source_reports[(0, 0.0)]
    best_variant = max(
        (variant for variant in source_reports if variant != (0, 0.0)),
        key=lambda variant: (
            source_reports[variant]["average_mAP"],
            source_reports[variant]["mAP_by_tiou"][-1],
            -variant[1],
            -variant[0],
        ),
    )
    best = source_reports[best_variant]
    average_gain = best["average_mAP"] - baseline["average_mAP"]
    high_tiou_gain = best["mAP_by_tiou"][-1] - baseline["mAP_by_tiou"][-1]
    source_gate_pass = (
        average_gain >= args.min_source_map_gain
        and high_tiou_gain >= args.min_source_high_tiou_gain
    )
    target_reports = None
    if source_gate_pass:
        target_val = make_dataset(
            cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
        )
        selected_heads = {best_variant[0]: heads[best_variant[0]]}
        selected_args = copy.copy(args)
        selected_args.gammas = (best_variant[1],)
        target_reports = evaluate_heads(
            model,
            selected_heads,
            target_val,
            "target",
            cfg,
            selected_args,
            eval_topk,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "input_dim": int(train_features.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epoch": best_variant[0],
            "gamma": best_variant[1],
            "state_dict": heads[best_variant[0]].state_dict(),
            "source_gate_pass": source_gate_pass,
            "checkpoint_sha256": file_sha256(args.checkpoint),
        },
        output,
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "target_gt_usage_for_selection": "forbidden",
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "weights": args.weights,
        "train_candidates": len(train_targets),
        "train_mean_iou": float(train_targets.mean()),
        "selected_epoch": best_variant[0],
        "selected_gamma": best_variant[1],
        "source_baseline": baseline,
        "source_selected": best,
        "source_average_mAP_gain": average_gain,
        "source_high_tiou_mAP_gain": high_tiou_gain,
        "source_gate_pass": source_gate_pass,
        "source_gate_thresholds": {
            "average_mAP_gain": args.min_source_map_gain,
            "high_tiou_mAP_gain": args.min_source_high_tiou_gain,
        },
        "source_variants": report_to_json(source_reports),
        "target_variants": (
            report_to_json(target_reports) if target_reports is not None else None
        ),
        "output_head": str(output),
    }
    report_path = Path(str(output) + ".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
