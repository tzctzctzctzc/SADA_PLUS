"""Source-calibrated target temporal-evidence boundary search audit."""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.evaluation import ANETdetection
from libs.models.SADA import make_meta_arch
from libs.utils import batched_nms
from libs.utils.utils import _update_config, read_yaml
from tools.audit_stage2_causal import pairwise_iou
from tools.temporal_boundary_search import (
    aligned_segment_iou,
    boundary_evidence,
    refine_with_evidence,
)


def parse_numbers(value, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(config_path, checkpoint_path, device, levels):
    cfg = read_yaml(config_path)
    if cfg["dataset_name"] != "epic":
        raise ValueError("P0 audit currently supports the EPIC S1-S4 protocol")
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["stage2"] = {
        "enabled": True,
        "start_epoch": 5,
        "weight": 1.0,
        "levels": levels,
        "routing": "confidence",
    }
    cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": "off",
        "target_regression_levels": levels,
    }
    _update_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    model.load_state_dict(checkpoint["ema_state_dict"])
    model.eval()
    return cfg, checkpoint, model


def encode_batch(model, batch):
    inputs, masks = model.preprocessing(batch, mode="validation")
    features, feature_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(features, feature_masks)
    logits, _ = model.cls_head_target(fpn_features, fpn_masks)
    offsets, _ = model.reg_head_target(fpn_features, fpn_masks)
    points = model.point_generator(fpn_features)
    return fpn_features, fpn_masks, logits, offsets, points


def coarse_proposals(model, batch, fpn_masks, logits, offsets, points, levels):
    targets = []
    for batch_index, video in enumerate(batch):
        segments, scores, labels = [], [], []
        for level in levels:
            probabilities = logits[level][batch_index].transpose(0, 1).sigmoid()
            level_scores, level_labels = probabilities.max(dim=-1)
            point = points[level]
            level_offsets = offsets[level][batch_index].transpose(0, 1)
            level_segments = torch.stack((
                point[:, 0] - level_offsets[:, 0] * point[:, 3],
                point[:, 0] + level_offsets[:, 1] * point[:, 3],
            ), dim=-1)
            valid_length = float(video["feats"].shape[-1])
            level_segments.clamp_(min=0.0, max=valid_length)
            usable = (
                fpn_masks[level][batch_index, 0]
                & torch.isfinite(level_segments).all(dim=-1)
                & ((level_segments[:, 1] - level_segments[:, 0])
                   > model.test_duration_thresh)
                & (level_scores >= model.stage2_alpha)
            )
            selected = torch.where(usable)[0]
            if selected.numel():
                segments.append(level_segments[selected])
                scores.append(level_scores[selected])
                labels.append(level_labels[selected])
        if not segments:
            targets.append({
                "segments": torch.empty((0, 2), device=model.device),
                "scores": torch.empty(0, device=model.device),
                "labels": torch.empty(0, dtype=torch.long, device=model.device),
            })
            continue
        all_segments = torch.cat(segments).detach().cpu()
        all_scores = torch.cat(scores).detach().cpu()
        all_labels = torch.cat(labels).detach().cpu()
        kept_segments, kept_scores, kept_labels = batched_nms(
            all_segments,
            all_scores,
            all_labels,
            model.test_iou_threshold,
            model.test_min_score,
            model.test_max_seg_num,
            use_soft_nms=False,
            multiclass=False,
            sigma=model.test_nms_sigma,
            voting_thresh=0.0,
        )
        keep = kept_scores >= model.stage2_alpha
        targets.append({
            "segments": kept_segments[keep].to(model.device),
            "scores": kept_scores[keep].to(model.device),
            "labels": kept_labels[keep].to(model.device),
        })
    return targets


def build_source_prototypes(model, dataset, device, batch_size, outside_ratio):
    class_count = model.num_class_orig
    feature_dim = model.cfg["model"]["fpn_dim"]
    inside_sum = torch.zeros(class_count, feature_dim, device=device)
    outside_sum = torch.zeros_like(inside_sum)
    inside_count = torch.zeros(class_count, device=device)
    outside_count = torch.zeros(class_count, device=device)

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_batch]
            fpn_features, fpn_masks, _, _, points = encode_batch(model, batch)
            point = points[0][:, 0]
            for index, video in enumerate(batch):
                valid = fpn_masks[0][index, 0]
                features = fpn_features[0][index].transpose(0, 1)
                segments = video["segments"]
                labels = video["labels"]
                action = torch.zeros_like(valid)
                for segment in segments:
                    action |= valid & (point >= segment[0]) & (point <= segment[1])
                for segment, label in zip(segments, labels):
                    label = int(label)
                    inside = valid & (point >= segment[0]) & (point <= segment[1])
                    width = max(float(segment[1] - segment[0]) * outside_ratio, 1.0)
                    outside = valid & ~action & (
                        ((point >= segment[0] - width) & (point < segment[0]))
                        | ((point > segment[1]) & (point <= segment[1] + width))
                    )
                    if inside.any():
                        inside_sum[label] += features[inside].sum(dim=0)
                        inside_count[label] += int(inside.sum())
                    if outside.any():
                        outside_sum[label] += features[outside].sum(dim=0)
                        outside_count[label] += int(outside.sum())

    global_inside = inside_sum.sum(dim=0) / inside_count.sum().clamp_min(1.0)
    global_outside = outside_sum.sum(dim=0) / outside_count.sum().clamp_min(1.0)
    inside = torch.stack([
        inside_sum[index] / inside_count[index]
        if inside_count[index] > 0 else global_inside
        for index in range(class_count)
    ])
    outside = torch.stack([
        outside_sum[index] / outside_count[index]
        if outside_count[index] > 0 else global_outside
        for index in range(class_count)
    ])
    return {
        "inside": F.normalize(inside, dim=-1).cpu(),
        "outside": F.normalize(outside, dim=-1).cpu(),
        "inside_count": inside_count.cpu(),
        "outside_count": outside_count.cpu(),
        "outside_ratio": outside_ratio,
    }


def move_video(video, device):
    result = {}
    for key, value in video.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def collect_records(
        model, dataset, device, batch_size, levels, prototypes,
        structural_parameters):
    records = []
    inside = prototypes["inside"].to(device)
    outside = prototypes["outside"].to(device)
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            batch = [move_video(video, device) for video in raw_batch]
            fpn_features, fpn_masks, logits, offsets, points = encode_batch(
                model, batch
            )
            proposals = coarse_proposals(
                model, batch, fpn_masks, logits, offsets, points, levels
            )
            point = points[0][:, 0]
            for index, (raw_video, video, proposal) in enumerate(zip(
                    raw_batch, batch, proposals)):
                valid = fpn_masks[0][index, 0]
                valid_points = point[valid]
                features = fpn_features[0][index].transpose(0, 1)[valid]
                actionness = logits[0][index].transpose(0, 1)[valid].sigmoid().max(
                    dim=-1
                ).values
                metadata = raw_video["metadata"]
                for segment, score, label in zip(
                        proposal["segments"], proposal["scores"], proposal["labels"]):
                    label_index = int(label)
                    evidence = {}
                    duration = float(segment[1] - segment[0])
                    for radius, window in structural_parameters:
                        key = "{:.6f}:{:.6f}".format(radius, window)
                        evidence[key] = {
                            "left": boundary_evidence(
                                valid_points, features, actionness,
                                float(segment[0]), duration, radius, window,
                                inside[label_index], outside[label_index], "left",
                            ),
                            "right": boundary_evidence(
                                valid_points, features, actionness,
                                float(segment[1]), duration, radius, window,
                                inside[label_index], outside[label_index], "right",
                            ),
                        }
                    records.append({
                        "video_id": metadata["video_id"],
                        "metadata": metadata,
                        "segment": segment.detach().cpu(),
                        "score": float(score),
                        "label": label_index,
                        "gt_segments": video["segments"].detach().cpu(),
                        "gt_labels": video["labels"].detach().cpu(),
                        "evidence": {
                            key: {
                                side: {name: value.cpu() for name, value in values.items()}
                                for side, values in sides.items()
                            }
                            for key, sides in evidence.items()
                        },
                    })
    return records


def refined_records(records, parameters):
    key = "{:.6f}:{:.6f}".format(
        parameters["search_radius_ratio"], parameters["window_ratio"]
    )
    refined, details = [], []
    for record in records:
        result, detail = refine_with_evidence(
            record["segment"],
            record["evidence"][key]["left"],
            record["evidence"][key]["right"],
            parameters,
        )
        refined.append(result)
        details.append(detail)
    return refined, details


def proposal_summary(records, refined, details):
    baseline_iou, refined_iou = [], []
    left_before, left_after, right_before, right_after = [], [], [], []
    class_correct = []
    for record, new_segment in zip(records, refined):
        gt = record["gt_segments"]
        if not gt.numel():
            baseline_iou.append(0.0)
            refined_iou.append(0.0)
            class_correct.append(False)
            continue
        overlap = pairwise_iou(record["segment"].unsqueeze(0), gt)[0]
        match = int(overlap.argmax())
        target = gt[match]
        baseline_iou.append(float(overlap[match]))
        refined_iou.append(float(aligned_segment_iou(new_segment, target)))
        duration = float(target[1] - target[0])
        if duration > 0:
            left_before.append(abs(float(record["segment"][0] - target[0])) / duration)
            left_after.append(abs(float(new_segment[0] - target[0])) / duration)
            right_before.append(abs(float(record["segment"][1] - target[1])) / duration)
            right_after.append(abs(float(new_segment[1] - target[1])) / duration)
        class_correct.append(record["label"] == int(record["gt_labels"][match]))

    baseline = np.asarray(baseline_iou)
    current = np.asarray(refined_iou)
    delta = current - baseline
    high_quality = baseline >= 0.7

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    return {
        "proposals": len(records),
        "mean_iou": {"baseline": mean(baseline), "refined": mean(current),
                     "delta": mean(delta)},
        "win_rate": mean(delta > 1e-4),
        "loss_rate": mean(delta < -1e-4),
        "tie_rate": mean(np.abs(delta) <= 1e-4),
        "iou50_rate": {"baseline": mean(baseline >= 0.5),
                       "refined": mean(current >= 0.5)},
        "iou70_rate": {"baseline": mean(baseline >= 0.7),
                       "refined": mean(current >= 0.7)},
        "high_quality_iou70": {
            "n": int(high_quality.sum()),
            "mean_delta_iou": mean(delta[high_quality]),
            "loss_rate": mean(delta[high_quality] < -1e-4),
        },
        "class_correct_rate": mean(class_correct),
        "normalized_left_error": {
            "baseline": mean(left_before), "refined": mean(left_after),
            "delta": mean(np.asarray(left_after) - np.asarray(left_before)),
        },
        "normalized_right_error": {
            "baseline": mean(right_before), "refined": mean(right_after),
            "delta": mean(np.asarray(right_after) - np.asarray(right_before)),
        },
        "left_applied_rate": mean([item["left"]["applied"] for item in details]),
        "right_applied_rate": mean([item["right"]["applied"] for item in details]),
    }


def prediction_dict(records, segments):
    output = {key: [] for key in ("video-id", "t-start", "t-end", "label", "score")}
    for record, segment in zip(records, segments):
        metadata = record["metadata"]
        converted = (
            segment.numpy() * float(metadata["feat_stride"])
            + 0.5 * float(metadata["feat_num_frames"])
        ) / float(metadata["fps"])
        output["video-id"].append(record["video_id"])
        output["t-start"].append(converted[0])
        output["t-end"].append(converted[1])
        output["label"].append(record["label"])
        output["score"].append(record["score"])
    return {
        "video-id": np.asarray(output["video-id"]),
        "t-start": np.asarray(output["t-start"]),
        "t-end": np.asarray(output["t-end"]),
        "label": np.asarray(output["label"]),
        "score": np.asarray(output["score"]),
    }


def detection_metrics(cfg, dataset, records, segments):
    evaluator = ANETdetection(
        dataset.json_file,
        dataset.split,
        tiou_thresholds=dataset.get_attributes()["tiou_thresholds"],
        action_subsampling=cfg["dataset"].get("action_subsampling", "all"),
    )
    result = evaluator.evaluate(prediction_dict(records, segments), verbose=False)
    return {
        "tiou_thresholds": evaluator.tiou_thresholds.tolist(),
        "map_by_tiou": result[0].tolist(),
        "average_map": float(result[1]),
    }


def parameter_grid(args):
    keys = (
        "search_radius_ratio", "window_ratio", "feature_weight",
        "semantic_weight", "distance_weight", "margin_threshold",
        "max_shift_ratio",
    )
    values = (
        parse_numbers(args.search_radius_ratios),
        parse_numbers(args.window_ratios),
        parse_numbers(args.feature_weights),
        parse_numbers(args.semantic_weights),
        parse_numbers(args.distance_weights),
        parse_numbers(args.margin_thresholds),
        parse_numbers(args.max_shift_ratios),
    )
    return [dict(zip(keys, combination)) for combination in itertools.product(*values)]


def source_select(args):
    levels = parse_numbers(args.levels, int)
    cfg, checkpoint, model = load_model(
        args.config, args.checkpoint, args.device, levels
    )
    source_train = make_dataset(
        cfg["dataset_name"], False, cfg["src_train_split"], **cfg["dataset"]
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    prototypes = build_source_prototypes(
        model, source_train, args.device, batch_size, args.prototype_outside_ratio
    )
    candidates = parameter_grid(args)
    structural = sorted({
        (item["search_radius_ratio"], item["window_ratio"])
        for item in candidates
    })
    records = collect_records(
        model, source_val, args.device, batch_size, levels, prototypes, structural
    )
    baseline_segments = [record["segment"] for record in records]
    baseline_detection = detection_metrics(
        cfg, source_val, records, baseline_segments
    )
    screened = []
    for parameters in candidates:
        refined, details = refined_records(records, parameters)
        summary = proposal_summary(records, refined, details)
        protected = (
            summary["iou70_rate"]["refined"]
            >= summary["iou70_rate"]["baseline"] - 1e-12
            and summary["high_quality_iou70"]["mean_delta_iou"] >= -0.002
        )
        screened.append({
            "parameters": parameters,
            "proposal": summary,
            "passes_high_quality_protection": protected,
        })
    eligible = [item for item in screened if item["passes_high_quality_protection"]]
    if not eligible:
        eligible = screened
    eligible.sort(
        key=lambda item: item["proposal"]["mean_iou"]["delta"], reverse=True
    )
    finalists = eligible[:args.map_finalists]
    for finalist in finalists:
        refined, _ = refined_records(records, finalist["parameters"])
        finalist["detection"] = detection_metrics(cfg, source_val, records, refined)
        finalist["detection"]["delta_average_map"] = (
            finalist["detection"]["average_map"]
            - baseline_detection["average_map"]
        )
    selected = max(finalists, key=lambda item: (
        item["detection"]["average_map"],
        item["proposal"]["mean_iou"]["delta"],
    ))
    artifact = {
        "version": 1,
        "parameters": selected["parameters"],
        "levels": levels,
        "prototypes": prototypes,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection": {
            "screening_metric": "source_val_mean_delta_iou",
            "protection": "iou70 non-decrease and high-quality mean delta >= -0.002",
            "final_metric": "source_val_fixed_proposal_average_map",
            "map_finalists": args.map_finalists,
        },
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_calibrated_temporal_evidence_boundary_search",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_train_videos": len(source_train),
        "source_val_videos": len(source_val),
        "source_val_proposals": len(records),
        "candidate_count": len(candidates),
        "protected_candidate_count": sum(
            item["passes_high_quality_protection"] for item in screened
        ),
        "baseline_detection": baseline_detection,
        "finalists": finalists,
        "selected": selected,
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location="cpu")
    levels = artifact["levels"]
    cfg, checkpoint, model = load_model(
        args.config, args.checkpoint, args.device, levels
    )
    target_val = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
    )
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    parameters = artifact["parameters"]
    structural = [(
        parameters["search_radius_ratio"], parameters["window_ratio"]
    )]
    records = collect_records(
        model, target_val, args.device, batch_size, levels,
        artifact["prototypes"], structural,
    )
    baseline = [record["segment"] for record in records]
    refined, details = refined_records(records, parameters)
    baseline_detection = detection_metrics(cfg, target_val, records, baseline)
    refined_detection = detection_metrics(cfg, target_val, records, refined)
    return {
        "audit": "locked_temporal_evidence_target_feasibility",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "locked_calibration": {
            key: value for key, value in artifact.items()
            if key != "prototypes"
        },
        "target_val_videos": len(target_val),
        "proposal": proposal_summary(records, refined, details),
        "detection": {
            "baseline": baseline_detection,
            "refined": refined_detection,
            "delta_average_map": (
                refined_detection["average_map"]
                - baseline_detection["average_map"]
            ),
        },
        "protocol": {
            "proposal_set": "fixed post-hard-class-agnostic-NMS proposals",
            "classification_labels_scores": "unchanged",
            "second_nms": "none",
            "actionness": "max sigmoid of trained dense classification logits",
            "target_gt_usage": "metrics only after locked refinement",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("source_select", "target_eval"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output")
    parser.add_argument("--calibration")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--prototype-outside-ratio", type=float, default=0.25)
    parser.add_argument("--search-radius-ratios", default="0.1,0.2")
    parser.add_argument("--window-ratios", default="0.05,0.1")
    parser.add_argument("--feature-weights", default="0,0.5,1")
    parser.add_argument("--semantic-weights", default="0,0.5,1")
    parser.add_argument("--distance-weights", default="0.1,0.25,0.5")
    parser.add_argument("--margin-thresholds", default="0,0.25")
    parser.add_argument("--max-shift-ratios", default="0.1,0.2")
    parser.add_argument("--map-finalists", type=int, default=10)
    args = parser.parse_args()
    if args.mode == "source_select" and not args.calibration_output:
        parser.error("--calibration-output is required for source_select")
    if args.mode == "target_eval" and not args.calibration:
        parser.error("--calibration is required for target_eval")
    seed_everything(args.seed)
    result = source_select(args) if args.mode == "source_select" else target_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
