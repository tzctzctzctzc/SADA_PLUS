"""Source-selected transfer audit for the BEGTR auxiliary boundary head."""

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.evaluation import ANETdetection
from libs.models.SADA import make_meta_arch
from libs.models.SADA.boundary_target_utils import (
    refine_segment_from_boundary_maps,
)
from libs.utils import batched_nms
from libs.utils.utils import _update_config, read_yaml


def parse_numbers(value, cast=float):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_video(video, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in video.items()
    }


def pairwise_iou(first, second):
    left = torch.maximum(first[:, None, 0], second[None, :, 0])
    right = torch.minimum(first[:, None, 1], second[None, :, 1])
    intersection = (right - left).clamp_min(0)
    first_length = (first[:, 1] - first[:, 0]).clamp_min(0)[:, None]
    second_length = (second[:, 1] - second[:, 0]).clamp_min(0)[None]
    return intersection / (first_length + second_length - intersection).clamp_min(1e-6)


def aligned_segment_iou(prediction, target):
    return pairwise_iou(prediction.unsqueeze(0), target.unsqueeze(0))[0, 0]


def load_model(config_path, checkpoint_path, device, model_kind):
    cfg = read_yaml(config_path)
    if cfg["dataset_name"] != "epic":
        raise ValueError("BEGTR audit currently supports EPIC S1-S4")
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": "off",
        "target_regression_levels": [0, 1, 2],
    }
    if cfg.get("begtr", {}).get("mode") != "source":
        raise ValueError("checkpoint config must enable begtr.mode=source")
    _update_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    state_key = "ema_state_dict" if model_kind == "ema" else "state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return cfg, checkpoint, model


def encode_batch(model, batch):
    inputs, masks = model.preprocessing(batch, mode="validation")
    features, feature_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(features, feature_masks)
    logits, _ = model.cls_head_target(fpn_features, fpn_masks)
    offsets, _ = model.reg_head_target(fpn_features, fpn_masks)
    starts, ends = model.aux_boundary_head(fpn_features, fpn_masks)
    points = model.point_generator(fpn_features)
    return fpn_masks, logits, offsets, starts, ends, points


def _match_nms_rows(post_segments, post_scores, post_labels,
                    pre_segments, pre_scores, pre_labels):
    used = torch.zeros(pre_scores.shape[0], dtype=torch.bool)
    indices = []
    for segment, score, label in zip(post_segments, post_scores, post_labels):
        match = (
            torch.isclose(pre_segments, segment[None], atol=1e-6, rtol=0).all(-1)
            & torch.isclose(pre_scores, score, atol=1e-6, rtol=0)
            & (pre_labels == label)
            & ~used
        )
        candidates = torch.where(match)[0]
        if not candidates.numel():
            raise RuntimeError("hard-NMS output could not be matched to its input")
        index = int(candidates[0])
        used[index] = True
        indices.append(index)
    return torch.tensor(indices, dtype=torch.long)


def coarse_proposals(model, batch, fpn_masks, logits, offsets, points, levels):
    outputs = []
    alpha = float(model.cfg["model"].get("pl_threshold", 0.7))
    for batch_index, video in enumerate(batch):
        segments, scores, labels, proposal_levels = [], [], [], []
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
                & (level_scores >= alpha)
            )
            selected = torch.where(usable)[0]
            if selected.numel():
                segments.append(level_segments[selected])
                scores.append(level_scores[selected])
                labels.append(level_labels[selected])
                proposal_levels.append(torch.full_like(selected, level))
        if not segments:
            outputs.append({
                "segments": torch.empty((0, 2), device=model.device),
                "scores": torch.empty(0, device=model.device),
                "labels": torch.empty(0, dtype=torch.long, device=model.device),
                "levels": torch.empty(0, dtype=torch.long, device=model.device),
            })
            continue
        pre_segments = torch.cat(segments).detach().cpu()
        pre_scores = torch.cat(scores).detach().cpu()
        pre_labels = torch.cat(labels).detach().cpu()
        pre_levels = torch.cat(proposal_levels).detach().cpu()
        kept_segments, kept_scores, kept_labels = batched_nms(
            pre_segments,
            pre_scores,
            pre_labels,
            model.test_iou_threshold,
            model.test_min_score,
            model.test_max_seg_num,
            use_soft_nms=False,
            multiclass=False,
            sigma=model.test_nms_sigma,
            voting_thresh=0.0,
        )
        keep = kept_scores >= alpha
        kept_segments = kept_segments[keep]
        kept_scores = kept_scores[keep]
        kept_labels = kept_labels[keep]
        indices = _match_nms_rows(
            kept_segments, kept_scores, kept_labels,
            pre_segments, pre_scores, pre_labels,
        )
        outputs.append({
            "segments": kept_segments.to(model.device),
            "scores": kept_scores.to(model.device),
            "labels": kept_labels.to(model.device),
            "levels": pre_levels[indices].to(model.device),
        })
    return outputs


def collect_records(model, dataset, device, batch_size, levels):
    records = []
    map_statistics = []
    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(batch_start, min(batch_start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            fpn_masks, logits, offsets, starts, ends, points = encode_batch(
                model, batch
            )
            proposals = coarse_proposals(
                model, batch, fpn_masks, logits, offsets, points, levels
            )
            for batch_index, (raw_video, video, proposal) in enumerate(zip(
                    raw_batch, batch, proposals)):
                per_level = {}
                for level in levels:
                    valid = fpn_masks[level][batch_index, 0]
                    per_level[level] = {
                        "points": points[level][:, 0][valid].detach().cpu(),
                        "start": starts[level][batch_index][valid].sigmoid().detach().cpu(),
                        "end": ends[level][batch_index][valid].sigmoid().detach().cpu(),
                    }
                    valid_values = torch.cat((
                        per_level[level]["start"], per_level[level]["end"]
                    ))
                    if valid_values.numel():
                        map_statistics.append((
                            float(valid_values.mean()), float(valid_values.std(unbiased=False))
                        ))
                metadata = raw_video["metadata"]
                for segment, score, label, level in zip(
                        proposal["segments"], proposal["scores"],
                        proposal["labels"], proposal["levels"]):
                    level = int(level)
                    records.append({
                        "video_id": metadata["video_id"],
                        "metadata": metadata,
                        "segment": segment.detach().cpu(),
                        "score": float(score),
                        "label": int(label),
                        "level": level,
                        "points": per_level[level]["points"],
                        "start": per_level[level]["start"],
                        "end": per_level[level]["end"],
                        "gt_segments": video["segments"].detach().cpu(),
                        "gt_labels": video["labels"].detach().cpu(),
                    })
    means = [item[0] for item in map_statistics]
    stds = [item[1] for item in map_statistics]
    return records, {
        "map_mean": float(np.mean(means)) if means else None,
        "mean_temporal_std": float(np.mean(stds)) if stds else None,
    }


def refine_records(records, parameters):
    segments, details = [], []
    for record in records:
        refined, detail = refine_segment_from_boundary_maps(
            record["points"], record["start"], record["end"],
            record["segment"], parameters,
        )
        segments.append(refined)
        details.append(detail)
    return segments, details


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
        "left_accepted_rate": mean([
            item["left"]["accepted"] for item in details
        ]),
        "right_accepted_rate": mean([
            item["right"]["accepted"] for item in details
        ]),
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
    return {key: np.asarray(value) for key, value in output.items()}


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
        "search_radius_ratio", "absolute_threshold", "gain_threshold",
        "margin_threshold", "max_shift_ratio",
    )
    values = (
        parse_numbers(args.search_radius_ratios),
        parse_numbers(args.absolute_thresholds),
        parse_numbers(args.gain_thresholds),
        parse_numbers(args.margin_thresholds),
        parse_numbers(args.max_shift_ratios),
    )
    return [
        {
            **dict(zip(keys, combination)),
            "min_duration_ratio": args.min_duration_ratio,
            "max_duration_ratio": args.max_duration_ratio,
        }
        for combination in itertools.product(*values)
    ]


def source_select(args):
    levels = parse_numbers(args.levels, int)
    cfg, checkpoint, model = load_model(
        args.config, args.checkpoint, args.device, args.model_kind
    )
    source_val = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    records, map_statistics = collect_records(
        model, source_val, args.device, batch_size, levels
    )
    baseline_segments = [record["segment"] for record in records]
    baseline_detection = detection_metrics(
        cfg, source_val, records, baseline_segments
    )
    screened = []
    for parameters in parameter_grid(args):
        refined, details = refine_records(records, parameters)
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
    ranked = eligible if eligible else screened
    ranked.sort(key=lambda item: item["proposal"]["mean_iou"]["delta"], reverse=True)
    finalists = ranked[:args.map_finalists]
    for finalist in finalists:
        refined, _ = refine_records(records, finalist["parameters"])
        finalist["detection"] = detection_metrics(cfg, source_val, records, refined)
        finalist["detection"]["delta_average_map"] = (
            finalist["detection"]["average_map"] - baseline_detection["average_map"]
        )
    selected = max(finalists, key=lambda item: (
        item["detection"]["average_map"],
        item["proposal"]["mean_iou"]["delta"],
    ))
    artifact = {
        "version": 1,
        "parameters": selected["parameters"],
        "levels": levels,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "model_kind": args.model_kind,
        "selection": {
            "source_only": True,
            "protection": "iou70 non-decrease and high-quality mean delta >= -0.002",
            "screening_metric": "source_val_mean_delta_iou",
            "final_metric": "source_val_fixed_proposal_average_map",
        },
    }
    torch.save(artifact, args.calibration_output)
    return {
        "audit": "source_selected_aux_boundary_transfer",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_val_videos": len(source_val),
        "source_val_proposals": len(records),
        "boundary_map_statistics": map_statistics,
        "candidate_count": len(screened),
        "protected_candidate_count": len(eligible),
        "baseline_detection": baseline_detection,
        "finalists": finalists,
        "selected": selected,
        "calibration_output": args.calibration_output,
        "target_gt_usage": "none",
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location="cpu")
    cfg, checkpoint, model = load_model(
        args.config, args.checkpoint, args.device, artifact["model_kind"]
    )
    target_val = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
    )
    batch_size = args.batch_size or int(cfg.get("loader", {}).get("batch_size", 1))
    records, map_statistics = collect_records(
        model, target_val, args.device, batch_size, artifact["levels"]
    )
    baseline = [record["segment"] for record in records]
    refined, details = refine_records(records, artifact["parameters"])
    baseline_detection = detection_metrics(cfg, target_val, records, baseline)
    refined_detection = detection_metrics(cfg, target_val, records, refined)
    return {
        "audit": "locked_aux_boundary_target_feasibility",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "locked_calibration": artifact,
        "target_val_videos": len(target_val),
        "boundary_map_statistics": map_statistics,
        "proposal": proposal_summary(records, refined, details),
        "detection": {
            "baseline": baseline_detection,
            "refined": refined_detection,
            "delta_average_map": (
                refined_detection["average_map"] - baseline_detection["average_map"]
            ),
        },
        "protocol": {
            "proposal_set": "fixed post-hard-class-agnostic-NMS proposals",
            "classification_labels_scores": "unchanged",
            "boundary_map": "same FPN level as each retained proposal",
            "second_nms": "none",
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
    parser.add_argument("--model-kind", choices=("ema", "student"), default="ema")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--search-radius-ratios", default="0.05,0.1,0.2")
    parser.add_argument("--absolute-thresholds", default="0.1,0.2,0.3,0.5")
    parser.add_argument("--gain-thresholds", default="0,0.05,0.1,0.2")
    parser.add_argument("--margin-thresholds", default="0,0.05,0.1,0.2")
    parser.add_argument("--max-shift-ratios", default="0.05,0.1,0.2")
    parser.add_argument("--min-duration-ratio", type=float, default=0.5)
    parser.add_argument("--max-duration-ratio", type=float, default=1.5)
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
