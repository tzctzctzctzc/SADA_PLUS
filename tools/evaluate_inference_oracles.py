"""Decompose SADA target-validation errors with inference-time Oracles.

This is a diagnostic only: every Oracle below reads target ground truth and must
never be reported as an unsupervised result.  Unlike the training Oracles, it
does not update model parameters, so each counterfactual changes one inference
factor while keeping the checkpoint fixed.
"""

import argparse
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PRED_COLUMNS = ["video-id", "t-start", "t-end", "label", "score"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_iou(segment, candidates):
    """Return temporal IoU between one segment and an N x 2 array."""
    if len(candidates) == 0:
        return np.empty(0, dtype=np.float64)
    intersection = np.maximum(
        0.0,
        np.minimum(float(segment[1]), candidates[:, 1])
        - np.maximum(float(segment[0]), candidates[:, 0]),
    )
    union = (
        float(segment[1])
        - float(segment[0])
        + candidates[:, 1]
        - candidates[:, 0]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def prediction_matches(predictions, ground_truth, class_aware=False):
    """Find the highest-IoU GT for every prediction in the same video."""
    best_gt = np.full(len(predictions), -1, dtype=np.int64)
    best_iou = np.zeros(len(predictions), dtype=np.float64)
    if predictions.empty or ground_truth.empty:
        return best_gt, best_iou

    gt_by_video = {
        video_id: group
        for video_id, group in ground_truth.groupby("video-id", sort=False)
    }
    for video_id, pred_group in predictions.groupby("video-id", sort=False):
        gt_group = gt_by_video.get(video_id)
        if gt_group is None or gt_group.empty:
            continue
        gt_segments = gt_group[["t-start", "t-end"]].to_numpy(dtype=np.float64)
        gt_labels = gt_group["label"].to_numpy(dtype=np.int64)
        for pred_index, row in pred_group.iterrows():
            ious = temporal_iou(
                (row["t-start"], row["t-end"]), gt_segments
            )
            if class_aware:
                ious = np.where(gt_labels == int(row["label"]), ious, -1.0)
            local_index = int(np.argmax(ious))
            if ious[local_index] >= 0.0:
                best_gt[pred_index] = int(gt_group.index[local_index])
                best_iou[pred_index] = float(ious[local_index])
    return best_gt, best_iou


def apply_match_oracle(
    predictions,
    ground_truth,
    *,
    class_aware,
    min_iou=0.1,
    replace_label=False,
    replace_boundary=False,
    drop_invalid=False,
    rerank=False,
):
    """Apply a transparent GT counterfactual to a prediction table."""
    result = predictions.copy().reset_index(drop=True)
    gt_index, match_iou = prediction_matches(
        result, ground_truth, class_aware=class_aware
    )
    valid = (gt_index >= 0) & (match_iou >= float(min_iou))
    matched_rows = gt_index[valid]
    if replace_label:
        result.loc[valid, "label"] = ground_truth.loc[
            matched_rows, "label"
        ].to_numpy(dtype=np.int64)
    if replace_boundary:
        result.loc[valid, ["t-start", "t-end"]] = ground_truth.loc[
            matched_rows, ["t-start", "t-end"]
        ].to_numpy(dtype=np.float64)
    if rerank:
        original = result["score"].to_numpy(dtype=np.float64)
        result["score"] = np.where(
            valid, match_iou + original * 1e-9, -1.0 + original * 1e-9
        )
    if drop_invalid:
        result = result.loc[valid].reset_index(drop=True)
    return result


def proposal_coverage(predictions, ground_truth, thresholds, class_aware=False):
    """Fraction of GT instances covered by at least one predicted segment."""
    hits = np.zeros(len(thresholds), dtype=np.int64)
    total = len(ground_truth)
    pred_by_video = {
        video_id: group
        for video_id, group in predictions.groupby("video-id", sort=False)
    }
    for _, gt in ground_truth.iterrows():
        pred_group = pred_by_video.get(gt["video-id"])
        if pred_group is None or pred_group.empty:
            continue
        if class_aware:
            pred_group = pred_group[
                pred_group["label"].to_numpy(dtype=np.int64)
                == int(gt["label"])
            ]
            if pred_group.empty:
                continue
        candidates = pred_group[["t-start", "t-end"]].to_numpy(
            dtype=np.float64
        )
        best = float(
            temporal_iou((gt["t-start"], gt["t-end"]), candidates).max()
        )
        hits += best >= np.asarray(thresholds)
    return {
        "total_gt": int(total),
        "recall": {
            "{:.1f}".format(threshold): float(hit / max(total, 1))
            for threshold, hit in zip(thresholds, hits)
        },
    }


def proposal_ceiling(predictions, ground_truth, min_iou, class_aware=False):
    """Emit one exact detection for every GT covered by the proposal set."""
    pred_by_video = {
        video_id: group
        for video_id, group in predictions.groupby("video-id", sort=False)
    }
    rows = []
    for _, gt in ground_truth.iterrows():
        pred_group = pred_by_video.get(gt["video-id"])
        if pred_group is None or pred_group.empty:
            continue
        if class_aware:
            pred_group = pred_group[
                pred_group["label"].to_numpy(dtype=np.int64)
                == int(gt["label"])
            ]
            if pred_group.empty:
                continue
        candidates = pred_group[["t-start", "t-end"]].to_numpy(
            dtype=np.float64
        )
        if temporal_iou((gt["t-start"], gt["t-end"]), candidates).max() < min_iou:
            continue
        rows.append(
            {
                "video-id": gt["video-id"],
                "t-start": float(gt["t-start"]),
                "t-end": float(gt["t-end"]),
                "label": int(gt["label"]),
                "score": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def results_frame(outputs, pre_nms=False):
    suffix = "_pre_nms" if pre_nms else ""
    rows = []
    for output in outputs:
        segments = output["segments" + suffix].detach().cpu().numpy()
        scores = output["scores" + suffix].detach().cpu().numpy()
        labels = output["labels" + suffix].detach().cpu().numpy()
        rows.extend(
            {
                "video-id": str(output["video_id"]),
                "t-start": float(segment[0]),
                "t-end": float(segment[1]),
                "label": int(label),
                "score": float(score),
            }
            for segment, score, label in zip(segments, scores, labels)
        )
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def move_video(video, device):
    moved = dict(video)
    moved["metadata"] = dict(video["metadata"])
    for key, value in moved.items():
        if isinstance(value, torch.Tensor) and key != "full_feats":
            moved[key] = value.to(device)
    return moved


def checkpoint_config(checkpoint, fallback_config):
    from libs.utils.utils import _update_config, read_yaml

    saved = checkpoint.get("config")
    if saved is None:
        if fallback_config is None:
            raise ValueError("checkpoint has no config; pass --config")
        cfg = read_yaml(fallback_config)
    elif hasattr(saved, "config"):
        cfg = copy.deepcopy(saved.config)
    elif isinstance(saved, dict):
        cfg = copy.deepcopy(saved)
    else:
        raise TypeError("unsupported checkpoint config type: {}".format(type(saved)))
    return _update_config(cfg)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_inference(cfg, checkpoint, device, weights, max_videos=None):
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch

    dataset = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
    )
    model = make_meta_arch(cfg["model_name"], cfg, device=device).to(device)
    state_key = "ema_state_dict" if weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    post_frames = []
    pre_frames = []
    total = len(dataset) if max_videos is None else min(len(dataset), max_videos)
    with torch.no_grad():
        for index in range(total):
            video = move_video(dataset[index], device)
            _, target_output = model(
                video_list_src=[video],
                video_list_tgt=[video],
                mode="test",
            )
            post_frames.append(results_frame(target_output, pre_nms=False))
            pre_frames.append(results_frame(target_output, pre_nms=True))
            print("inference {}/{}".format(index + 1, total), flush=True)
    return (
        pd.concat(post_frames, ignore_index=True),
        pd.concat(pre_frames, ignore_index=True),
    )


def metric_summary(evaluator, predictions):
    metrics = evaluator.evaluate(predictions, verbose=False)
    maps, average_map, class_map = metrics[:3]
    return {
        "average_mAP_percent": float(average_map * 100.0),
        "mAP_by_tIoU_percent": [float(value * 100.0) for value in maps],
        "class_mAP_percent": [float(value * 100.0) for value in class_map],
        "prediction_count": int(len(predictions)),
    }


def build_oracles(post_predictions, ground_truth, min_iou):
    """Build non-additive counterfactuals with explicit semantics."""
    return {
        "baseline": post_predictions,
        "class_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=False,
            min_iou=min_iou,
            replace_label=True,
        ),
        "boundary_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=True,
            min_iou=min_iou,
            replace_boundary=True,
        ),
        "ranking_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=True,
            min_iou=min_iou,
            rerank=True,
        ),
        "validity_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=True,
            min_iou=min_iou,
            drop_invalid=True,
        ),
        "class_validity_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=False,
            min_iou=min_iou,
            replace_label=True,
            drop_invalid=True,
        ),
        "class_validity_ranking_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=False,
            min_iou=min_iou,
            replace_label=True,
            drop_invalid=True,
            rerank=True,
        ),
        "class_boundary_oracle": apply_match_oracle(
            post_predictions,
            ground_truth,
            class_aware=False,
            min_iou=min_iou,
            replace_label=True,
            replace_boundary=True,
        ),
        "post_nms_proposal_ceiling": proposal_ceiling(
            post_predictions, ground_truth, min_iou, class_aware=False
        ),
        "post_nms_class_aware_ceiling": proposal_ceiling(
            post_predictions, ground_truth, min_iou, class_aware=True
        ),
        "gt_sanity": ground_truth.assign(score=1.0)[PRED_COLUMNS],
    }


def main():
    from libs.evaluation.evaluator_anet import (
        ANETdetection,
        load_gt_seg_from_json,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--min-match-iou", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=None)
    args = parser.parse_args()

    if not 0.0 <= args.min_match_iou <= 1.0:
        raise ValueError("--min-match-iou must be in [0, 1]")
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint_config(checkpoint, args.config)
    post_predictions, pre_predictions = run_inference(
        cfg,
        checkpoint,
        args.device,
        args.weights,
        max_videos=args.max_videos,
    )
    ground_truth = load_gt_seg_from_json(
        cfg["dataset"]["json_file"],
        split=cfg["tgt_val_split"],
        action_subsampling=cfg["dataset"].get("action_subsampling", "all"),
    ).reset_index(drop=True)
    if args.max_videos is not None:
        evaluated_videos = set(post_predictions["video-id"].unique())
        ground_truth = ground_truth[
            ground_truth["video-id"].isin(evaluated_videos)
        ].reset_index(drop=True)
    thresholds = np.linspace(0.1, 0.5, 5)
    evaluator = ANETdetection(
        cfg["dataset"]["json_file"],
        split=None if args.max_videos is not None else cfg["tgt_val_split"],
        tiou_thresholds=thresholds,
        num_workers=args.num_workers,
        action_subsampling=cfg["dataset"].get("action_subsampling", "all"),
    )
    if args.max_videos is not None:
        evaluator.ground_truth = ground_truth.copy()
        evaluator.activity_index = {
            value: index
            for index, value in enumerate(sorted(ground_truth["label"].unique()))
        }
        evaluator.idx_to_activity_index = {
            index: value for value, index in evaluator.activity_index.items()
        }
        evaluator.ground_truth["label"] = evaluator.ground_truth["label"].replace(
            evaluator.activity_index
        )

    oracle_frames = build_oracles(
        post_predictions, ground_truth, args.min_match_iou
    )
    oracle_frames["pre_nms_proposal_ceiling"] = proposal_ceiling(
        pre_predictions, ground_truth, args.min_match_iou, class_aware=False
    )
    oracle_frames["pre_nms_class_aware_ceiling"] = proposal_ceiling(
        pre_predictions, ground_truth, args.min_match_iou, class_aware=True
    )
    output = {
        "schema_version": "sada-inference-oracle-v1",
        "warning": "All non-baseline rows use target validation GT and are diagnostics, not UDA results.",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "target_split": cfg["tgt_val_split"],
        "min_match_iou": args.min_match_iou,
        "counts": {
            "videos": int(post_predictions["video-id"].nunique()),
            "ground_truth": int(len(ground_truth)),
            "post_nms_predictions": int(len(post_predictions)),
            "pre_nms_predictions": int(len(pre_predictions)),
        },
        "proposal_coverage": {
            "post_nms_class_agnostic": proposal_coverage(
                post_predictions, ground_truth, thresholds, class_aware=False
            ),
            "post_nms_class_aware": proposal_coverage(
                post_predictions, ground_truth, thresholds, class_aware=True
            ),
            "pre_nms_class_agnostic": proposal_coverage(
                pre_predictions, ground_truth, thresholds, class_aware=False
            ),
            "pre_nms_class_aware": proposal_coverage(
                pre_predictions, ground_truth, thresholds, class_aware=True
            ),
        },
        "oracles": {
            name: metric_summary(evaluator, frame)
            for name, frame in oracle_frames.items()
        },
    }
    baseline = output["oracles"]["baseline"]["average_mAP_percent"]
    for metrics in output["oracles"].values():
        metrics["delta_vs_baseline_percent"] = (
            metrics["average_mAP_percent"] - baseline
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
