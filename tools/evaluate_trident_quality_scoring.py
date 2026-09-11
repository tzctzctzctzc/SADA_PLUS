"""Select Trident localization-quality scoring on source and report target once."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from libs.utils import batched_nms
from libs.models.SADA.trident_quality import quality_weighted_scores
from tools.evaluate_inference_oracles import (
    PRED_COLUMNS,
    checkpoint_config,
    load_checkpoint,
    move_video,
)


def raw_split_outputs(model, dataset, domain, device):
    outputs = []
    model.test_trident_quality_gamma = 0.0
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            video = move_video(dataset[index], device)
            source, target = model(
                video_list_src=[video], video_list_tgt=[video], mode="test"
            )
            outputs.append((source if domain == "source" else target)[0])
            print(
                "{} inference {}/{}".format(domain, index + 1, len(dataset)),
                flush=True,
            )
    return outputs


def apply_quality_nms(outputs, cfg, gamma):
    rows = []
    test_cfg = cfg["test_cfg"]
    for output in outputs:
        segments = output["segments_pre_nms"]
        labels = output["labels_pre_nms"]
        scores = quality_weighted_scores(
            output["scores_pre_nms"], output["qualities_pre_nms"], gamma
        )
        if test_cfg["nms_method"] == "none":
            kept_segments, kept_scores, kept_labels = segments, scores, labels
        else:
            kept_segments, kept_scores, kept_labels = batched_nms(
                segments,
                scores,
                labels,
                test_cfg["iou_threshold"],
                test_cfg["min_score"],
                test_cfg["max_seg_num"],
                use_soft_nms=test_cfg["nms_method"] == "soft",
                multiclass=test_cfg["multiclass_nms"],
                sigma=test_cfg["nms_sigma"],
                voting_thresh=test_cfg["voting_thresh"],
            )
        rows.extend({
            "video-id": str(output["video_id"]),
            "t-start": float(segment[0]),
            "t-end": float(segment[1]),
            "label": int(label),
            "score": float(score),
        } for segment, score, label in zip(
            kept_segments, kept_scores, kept_labels
        ))
    return pd.DataFrame(rows, columns=PRED_COLUMNS)


def evaluate(evaluator, predictions):
    maps, average_map, _ = evaluator.evaluate(predictions, verbose=False)[:3]
    return {
        "average_mAP_percent": float(average_map * 100.0),
        "mAP_by_tIoU_percent": [float(value * 100.0) for value in maps],
        "prediction_count": int(len(predictions)),
    }


def make_evaluator(dataset, cfg):
    from libs.evaluation.evaluator_anet import ANETdetection

    return ANETdetection(
        dataset.json_file,
        dataset.split,
        tiou_thresholds=np.linspace(0.1, 0.5, 5),
        num_workers=cfg["loader"].get("num_workers", 4),
        action_subsampling=cfg["dataset"].get("action_subsampling", "all"),
    )


def main():
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0, 2.0)
    )
    args = parser.parse_args()
    if any(value < 0.0 for value in args.gammas):
        raise ValueError("all gammas must be non-negative")

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    cfg = checkpoint_config(checkpoint, args.config)
    cfg = copy.deepcopy(cfg)
    cfg["test_cfg"]["trident_quality_gamma"] = 0.0
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    model.load_state_dict(checkpoint[state_key])

    source_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["src_val_split"], **cfg["dataset"]
    )
    target_dataset = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_val_split"], **cfg["dataset"]
    )
    source_outputs = raw_split_outputs(
        model, source_dataset, "source", args.device
    )
    source_evaluator = make_evaluator(source_dataset, cfg)
    source_results = {
        str(gamma): evaluate(
            source_evaluator, apply_quality_nms(source_outputs, cfg, gamma)
        ) for gamma in args.gammas
    }
    best_gamma = min(
        args.gammas,
        key=lambda gamma: (
            -source_results[str(gamma)]["average_mAP_percent"], gamma
        ),
    )

    target_outputs = raw_split_outputs(
        model, target_dataset, "target", args.device
    )
    target_evaluator = make_evaluator(target_dataset, cfg)
    target_results = {
        str(gamma): evaluate(
            target_evaluator, apply_quality_nms(target_outputs, cfg, gamma)
        ) for gamma in sorted(set((0.0, best_gamma)))
    }
    baseline = target_results["0.0"]["average_mAP_percent"]
    selected = target_results[str(best_gamma)]["average_mAP_percent"]
    report = {
        "protocol_version": "trident-boundary-peak-quality-source-select-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "weights": args.weights,
        "selection_split": list(cfg["src_val_split"]),
        "target_gt_used_for_selection": False,
        "gammas": list(args.gammas),
        "source_results": source_results,
        "selected_gamma": float(best_gamma),
        "target_results": target_results,
        "target_delta_percent": float(selected - baseline),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
