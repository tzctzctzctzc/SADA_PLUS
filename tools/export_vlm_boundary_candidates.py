"""Export frozen-SADA proposals with exact origin-anchor provenance.

Proposal selection is prediction-only. Ground truth is attached only after the
complete proposal set has been frozen and is used solely by offline scoring.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vlm_boundary_utils import (
    PROPOSAL_SCHEMA_VERSION,
    clipped_segment,
    file_sha256,
    grid_to_seconds,
    hard_class_agnostic_nms_indices,
    state_dict_sha256,
    temporal_iou,
    write_jsonl,
)


DEFAULT_CLASS_NAMES = {
    0: "TAKE",
    1: "PUT",
    2: "WASH",
    3: "OPEN",
    4: "CLOSE",
    5: "INSERT",
    6: "TURN-ON",
    7: "CUT",
    8: "TURN-OFF",
    9: "POUR",
}


def kept_label(label, action_subsampling):
    if action_subsampling == "10_majority_classes":
        return int(label) < 10
    if action_subsampling == "20_random_classes":
        return int(label) < 20
    return True


def prediction_only_video(video, device):
    """Strip annotation tensors before the frozen proposal model sees a video."""
    return {
        "feats": video["feats"].to(device),
        "metadata": dict(video["metadata"]),
    }


def parse_levels(value):
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or len(set(levels)) != len(levels) or any(level < 0 for level in levels):
        raise ValueError("levels must be unique non-negative integers")
    return levels


def load_class_names(path):
    if not path:
        return dict(DEFAULT_CLASS_NAMES)
    with open(path, encoding="utf-8") as stream:
        result = dict(DEFAULT_CLASS_NAMES)
        result.update(
            {int(row["id"]): row["key"] for row in csv.DictReader(stream)}
        )
        return result


def available_video_ids(root):
    if not root:
        return None
    return {
        path.stem
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }


def split_for(cfg, domain, split):
    key = "{}_{}_split".format("src" if domain == "source" else "tgt", split)
    return cfg[key]


def model_modules(model, domain):
    if domain == "source":
        return model.backbone_source, model.neck_source, model.cls_head, model.reg_head
    return (
        model.backbone_target,
        model.neck_target,
        model.cls_head_target,
        model.reg_head_target,
    )


def proposal_id(row):
    return "{}:l{}:x{}:a{}:c{}".format(
        row["video_id"],
        row["fpn_level"],
        row["absolute_feature_index"],
        row["origin_anchor_index"],
        row["predicted_class_id"],
    )


def attach_posthoc_gt(rows, database, action_subsampling):
    """Attach GT only after proposal selection has completely finished."""
    for row in rows:
        actions = [
            action
            for action in database[row["video_id"]].get("annotations", [])
            if kept_label(int(action["label_id"]), action_subsampling)
        ]
        coarse = row["coarse_segment_seconds"]
        overlaps = [temporal_iou(coarse, action["segment"]) for action in actions]
        same_class = [
            (index, overlap)
            for index, (action, overlap) in enumerate(zip(actions, overlaps))
            if int(action["label_id"]) == int(row["predicted_class_id"])
        ]
        same_index, same_iou = (
            max(same_class, key=lambda item: item[1]) if same_class else (None, 0.0)
        )
        any_index = max(range(len(actions)), key=overlaps.__getitem__) if actions else None
        any_iou = overlaps[any_index] if any_index is not None else 0.0

        if same_index is not None and same_iou > 0:
            match_index = same_index
            kind = "VALID_ACTION"
            match_iou = same_iou
        elif any_index is not None and any_iou > 0:
            match_index = any_index
            kind = "WRONG_ACTION"
            match_iou = any_iou
        else:
            match_index = None
            kind = "NO_ACTION"
            match_iou = 0.0

        matched = actions[match_index] if match_index is not None else None
        row["posthoc_gt"] = {
            "kind": kind,
            "matched_segment_seconds": (
                [float(value) for value in matched["segment"]] if matched else None
            ),
            "matched_class_id": int(matched["label_id"]) if matched else None,
            "matched_label": matched.get("label") if matched else None,
            "coarse_iou": float(match_iou),
            "same_class_max_iou": float(same_iou),
            "any_target_class_max_iou": float(any_iou),
        }


@torch.no_grad()
def export_video(model, video, domain, levels, alpha, nms_iou, max_proposals, class_names):
    backbone, neck, cls_head, reg_head = model_modules(model, domain)
    inputs, masks = model.preprocessing([video], mode="validation")
    feats, backbone_masks, _ = backbone(inputs, masks)
    fpn_feats, fpn_masks = neck(feats, backbone_masks)
    cls_logits, _ = cls_head(fpn_feats, fpn_masks)
    offsets, _ = reg_head(fpn_feats, fpn_masks)
    points = model.point_generator(fpn_feats)

    metadata = video["metadata"]
    video_id = metadata["video_id"]
    fps = float(metadata["fps"])
    feat_stride = float(metadata["feat_stride"])
    num_frames = float(metadata["feat_num_frames"])
    duration = float(metadata["duration"])
    crop_start = float(metadata.get("crop_start", 0.0))
    feature_length = float(video["feats"].shape[-1])

    candidates = []
    grid_segments = []
    candidate_scores = []
    for level in levels:
        if level >= len(points):
            raise ValueError("requested FPN level {} does not exist".format(level))
        point_grid = points[level]
        valid = fpn_masks[level][0, 0]
        probabilities = cls_logits[level][0].transpose(0, 1).sigmoid()
        scores, labels = probabilities.max(dim=-1)
        level_offsets = offsets[level][0].transpose(0, 1)
        segments = torch.stack(
            (
                point_grid[:, 0] - level_offsets[:, 0] * point_grid[:, 3],
                point_grid[:, 0] + level_offsets[:, 1] * point_grid[:, 3],
            ),
            dim=-1,
        )
        segments[:, 0].clamp_(0.0, feature_length)
        segments[:, 1].clamp_(0.0, feature_length)
        usable = (
            valid
            & torch.isfinite(segments).all(dim=-1)
            & ((segments[:, 1] - segments[:, 0]) > model.test_duration_thresh)
            & (scores >= float(alpha))
        )
        for anchor_index in torch.where(usable)[0].tolist():
            point = float(point_grid[anchor_index, 0])
            stride_grid = float(point_grid[anchor_index, 3])
            segment_grid = [float(value) for value in segments[anchor_index].tolist()]
            segment_seconds = clipped_segment(
                [
                    grid_to_seconds(
                        value + crop_start, feat_stride, num_frames, fps
                    )
                    for value in segment_grid
                ],
                duration,
            )
            if segment_seconds[1] <= segment_seconds[0]:
                continue
            row = {
                "proposal_schema_version": PROPOSAL_SCHEMA_VERSION,
                "video_id": video_id,
                "video_duration_seconds": duration,
                "predicted_class_id": int(labels[anchor_index]),
                "predicted_class_name": class_names.get(
                    int(labels[anchor_index]), "class_{}".format(int(labels[anchor_index]))
                ),
                "score": float(scores[anchor_index]),
                "coarse_segment_seconds": segment_seconds,
                "coarse_segment_grid": segment_grid,
                "fpn_level": int(level),
                "origin_anchor_index": int(anchor_index),
                "origin_point_grid": point,
                "origin_stride_grid": stride_grid,
                "origin_stride_seconds": stride_grid * feat_stride / fps,
                "absolute_feature_index": int(round(point + crop_start)),
                "absolute_center_time": grid_to_seconds(
                    point + crop_start, feat_stride, num_frames, fps
                ),
                "feat_stride_frames": feat_stride,
                "feature_num_frames": num_frames,
                "fps": fps,
            }
            row["proposal_id"] = proposal_id(row)
            candidates.append(row)
            grid_segments.append(segment_grid)
            candidate_scores.append(float(scores[anchor_index]))

    if not candidates:
        return []
    keep = hard_class_agnostic_nms_indices(
        torch.tensor(grid_segments, dtype=torch.float32),
        torch.tensor(candidate_scores, dtype=torch.float32),
        nms_iou,
        max_proposals,
    ).tolist()
    return [candidates[index] for index in keep]


def main():
    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.utils.utils import _update_config, read_yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("ema", "student"), default="ema")
    parser.add_argument("--domain", choices=("source", "target"), default="target")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--levels", default="0,1,2")
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--nms-iou", type=float)
    parser.add_argument("--max-proposals-per-video", type=int)
    parser.add_argument("--ontology-csv")
    parser.add_argument("--video-root")
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    levels = parse_levels(args.levels)
    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["diagnostic"] = {
        "oracle_routing": "off",
        "oracle_target_regression": "off",
    }
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"],
        False,
        split_for(cfg, args.domain, args.split),
        **cfg["dataset"]
    )
    video_ids = available_video_ids(args.video_root)
    class_names = load_class_names(args.ontology_csv)

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    alpha = float(
        cfg["model"].get("pl_threshold", 0.7) if args.alpha is None else args.alpha
    )
    nms_iou = float(model.test_iou_threshold if args.nms_iou is None else args.nms_iou)
    max_proposals = int(
        model.test_max_seg_num
        if args.max_proposals_per_video is None
        else args.max_proposals_per_video
    )
    if not 0 <= alpha <= 1 or not 0 <= nms_iou <= 1 or max_proposals <= 0:
        raise ValueError("invalid proposal selection parameters")

    rows = []
    skipped_missing_video = 0
    for index in range(len(dataset)):
        raw_video = dataset[index]
        video_id = raw_video["metadata"]["video_id"]
        if video_ids is not None and video_id not in video_ids:
            skipped_missing_video += 1
            continue
        rows.extend(
            export_video(
                model,
                prediction_only_video(raw_video, args.device),
                args.domain,
                levels,
                alpha,
                nms_iou,
                max_proposals,
                class_names,
            )
        )

    if len({row["proposal_id"] for row in rows}) != len(rows):
        raise RuntimeError("proposal IDs are not unique")

    database = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    attach_posthoc_gt(rows, database, cfg["dataset"].get("action_subsampling", "all"))
    write_jsonl(args.output, rows)

    metadata = {
        "proposal_schema_version": PROPOSAL_SCHEMA_VERSION,
        "branch_base": "main@d184127",
        "config": str(Path(args.config).resolve()),
        "config_sha256": file_sha256(args.config),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "teacher_state_key": state_key,
        "teacher_state_sha256": state_dict_sha256(checkpoint[state_key]),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "annotation_json": str(Path(cfg["dataset"]["json_file"]).resolve()),
        "annotation_json_sha256": file_sha256(cfg["dataset"]["json_file"]),
        "ontology": (
            str(Path(args.ontology_csv).resolve())
            if args.ontology_csv
            else "built_in_epic_10_majority"
        ),
        "ontology_sha256": (
            file_sha256(args.ontology_csv) if args.ontology_csv else None
        ),
        "video_root": str(Path(args.video_root).resolve()) if args.video_root else None,
        "domain": args.domain,
        "split": args.split,
        "levels": levels,
        "alpha": alpha,
        "nms": "hard_class_agnostic_exact_origin_v1",
        "nms_iou": nms_iou,
        "max_proposals_per_video": max_proposals,
        "exact_origin_key_fields": [
            "video_id",
            "fpn_level",
            "absolute_feature_index",
            "origin_anchor_index",
        ],
        "selection_gt_use": "none",
        "posthoc_gt_use": "attached only after complete proposal selection",
        "videos_in_dataset": len(dataset),
        "videos_skipped_without_mp4": skipped_missing_video,
        "proposals": len(rows),
        "posthoc_gt_counts": {
            kind: sum(row["posthoc_gt"]["kind"] == kind for row in rows)
            for kind in ("VALID_ACTION", "WRONG_ACTION", "NO_ACTION")
        },
        "seed": args.seed,
    }
    Path(str(args.output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
