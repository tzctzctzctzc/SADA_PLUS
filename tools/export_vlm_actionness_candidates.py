"""Export low-confidence target anchors for an offline VLM actionness audit.

Candidate selection only uses frozen SADA predictions. Target GT is attached
after selection and is consumed solely by the separate scoring script.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def video_ids_in(root):
    root = Path(root)
    return {
        path.stem
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }


def named_paths(values):
    result = {}
    for value in values or ():
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError("validity models must use NAME=PATH")
        if name in result:
            raise ValueError("duplicate validity model name: {}".format(name))
        result[name] = path
    return result


def choose_indices(
    scores,
    centers,
    valid,
    beta,
    alpha,
    top_k,
    min_gap_sec,
    score_band="low",
):
    if score_band == "low":
        in_band = lambda score: beta < score < alpha
    elif score_band == "high":
        in_band = lambda score: score >= alpha
    else:
        raise ValueError("score_band must be low or high")
    candidates = [
        index
        for index in range(scores.numel())
        if bool(valid[index]) and in_band(float(scores[index]))
    ]
    candidates.sort(key=lambda index: float(scores[index]), reverse=True)
    chosen = []
    for index in candidates:
        center = float(centers[index])
        if any(abs(center - float(centers[old])) < min_gap_sec for old in chosen):
            continue
        chosen.append(index)
        if len(chosen) == top_k:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--dataset-json")
    parser.add_argument("--feature-dir")
    parser.add_argument(
        "--subset",
        action="append",
        help="Override configured target split; may be repeated.",
    )
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--score-band",
        choices=("low", "high"),
        default="low",
        help="Export beta<score<alpha or score>=alpha anchors.",
    )
    parser.add_argument("--top-k-per-video-level", type=int, default=5)
    parser.add_argument("--min-gap-sec", type=float, default=0.75)
    parser.add_argument("--levels", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument(
        "--reuse-manifest",
        help="Reuse these exact candidate anchors and only refresh their logits/GT.",
    )
    parser.add_argument(
        "--validity-model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Score each exact anchor with a pre-calibrated source-only model.",
    )
    args = parser.parse_args()

    import torch

    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.anchor_validity import (
        anchor_validity_features,
        assignment_protocol_from_model,
        cohort_protocol_from_config,
        load_validity_model,
        probe_probabilities,
        validate_validity_provenance,
        validity_model_to,
    )
    from libs.models.SADA.stage1_calibration import state_dict_sha256
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import (
        crop_targets,
        kept_label,
        move_video,
    )

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    if args.dataset_json:
        cfg["dataset"]["json_file"] = args.dataset_json
    if args.feature_dir:
        cfg["dataset"]["feat_folder"] = args.feature_dir
    _update_config(cfg)

    reuse_rows = None
    reuse_by_key = {}
    if args.reuse_manifest:
        reuse_rows = [
            json.loads(line)
            for line in Path(args.reuse_manifest)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        reuse_by_key = {
            (
                row["video_id"],
                int(row["fpn_level"]),
                int(row["anchor_index"]),
            ): row
            for row in reuse_rows
        }
        if len(reuse_by_key) != len(reuse_rows):
            raise ValueError("--reuse-manifest contains duplicate anchor keys")

    split = args.subset
    if split is None:
        split = cfg["tgt_train_split"] if args.split == "train" else cfg["tgt_val_split"]
    # Full-video deterministic inference makes the cached absolute timestamps
    # reusable across later training crops.
    dataset = make_dataset(cfg["dataset_name"], False, split, **cfg["dataset"])
    available_videos = video_ids_in(args.video_root)
    dataset.data_list = tuple(
        item
        for item in dataset.data_list
        if item["id"] in available_videos
        and (reuse_rows is None or item["id"] in {row["video_id"] for row in reuse_rows})
    )
    if not dataset.data_list:
        raise RuntimeError("No dataset videos have a matching MP4 in --video-root")

    annotations = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    teacher_state_sha256 = state_dict_sha256(checkpoint[state_key])
    validity_models = {}
    for name, path in named_paths(args.validity_model).items():
        payload, validity_model, artifact_sha256 = load_validity_model(path)
        expected_state = payload["teacher_provenance"][
            "routing_teacher_state_sha256"
        ]
        if teacher_state_sha256 != expected_state:
            raise ValueError(
                "validity model {} belongs to a different teacher state".format(name)
            )
        validate_validity_provenance(
            payload,
            payload["teacher_provenance"],
            assignment_protocol_from_model(model),
            cohort_protocol_from_config(cfg, model, args.beta, alpha),
        )
        validity_models[name] = {
            "model": validity_model_to(validity_model, args.device),
            "threshold": float(payload["threshold"]),
            "artifact_sha256": artifact_sha256,
            "path": path,
        }
    rows = []
    totals = {
        "gt_foreground_anchors": 0,
        "original_selected_foreground_anchors": 0,
        "original_selected_correct_action_anchors": 0,
    }
    with torch.no_grad():
        for raw_video in dataset:
            video = move_video(raw_video, args.device)
            inputs, masks = model.preprocessing([video], mode="eval")
            backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
            logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
            offsets = None
            if validity_models:
                offsets, _ = model.reg_head_target(fpn_feats, fpn_masks)
            points = model.point_generator(fpn_feats)

            metadata = raw_video["metadata"]
            video_id = metadata["video_id"]
            gt_segments, gt_labels = crop_targets(
                annotations[video_id],
                metadata,
                raw_video["feats"].shape[-1],
                cfg,
                args.device,
            )
            gt_cls, _ = model.label_points(points, [gt_segments], [gt_labels])
            gt_cls = gt_cls[0]
            crop_start = float(metadata.get("crop_start", 0))
            feat_stride = float(metadata["feat_stride"])
            fps = float(metadata["fps"])
            frame_offset = 0.5 * float(cfg["dataset"]["num_frames"])
            duration = float(metadata["duration"])

            flat_start = 0
            for level, (point_grid, level_logits, level_mask) in enumerate(
                zip(points, logits, fpn_masks)
            ):
                length = point_grid.shape[0]
                labels = gt_cls[flat_start : flat_start + length]
                flat_start += length
                if level not in args.levels:
                    continue

                probabilities = level_logits[0].transpose(0, 1).sigmoid()
                level_validity = {}
                if validity_models:
                    level_offsets = offsets[level][0].transpose(0, 1)
                    validity_features = anchor_validity_features(
                        probabilities,
                        level_offsets[:, 0],
                        level_offsets[:, 1],
                        float(point_grid[0, 3]),
                        level,
                    )
                    fpn_rows = fpn_feats[level][0].transpose(0, 1)
                    for name, validity in validity_models.items():
                        level_validity[name] = probe_probabilities(
                            validity_features,
                            validity["model"],
                            fpn_features=fpn_rows,
                        )
                top_scores, top_classes = probabilities.topk(k=2, dim=-1)
                scores = top_scores[:, 0]
                pseudo_class = top_classes[:, 0]
                valid = level_mask[0, 0].bool()
                gt_action = labels.sum(dim=-1) > 0
                correct = labels.gather(
                    -1, pseudo_class.unsqueeze(-1)
                ).squeeze(-1) > 0
                original_selected = valid & (scores >= alpha)
                totals["gt_foreground_anchors"] += int((valid & gt_action).sum())
                totals["original_selected_foreground_anchors"] += int(
                    (original_selected & gt_action).sum()
                )
                totals["original_selected_correct_action_anchors"] += int(
                    (original_selected & correct).sum()
                )
                centers = (
                    (point_grid[:, 0] + crop_start) * feat_stride + frame_offset
                ) / fps
                if reuse_rows is None:
                    chosen = choose_indices(
                        scores,
                        centers,
                        valid,
                        args.beta,
                        alpha,
                        args.top_k_per_video_level,
                        args.min_gap_sec,
                        args.score_band,
                    )
                else:
                    chosen = sorted(
                        index
                        for candidate_video, candidate_level, index in reuse_by_key
                        if candidate_video == video_id and candidate_level == level
                    )
                for index in chosen:
                    if index >= probabilities.shape[0] or not bool(valid[index]):
                        raise ValueError(
                            "reused anchor is invalid: {} level {} index {}".format(
                                video_id, level, index
                            )
                        )
                    center_sec = min(max(float(centers[index]), 0.0), duration)
                    absolute_feature_index = int(
                        round(float(point_grid[index, 0]) + crop_start)
                    )
                    gt_vector = labels[index]
                    gt_action = bool(gt_vector.sum() > 0)
                    predicted_class = int(pseudo_class[index])
                    second_class = int(top_classes[index, 1])
                    correct_class = bool(gt_action and gt_vector[predicted_class] > 0)
                    center_all_gt_class_ids = sorted(
                        {
                            int(action["label_id"])
                            for action in annotations[video_id]["annotations"]
                            if float(action["segment"][0]) <= center_sec
                            <= float(action["segment"][1])
                        }
                    )
                    center_gt_class_ids = sorted(
                        {
                            int(action["label_id"])
                            for action in annotations[video_id]["annotations"]
                            if kept_label(
                                int(action["label_id"]),
                                cfg["dataset"].get("action_subsampling", "all"),
                            )
                            and float(action["segment"][0]) <= center_sec
                            <= float(action["segment"][1])
                        }
                    )
                    reused = reuse_by_key.get((video_id, level, int(index)), {})
                    if reused and int(reused["sada_top1_class"]) != predicted_class:
                        raise ValueError(
                            "reused top-1 changed for {}: {} -> {}".format(
                                reused["candidate_id"],
                                reused["sada_top1_class"],
                                predicted_class,
                            )
                        )
                    row = {
                            **reused,
                            "candidate_id": reused.get(
                                "candidate_id",
                                "{}-l{}-{:06d}".format(video_id, level, int(index)),
                            ),
                            "video_id": video_id,
                            "center_sec": center_sec,
                            "absolute_center_time": center_sec,
                            "absolute_feature_index": absolute_feature_index,
                            "fpn_level": level,
                            "anchor_index": int(index),
                            "sada_score": float(scores[index]),
                            "sada_top1_class": predicted_class,
                            "sada_top2_score": float(top_scores[index, 1]),
                            "sada_top2_class": second_class,
                            "score_band": reused.get("score_band", args.score_band),
                            "alpha": alpha,
                            "beta": args.beta,
                            "gt_is_foreground": gt_action,
                            "sada_top1_is_correct": correct_class,
                            "gt_class_ids": torch.nonzero(gt_vector > 0)
                            .flatten()
                            .cpu()
                            .tolist(),
                            "center_gt_class_ids": center_gt_class_ids,
                            "center_all_gt_class_ids": center_all_gt_class_ids,
                            "center_in_target_action": bool(center_gt_class_ids),
                            "sada_top1_matches_center_gt": (
                                predicted_class in center_gt_class_ids
                            ),
                        }
                    if validity_models:
                        row["structure_scores"] = {
                            name: float(level_validity[name][index])
                            for name in validity_models
                        }
                        row["structural_passes"] = {
                            name: (
                                row["structure_scores"][name]
                                >= validity_models[name]["threshold"]
                            )
                            for name in validity_models
                        }
                    rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    foreground = sum(row["gt_is_foreground"] for row in rows)
    center_foreground = sum(row["center_in_target_action"] for row in rows)
    metadata = {
        "selection_uses_target_gt": False,
        "target_gt_use": "attached after candidate selection for offline scoring only",
        "videos": len(dataset),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "levels": list(args.levels),
        "alpha": alpha,
        "beta": args.beta,
        "score_band": (
            sorted({row["score_band"] for row in reuse_rows})
            if reuse_rows is not None
            else args.score_band
        ),
        "top_k_per_video_level": args.top_k_per_video_level,
        "min_gap_sec": args.min_gap_sec,
        "manifest_candidates": len(rows),
        "candidate_selection": (
            "reused exact manifest anchors"
            if reuse_rows is not None
            else "frozen SADA {}-confidence top-k".format(args.score_band)
        ),
        "reuse_manifest": args.reuse_manifest,
        "teacher_state_sha256": teacher_state_sha256,
        "validity_models": {
            name: {
                "path": validity["path"],
                "artifact_sha256": validity["artifact_sha256"],
                "threshold": validity["threshold"],
            }
            for name, validity in validity_models.items()
        },
        "candidate_gt_foreground_anchors": foreground,
        "candidate_center_in_target_action": center_foreground,
        **totals,
    }
    metadata_path = Path(str(output) + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "videos": len(dataset),
                "candidates": len(rows),
                "gt_foreground_for_posthoc_audit": foreground,
                "gt_background_for_posthoc_audit": len(rows) - foreground,
                "output": str(output),
                "metadata": str(metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
