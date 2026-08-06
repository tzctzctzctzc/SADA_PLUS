"""Export GT-free, deduplicated target anchors for offline VLM routing."""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def video_ids_in(root):
    return {
        path.stem
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4" and path.stat().st_size
    }


def time_bin(center_sec, bin_sec):
    return int(math.floor(float(center_sec) / bin_sec + 0.5))


def limit_queries(rows, max_per_video, min_gap_sec):
    if max_per_video <= 0:
        return rows
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    selected = []
    for video_rows in by_video.values():
        chosen = []
        for row in sorted(
            video_rows,
            key=lambda item: (-item["sada_score_max"], item["candidate_id"]),
        ):
            if any(
                abs(row["center_sec"] - old["center_sec"]) < min_gap_sec
                for old in chosen
            ):
                continue
            chosen.append(row)
            if len(chosen) == max_per_video:
                break
        selected.extend(chosen)
    return sorted(selected, key=lambda row: row["candidate_id"])


def decode_teacher_offsets(model, fpn_feats, fpn_masks):
    """Return class-specific decoded offsets as [class, time, left/right]."""
    import torch.nn.functional as F

    raw_offsets, _ = model.reg_head_target(fpn_feats, fpn_masks)
    if not model.use_trident_head:
        return tuple(
            offsets[0].transpose(0, 1).unsqueeze(0).expand(
                model.num_class_orig, -1, -1
            )
            for offsets in raw_offsets
        )

    start_logits, _ = model.start_head(fpn_feats, fpn_masks)
    end_logits, _ = model.end_head(fpn_feats, fpn_masks)
    decoded = []
    for offsets, starts, ends in zip(raw_offsets, start_logits, end_logits):
        offsets = offsets[0].transpose(0, 1)
        starts = starts[0]
        ends = ends[0]

        start_windows = F.pad(starts, (model.num_bins, 0)).unsqueeze(-1)
        window_size = list(start_windows.size())
        window_size[-1] = model.num_bins + 1
        window_size[-2] -= model.num_bins
        window_stride = list(start_windows.stride())
        window_stride[-2] = window_stride[-1]
        start_windows = start_windows.as_strided(
            size=window_size, stride=window_stride
        )

        end_windows = F.pad(ends, (0, model.num_bins)).unsqueeze(-1)
        end_windows = end_windows.as_strided(
            size=window_size, stride=window_stride
        )
        decoded.append(
            model.decode_offset(
                offsets, start_windows, end_windows, mode="test"
            )
        )
    return tuple(decoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--time-bin-sec", type=float, default=0.5)
    parser.add_argument(
        "--exact-anchor", action="store_true",
        help="export one VLM query per exact FPN anchor without merging members",
    )
    parser.add_argument("--max-queries-per-video", type=int, default=0)
    parser.add_argument("--min-query-gap-sec", type=float, default=1.0)
    parser.add_argument("--levels", type=int, nargs="+", default=range(6))
    args = parser.parse_args()

    if args.time_bin_sec <= 0:
        raise ValueError("--time-bin-sec must be positive")

    import torch

    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import move_video

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)

    target_split = cfg["tgt_train_split"]
    dataset = make_dataset(cfg["dataset_name"], False, target_split, **cfg["dataset"])
    expected_videos = len(dataset.data_list)
    available_videos = video_ids_in(args.video_root)
    dataset.data_list = tuple(
        item for item in dataset.data_list if item["id"] in available_videos
    )
    if not dataset.data_list:
        raise RuntimeError("No target-train video has a matching MP4")

    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    if state_key not in checkpoint:
        raise ValueError("checkpoint has no {}".format(state_key))
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    alpha = float(cfg["model"].get("pl_threshold", 0.7))
    groups = defaultdict(list)
    exact_rows = []
    level_counts = Counter()
    valid_anchors = 0
    low_confidence_anchors = 0

    with torch.no_grad():
        for raw_video in dataset:
            video = move_video(raw_video, args.device)
            inputs, masks = model.preprocessing([video], mode="eval")
            backbone_feats, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn_feats, fpn_masks = model.neck_target(backbone_feats, backbone_masks)
            logits, _ = model.cls_head_target(fpn_feats, fpn_masks)
            points = model.point_generator(fpn_feats)
            decoded_offsets = (
                decode_teacher_offsets(model, fpn_feats, fpn_masks)
                if args.exact_anchor
                else None
            )

            metadata = raw_video["metadata"]
            video_id = str(metadata["video_id"])
            crop_start = float(metadata.get("crop_start", 0))
            feat_stride = float(metadata["feat_stride"])
            fps = float(metadata["fps"])
            frame_offset = 0.5 * float(cfg["dataset"]["num_frames"])
            duration = float(metadata["duration"])
            global_anchor_offset = 0

            for level, (point_grid, level_logits, level_mask) in enumerate(
                zip(points, logits, fpn_masks)
            ):
                level_length = int(point_grid.shape[0])
                if level not in args.levels:
                    global_anchor_offset += level_length
                    continue
                probabilities = level_logits[0].transpose(0, 1).sigmoid()
                top_scores, top_classes = probabilities.topk(k=2, dim=-1)
                valid = level_mask[0, 0].bool()
                eligible = valid & (top_scores[:, 0] > args.beta) & (
                    top_scores[:, 0] < alpha
                )
                valid_anchors += int(valid.sum())
                low_confidence_anchors += int(eligible.sum())
                level_counts[level] += int(eligible.sum())
                centers = (
                    (point_grid[:, 0] + crop_start) * feat_stride + frame_offset
                ) / fps

                for index in torch.nonzero(eligible).flatten().tolist():
                    center_sec = min(max(float(centers[index]), 0.0), duration)
                    top1 = int(top_classes[index, 0])
                    top2 = int(top_classes[index, 1])
                    if args.exact_anchor:
                        time_us = int(round(center_sec * 1000000.0))
                        global_index = global_anchor_offset + int(index)
                        teacher_offsets = decoded_offsets[level][top1, index]
                        reg_left = float(teacher_offsets[0])
                        reg_right = float(teacher_offsets[1])
                        point_stride = float(point_grid[index, 3])
                        reg_range_min = float(point_grid[index, 1])
                        reg_range_max = float(point_grid[index, 2])
                        max_distance = max(reg_left, reg_right) * point_stride
                        center_distance = abs(reg_right - reg_left) / 2.0
                        center_radius = float(model.train_center_sample_radius)
                        range_eligible = (
                            reg_range_min <= max_distance <= reg_range_max
                        )
                        center_eligible = center_distance <= center_radius
                        duration_eligible = reg_left + reg_right > 0.0
                        structural_eligible = (
                            range_eligible
                            and center_eligible
                            and duration_eligible
                        )
                        structure = {
                            "teacher_regression_left": reg_left,
                            "teacher_regression_right": reg_right,
                            "point_stride": point_stride,
                            "point_regression_range_min": reg_range_min,
                            "point_regression_range_max": reg_range_max,
                            "teacher_predicted_max_distance": max_distance,
                            "teacher_predicted_center_distance": center_distance,
                            "center_sample_radius": center_radius,
                            "teacher_range_eligible": range_eligible,
                            "teacher_center_eligible": center_eligible,
                            "teacher_duration_eligible": duration_eligible,
                            "teacher_structural_eligible": structural_eligible,
                        }
                        candidate_id = (
                            "{}-l{:02d}-g{:07d}-t{:012d}-c{:02d}".format(
                                video_id, level, global_index, time_us, top1
                            )
                        )
                        member = {
                            "fpn_level": level,
                            "anchor_index": int(index),
                            "global_anchor_index": global_index,
                            "absolute_center_time": center_sec,
                            "teacher_top1_class": top1,
                            "center_sampling_position": float(point_grid[index, 0]),
                            **structure,
                        }
                        score = float(top_scores[index, 0])
                        exact_rows.append(
                            {
                                "candidate_id": candidate_id,
                                "anchor_id": candidate_id,
                                "frame_cache_id": candidate_id,
                                "video_id": video_id,
                                "fpn_level": level,
                                "anchor_index": int(index),
                                "global_anchor_index": global_index,
                                "absolute_center_time": center_sec,
                                "center_sec": center_sec,
                                "teacher_top1_class": top1,
                                "sada_top1_class": top1,
                                "sada_top2_class": top2,
                                "sada_score_min": score,
                                "sada_score_max": score,
                                "sada_score_mean": score,
                                **structure,
                                "member_anchors": [member],
                            }
                        )
                        continue
                    bin_index = time_bin(center_sec, args.time_bin_sec)
                    groups[(video_id, bin_index, top1)].append(
                        {
                            "fpn_level": level,
                            "anchor_index": int(index),
                            "center_sec": center_sec,
                            "sada_score": float(top_scores[index, 0]),
                            "sada_top2_class": top2,
                        }
                    )
                global_anchor_offset += level_length

    member_counts = Counter()
    if args.exact_anchor:
        rows = exact_rows
        member_counts.update(len(row["member_anchors"]) for row in rows)
    else:
        rows = []
        for (video_id, bin_index, top1), members in sorted(groups.items()):
            representative = max(members, key=lambda member: member["sada_score"])
            center_sec = representative["center_sec"]
            top2 = representative["sada_top2_class"]
            scores = [member["sada_score"] for member in members]
            candidate_id = "{}-tb{:07d}-c{:02d}-c{:02d}".format(
                video_id, bin_index, top1, top2
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "anchor_id": candidate_id,
                    "frame_cache_id": candidate_id,
                    "video_id": video_id,
                    "center_sec": center_sec,
                    "time_bin_index": bin_index,
                    "time_bin_sec": args.time_bin_sec,
                    "sada_top1_class": top1,
                    "sada_top2_class": top2,
                    "sada_score_min": min(scores),
                    "sada_score_max": max(scores),
                    "sada_score_mean": sum(scores) / len(scores),
                    "member_anchors": [
                        {
                            "fpn_level": member["fpn_level"],
                            "anchor_index": member["anchor_index"],
                        }
                        for member in members
                    ],
                }
            )
            member_counts[len(members)] += 1

    all_deduplicated_queries = len(rows)
    rows = limit_queries(
        rows, args.max_queries_per_video, args.min_query_gap_sec
    )
    selected_member_counts = Counter(len(row["member_anchors"]) for row in rows)
    selected_anchor_count = sum(len(row["member_anchors"]) for row in rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "selection_uses_target_gt": False,
        "source": "frozen SADA target prediction",
        "target_split": list(target_split),
        "expected_target_videos": expected_videos,
        "available_target_videos": len(dataset),
        "missing_target_videos": expected_videos - len(dataset),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "levels": list(args.levels),
        "beta": args.beta,
        "alpha": alpha,
        "exact_anchor": args.exact_anchor,
        "time_bin_sec": None if args.exact_anchor else args.time_bin_sec,
        "valid_anchors": valid_anchors,
        "low_confidence_anchors": low_confidence_anchors,
        "low_confidence_by_level": dict(sorted(level_counts.items())),
        "deduplicated_queries_before_budget": all_deduplicated_queries,
        "deduplication_ratio": (
            low_confidence_anchors / all_deduplicated_queries
            if all_deduplicated_queries
            else None
        ),
        "max_queries_per_video": args.max_queries_per_video,
        "min_query_gap_sec": args.min_query_gap_sec,
        "deduplicated_vlm_queries": len(rows),
        "selected_member_anchors": selected_anchor_count,
        "members_per_query_before_budget": dict(sorted(member_counts.items())),
        "selected_members_per_query": dict(sorted(selected_member_counts.items())),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
