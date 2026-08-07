"""Export GT-free VLM queries keyed by absolute feature position."""

import argparse
import json
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


def decode_teacher_offsets(model, fpn_feats, fpn_masks, domain="target"):
    """Return class-specific decoded offsets as [class, time, left/right]."""
    import torch.nn.functional as F

    reg_head = model.reg_head if domain == "source" else model.reg_head_target
    raw_offsets, _ = reg_head(fpn_feats, fpn_masks)
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
            model.decode_offset(offsets, start_windows, end_windows, mode="test")
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
    parser.add_argument("--max-queries-per-video", type=int, default=0)
    parser.add_argument("--min-query-gap-sec", type=float, default=1.0)
    parser.add_argument("--levels", type=int, nargs="+", default=range(6))
    parser.add_argument(
        "--exact-anchor",
        action="store_true",
        help="compatibility flag; absolute-feature exact semantics are now mandatory",
    )
    args = parser.parse_args()

    import torch

    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.utils.utils import _update_config, read_yaml
    from tools.audit_original_sada_pseudolabels import move_video

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    _update_config(cfg)
    dataset = make_dataset(
        cfg["dataset_name"], False, cfg["tgt_train_split"], **cfg["dataset"]
    )
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
    grouped = defaultdict(list)
    level_counts = Counter()
    valid_anchors = 0
    low_confidence_anchors = 0
    with torch.no_grad():
        for raw_video in dataset:
            video = move_video(raw_video, args.device)
            inputs, masks = model.preprocessing([video], mode="eval")
            backbone, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn, fpn_masks = model.neck_target(backbone, backbone_masks)
            logits, _ = model.cls_head_target(fpn, fpn_masks)
            points = model.point_generator(fpn)
            metadata = raw_video["metadata"]
            video_id = str(metadata["video_id"])
            crop_start = int(metadata.get("crop_start", 0))
            feat_stride = float(metadata["feat_stride"])
            fps = float(metadata["fps"])
            frame_offset = 0.5 * float(cfg["dataset"]["num_frames"])
            duration = float(metadata["duration"])

            for level, (point_grid, level_logits, level_mask) in enumerate(
                zip(points, logits, fpn_masks)
            ):
                if level not in args.levels:
                    continue
                probabilities = level_logits[0].transpose(0, 1).sigmoid()
                top_scores = probabilities.topk(k=2, dim=-1).values
                valid = level_mask[0, 0].bool()
                eligible = valid & (top_scores[:, 0] > args.beta) & (
                    top_scores[:, 0] < alpha
                )
                valid_anchors += int(valid.sum())
                low_confidence_anchors += int(eligible.sum())
                level_counts[level] += int(eligible.sum())
                for index in torch.nonzero(eligible).flatten().tolist():
                    local_coordinate = float(point_grid[index, 0])
                    rounded = int(round(local_coordinate))
                    if abs(local_coordinate - rounded) > 1e-6:
                        raise ValueError("FPN point is not an integer feature coordinate")
                    absolute_index = crop_start + rounded
                    center_sec = (
                        absolute_index * feat_stride + frame_offset
                    ) / fps
                    center_sec = min(max(float(center_sec), 0.0), duration)
                    grouped[(video_id, absolute_index)].append(
                        {
                            "center_sec": center_sec,
                            "fpn_level": level,
                            "sada_score": float(top_scores[index, 0]),
                        }
                    )

    rows = []
    for (video_id, absolute_index), members in sorted(grouped.items()):
        center_sec = members[0]["center_sec"]
        if any(abs(member["center_sec"] - center_sec) > 1e-6 for member in members):
            raise ValueError("one absolute feature index has inconsistent timestamps")
        time_us = int(round(center_sec * 1000000.0))
        candidate_id = "{}-p{:07d}-t{:012d}".format(
            video_id, absolute_index, time_us
        )
        scores = [member["sada_score"] for member in members]
        rows.append(
            {
                "candidate_id": candidate_id,
                "anchor_id": candidate_id,
                "frame_cache_id": candidate_id,
                "video_id": video_id,
                "absolute_feature_index": absolute_index,
                "absolute_center_time": center_sec,
                "center_sec": center_sec,
                "source_candidate_count": len(members),
                "source_fpn_levels": sorted(
                    {member["fpn_level"] for member in members}
                ),
                "sada_score_min": min(scores),
                "sada_score_max": max(scores),
                "sada_score_mean": sum(scores) / len(scores),
            }
        )

    query_count_before_budget = len(rows)
    rows = limit_queries(rows, args.max_queries_per_video, args.min_query_gap_sec)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "selection_uses_target_gt": False,
        "source": "frozen SADA target prediction; VLM input remains candidate-blind",
        "semantic_key_fields": ["video_id", "absolute_feature_index"],
        "target_split": list(cfg["tgt_train_split"]),
        "expected_target_videos": expected_videos,
        "available_target_videos": len(dataset),
        "missing_target_videos": expected_videos - len(dataset),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "weights": args.weights,
        "levels": list(args.levels),
        "beta": args.beta,
        "alpha": alpha,
        "valid_anchors": valid_anchors,
        "low_confidence_anchors": low_confidence_anchors,
        "low_confidence_by_level": dict(sorted(level_counts.items())),
        "absolute_semantic_queries_before_budget": query_count_before_budget,
        "deduplication_ratio": (
            low_confidence_anchors / query_count_before_budget
            if query_count_before_budget
            else None
        ),
        "max_queries_per_video": args.max_queries_per_video,
        "min_query_gap_sec": args.min_query_gap_sec,
        "selected_vlm_queries": len(rows),
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
