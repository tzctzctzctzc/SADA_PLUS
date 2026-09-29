import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def stable_rows(rows, seed):
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            "{}:{}".format(seed, row["candidate_id"]).encode()
        ).hexdigest(),
    )


def limit_per_video(rows, max_per_video, max_proposals, seed):
    grouped = defaultdict(list)
    for row in stable_rows(rows, seed):
        grouped[row["video_id"]].append(row)
    selected = []
    for video_id in sorted(grouped):
        selected.extend(grouped[video_id][:max_per_video])
    return stable_rows(selected, seed)[:max_proposals]


def video_ids_in(root):
    return {
        path.stem
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4" and path.stat().st_size
    }


def move_video(video, device):
    import torch

    result = dict(video)
    result["metadata"] = dict(video["metadata"])
    for key, value in result.items():
        if isinstance(value, torch.Tensor) and key != "full_feats":
            result[key] = value.to(device)
    return result


def center_annotations(database, video_id, center_sec):
    labels = []
    for item in database[video_id]["annotations"]:
        start, end = (float(value) for value in item["segment"])
        if start <= center_sec <= end:
            labels.append(int(item["label_id"]))
    return sorted(set(labels))


def trident_neighbour_logits(boundary_logits, num_bins, side):
    """Arrange boundary logits exactly as TriDet inference does."""
    import torch.nn.functional as F

    if side == "start":
        padded = F.pad(boundary_logits, (num_bins, 0), value=0)
    elif side == "end":
        padded = F.pad(boundary_logits, (0, num_bins), value=0)
    else:
        raise ValueError("side must be start or end")
    padded = padded.unsqueeze(-1)
    size = list(padded.size())
    size[-1] = num_bins + 1
    size[-2] -= num_bins
    stride = list(padded.stride())
    stride[-2] = stride[-1]
    return padded.as_strided(size=size, stride=stride)


def load_checkpoint(torch, checkpoint_path, device, trusted_legacy_pickle=False):
    """Prefer safe loading; allow legacy pickle only after matched-baseline audit."""
    try:
        return torch.load(
            checkpoint_path, map_location=device, weights_only=True
        )
    except pickle.UnpicklingError as exc:
        if not trusted_legacy_pickle:
            raise RuntimeError(
                "checkpoint needs legacy pickle loading; verify it against a "
                "completed matched baseline before opting in"
            ) from exc
        return torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument(
        "--include-missing-videos",
        action="store_true",
        help="export from feature files even when raw MP4s will be streamed later",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--domain", choices=("source", "target"), required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("student", "ema"), default="ema")
    parser.add_argument(
        "--trident-head",
        action="store_true",
        help="export proposals from a checkpoint trained with the TriDet head",
    )
    parser.add_argument("--levels", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--max-per-video", type=int, default=30)
    parser.add_argument("--max-proposals", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--trusted-matched-baseline-root",
        help="completed SlowFast matched baseline; required only for legacy checkpoint pickle fallback",
    )
    args = parser.parse_args()

    import torch

    from libs.datasets import make_dataset
    from libs.models.SADA import make_meta_arch
    from libs.models.SADA.stage1_calibration import model_state_sha256
    from libs.models.SADA.vlm_routing import file_sha256
    from libs.utils.utils import _update_config, read_yaml

    trusted_legacy_pickle = False
    trusted_baseline = None
    if args.trusted_matched_baseline_root:
        if (args.domain, args.split) not in (('target', 'train'), ('source', 'val')):
            raise ValueError("matched-baseline export supports target-train and source-validation calibration only")
        baseline_root = Path(args.trusted_matched_baseline_root).resolve()
        complete = json.loads((baseline_root / "complete.json").read_text())
        if complete.get("model") != "slowfast" or complete.get("seed") != 6:
            raise ValueError("trusted baseline must be a completed SlowFast seed6 run")
        from tools.run_rq3_matrix import completed_baseline

        _, audited_checkpoint = completed_baseline(
            baseline_root, complete["shift"]
        )
        if Path(audited_checkpoint).resolve() != Path(args.checkpoint).resolve():
            raise ValueError("checkpoint is not the source-val-selected baseline SO checkpoint")
        if Path(complete["so"]).resolve() != Path(args.config).resolve():
            raise ValueError("config is not the audited baseline SO config")
        trusted_legacy_pickle = True
        trusted_baseline = {
            "root": str(baseline_root),
            "shift": complete["shift"],
            "seed": 6,
        }

    if args.max_per_video <= 0 or args.max_proposals <= 0:
        raise ValueError("proposal limits must be positive")
    levels = tuple(sorted(set(args.levels)))
    if not levels or any(level < 0 or level > 5 for level in levels):
        raise ValueError("levels must be unique values from 0 to 5")

    cfg = read_yaml(args.config)
    cfg["stage1"] = {"enabled": False, "perturbation": "shift"}
    cfg["vlm_routing"] = {"enabled": False}
    cfg["vlm_target_adapter"] = {"enabled": False}
    if args.trident_head:
        cfg["model"]["backbone_type"] = "SGP"
        cfg["model"]["use_trident_head"] = True
    _update_config(cfg)
    split_key = "{}_{}_split".format(
        "src" if args.domain == "source" else "tgt", args.split
    )
    dataset = make_dataset(
        cfg["dataset_name"], False, cfg[split_key], **cfg["dataset"]
    )
    available = video_ids_in(args.video_root)
    expected_videos = len(dataset.data_list)
    expected_video_ids = sorted(item['id'] for item in dataset.data_list)
    available_dataset_videos = sum(
        item["id"] in available for item in dataset.data_list
    )
    if not args.include_missing_videos:
        dataset.data_list = tuple(
            item for item in dataset.data_list if item["id"] in available
        )
    if not dataset.data_list:
        raise RuntimeError("no dataset video has a matching MP4")

    if str(args.device).startswith('cuda'):
        from tools.rq3_gpu import guard_before_cuda
        guard_before_cuda()
    model = make_meta_arch(cfg["model_name"], cfg, device=args.device).to(args.device)
    checkpoint = load_checkpoint(
        torch, args.checkpoint, args.device,
        trusted_legacy_pickle=trusted_legacy_pickle,
    )
    state_key = "ema_state_dict" if args.weights == "ema" else "state_dict"
    state_dict = checkpoint.get(state_key, checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    provenance = {
        "selection_teacher_checkpoint_sha256": file_sha256(args.checkpoint),
        "selection_teacher_state_sha256": model_state_sha256(model),
        "selection_teacher_weights": args.weights,
        "selection_alpha": float(cfg["model"].get("pl_threshold", 0.7)),
        "selection_levels": list(levels),
        "trusted_matched_baseline": trusted_baseline,
    }
    database = json.loads(
        Path(cfg["dataset"]["json_file"]).read_text(encoding="utf-8")
    )["database"]

    grouped = defaultdict(list)
    with torch.no_grad():
        for raw_video in dataset:
            video = move_video(raw_video, args.device)
            inputs, masks = model.preprocessing([video], mode="validation")
            backbone, backbone_masks, _ = model.backbone_target(inputs, masks)
            fpn, fpn_masks = model.neck_target(backbone, backbone_masks)
            logits, _ = model.cls_head_target(fpn, fpn_masks)
            offsets, _ = model.reg_head_target(fpn, fpn_masks)
            if model.use_trident_head:
                start_logits, _ = model.start_head(fpn, fpn_masks)
                end_logits, _ = model.end_head(fpn, fpn_masks)
            else:
                start_logits = (None,) * len(offsets)
                end_logits = (None,) * len(offsets)
            points = model.point_generator(fpn)
            gt_cls = None
            if args.domain == "source":
                gt_cls, _ = model.label_points(
                    points, [video["segments"]], [video["labels"]]
                )
                gt_cls = gt_cls[0]
            metadata = raw_video["metadata"]
            video_id = str(metadata["video_id"])
            feat_stride = float(metadata["feat_stride"])
            frame_offset = 0.5 * float(metadata["feat_num_frames"])
            fps = float(metadata["fps"])
            duration = float(metadata["duration"])
            start = 0
            for level, (
                grid, mask, raw_logits, raw_offsets, raw_start, raw_end
            ) in enumerate(
                zip(points, fpn_masks, logits, offsets, start_logits, end_logits)
            ):
                length = grid.shape[0]
                if level not in levels:
                    start += length
                    continue
                probabilities = raw_logits[0].transpose(0, 1).sigmoid()
                scores, classes = probabilities.max(dim=-1)
                valid = mask[0, 0].bool()
                selected = valid & (scores >= provenance["selection_alpha"])
                raw_offsets = raw_offsets[0].transpose(0, 1)
                if model.use_trident_head:
                    start_neighbours = trident_neighbour_logits(
                        raw_start[0], model.num_bins, "start"
                    )
                    end_neighbours = trident_neighbour_logits(
                        raw_end[0], model.num_bins, "end"
                    )
                    decoded = model.decode_offset(
                        raw_offsets,
                        start_neighbours,
                        end_neighbours,
                        mode="test",
                    )
                else:
                    decoded = raw_offsets
                for index in torch.nonzero(selected).flatten().tolist():
                    coordinate = float(grid[index, 0])
                    absolute_index = int(round(coordinate))
                    if abs(coordinate - absolute_index) > 1e-6:
                        raise ValueError("FPN point is not an integer feature coordinate")
                    center_sec = min(
                        max((absolute_index * feat_stride + frame_offset) / fps, 0.0),
                        duration,
                    )
                    if model.use_trident_head:
                        proposal_offset = decoded[int(classes[index]), index]
                    else:
                        proposal_offset = decoded[index]
                    left_grid = coordinate - float(proposal_offset[0]) * float(grid[index, 3])
                    right_grid = coordinate + float(proposal_offset[1]) * float(grid[index, 3])
                    proposal_start = min(
                        max((left_grid * feat_stride + frame_offset) / fps, 0.0),
                        duration,
                    )
                    proposal_end = min(
                        max((right_grid * feat_stride + frame_offset) / fps, 0.0),
                        duration,
                    )
                    member = {
                        "fpn_level": level,
                        "score": float(scores[index]),
                        "teacher_top1_class": int(classes[index]),
                        "proposal_start_sec": proposal_start,
                        "proposal_end_sec": proposal_end,
                        "absolute_center_time": center_sec,
                    }
                    if gt_cls is not None:
                        assigned = gt_cls[start + index]
                        member["gt_assigned_class_ids"] = torch.nonzero(
                            assigned > 0
                        ).flatten().tolist()
                    grouped[(video_id, absolute_index)].append(member)
                start += length

    rows = []
    for (video_id, absolute_index), members in sorted(grouped.items()):
        strongest = max(members, key=lambda item: item["score"])
        center_sec = strongest["absolute_center_time"]
        row = {
            "candidate_id": "{}-p{:07d}".format(video_id, absolute_index),
            "video_id": video_id,
            "absolute_feature_index": absolute_index,
            "absolute_center_time": center_sec,
            "center_sec": center_sec,
            "proposal_start_sec": strongest["proposal_start_sec"],
            "proposal_end_sec": strongest["proposal_end_sec"],
            "source_fpn_levels": [strongest["fpn_level"]],
            "teacher_top1_class": strongest["teacher_top1_class"],
            "teacher_score": strongest["score"],
        }
        if args.domain == "source":
            assigned = strongest["gt_assigned_class_ids"]
            center_labels = center_annotations(database, video_id, center_sec)
            row["gt_assigned_class_ids"] = assigned
            row["center_all_gt_class_ids"] = center_labels
            row["semantic_gt_kind"] = (
                "KNOWN_ACTION" if any(label < 10 for label in center_labels)
                else "UNKNOWN_ACTION" if center_labels
                else "BACKGROUND"
            )
        rows.append(row)

    before_limit = len(rows)
    rows = limit_per_video(
        rows, args.max_per_video, args.max_proposals, args.seed
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "schema_version": "vlm-target-adapter-proposal-manifest-v1",
        "domain": args.domain,
        "split": args.split,
        "selection_uses_gt": False,
        "vlm_input_fields": [
            "chronological_proposal_frames",
            "proposal_temporal_extent",
        ],
        "vlm_forbidden_fields": [
            "teacher_top1_class",
            "teacher_score",
            "gt_assigned_class_ids",
            "center_all_gt_class_ids",
            "semantic_gt_kind",
        ],
        "selection_teacher_provenance": provenance,
        "expected_videos": expected_videos,
        "expected_video_ids": expected_video_ids,
        "available_videos": available_dataset_videos,
        "video_filter_applied": not args.include_missing_videos,
        "proposals_before_budget": before_limit,
        "selected_proposals": len(rows),
        "max_per_video": args.max_per_video,
        "max_proposals": args.max_proposals,
        "seed": args.seed,
    }
    Path(str(output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **metadata}, indent=2))


if __name__ == "__main__":
    main()
