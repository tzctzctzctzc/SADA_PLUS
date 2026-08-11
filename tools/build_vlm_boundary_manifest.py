"""Build coarse and fine VLM query manifests from frozen SADA proposals."""

import argparse
import hashlib
import json
import string
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vlm_boundary_utils import (
    QUERY_SCHEMA_VERSION,
    file_sha256,
    fine_step_seconds,
    json_sha256,
    read_jsonl,
    transition_pair,
    transition_pair_grid,
    video_frame_index,
    write_jsonl,
)


def dynamic_context_window(
    segment,
    duration,
    origin_stride_seconds,
    context_ratio=1.0,
    stride_multiplier=4.0,
    minimum_margin=4.0,
    maximum_margin=8.0,
):
    left, right = map(float, segment)
    proposal_duration = max(0.0, right - left)
    margin = max(
        float(minimum_margin),
        proposal_duration * float(context_ratio),
        float(origin_stride_seconds) * float(stride_multiplier),
    )
    margin = min(margin, float(maximum_margin))
    return [max(0.0, left - margin), min(float(duration), right + margin)]


def _option_candidates(query_id, side, pairs, seed):
    options = option_permutation(query_id, side, len(pairs), int(seed))
    return [
        {
            "option": options[index],
            **pair,
        }
        for index, pair in enumerate(pairs)
    ]


def _materialize_frames(frame_indices, fps):
    ordered = sorted(set(map(int, frame_indices)))
    frames = [
        {
            "frame_id": "F{}".format(index + 1),
            "frame_index": int(frame_index),
            "timestamp_sec": float(frame_index) / float(fps),
        }
        for index, frame_index in enumerate(ordered)
    ]
    return frames, {frame["frame_index"]: frame["frame_id"] for frame in frames}


def _named_candidates(candidates, frame_ids):
    return [
        {
            "option": candidate["option"],
            "before_frame_id": frame_ids[candidate["before_frame_index"]],
            "after_frame_id": frame_ids[candidate["after_frame_index"]],
            "timestamp_sec": float(candidate["timestamp_sec"]),
        }
        for candidate in candidates
    ]


def adaptive_coarse_evidence(
    segment,
    window,
    video_duration,
    fps,
    query_id,
    context_count=6,
    candidate_count=5,
    inside_count=4,
    transition_gap_frames=2,
    option_seed=20260810,
):
    """Separate broad context/instance anchors from selectable frame pairs."""
    if min(int(context_count), int(candidate_count), int(inside_count)) < 2:
        raise ValueError("all coarse sampling counts must be at least two")
    if int(candidate_count) % 2 == 0:
        raise ValueError("coarse candidate count must be odd")
    left, right = map(float, segment)
    context_indices = {
        video_frame_index(value, video_duration, fps)
        for value in np.linspace(window[0], window[1], int(context_count))
    }
    anchor_indices = {
        video_frame_index(value, video_duration, fps)
        for value in np.linspace(left, right, int(inside_count) + 2)[1:-1]
    }
    minimum_duration = 1.0 / float(fps)
    search_ranges = {
        "start": [float(window[0]), min(float(window[1]), right - minimum_duration)],
        "end": [max(float(window[0]), left + minimum_duration), float(window[1])],
    }
    side_candidates = {}
    all_indices = set(context_indices) | set(anchor_indices)
    for side, (search_left, search_right) in search_ranges.items():
        pairs = []
        seen = set()
        for center in np.linspace(search_left, search_right, int(candidate_count)):
            pair = transition_pair(
                center,
                video_duration,
                fps,
                transition_gap_frames,
            )
            key = (pair["before_frame_index"], pair["after_frame_index"])
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
                all_indices.update(key)
        side_candidates[side] = _option_candidates(
            query_id, side, pairs, option_seed
        )
    frames, frame_ids = _materialize_frames(all_indices, fps)
    return {
        "frames": frames,
        "context_frame_ids": [frame_ids[index] for index in sorted(context_indices)],
        "instance_anchor_frame_ids": [
            frame_ids[index] for index in sorted(anchor_indices)
        ],
        "start_candidates": _named_candidates(side_candidates["start"], frame_ids),
        "end_candidates": _named_candidates(side_candidates["end"], frame_ids),
    }


def build_coarse_rows(
    proposals,
    context_frame_count=6,
    boundary_candidate_count=5,
    inside_frame_count=4,
    transition_gap_frames=2,
    option_seed=20260810,
    task_mode="validity-and-boundary",
    **window_options
):
    if task_mode not in ("validity-and-boundary", "boundary-only"):
        raise ValueError("unknown task mode")
    rows = []
    for proposal in proposals:
        window = dynamic_context_window(
            proposal["coarse_segment_seconds"],
            proposal["video_duration_seconds"],
            proposal["origin_stride_seconds"],
            **window_options,
        )
        query_id = proposal["proposal_id"] + ":coarse"
        evidence = adaptive_coarse_evidence(
            proposal["coarse_segment_seconds"],
            window,
            proposal["video_duration_seconds"],
            proposal["fps"],
            query_id,
            context_count=context_frame_count,
            candidate_count=boundary_candidate_count,
            inside_count=inside_frame_count,
            transition_gap_frames=transition_gap_frames,
            option_seed=option_seed,
        )
        row = {
            "query_schema_version": QUERY_SCHEMA_VERSION,
            "stage": "coarse",
            "task_mode": task_mode,
            "query_id": query_id,
            "proposal_id": proposal["proposal_id"],
            "video_id": proposal["video_id"],
            "video_duration_seconds": float(proposal["video_duration_seconds"]),
            "predicted_class_id": int(proposal["predicted_class_id"]),
            "predicted_class_name": proposal["predicted_class_name"],
            "coarse_segment_seconds": list(map(float, proposal["coarse_segment_seconds"])),
            "context_window_seconds": window,
            "fps": float(proposal["fps"]),
            "sampling_protocol": "separate-context-real-frame-transitions-v3",
            **evidence,
        }
        row["query_sha256"] = json_sha256(row)
        rows.append(row)
    return rows


def decision_by_query(predictions):
    result = {}
    for row in predictions:
        if row.get("run_status") != "ok" or not isinstance(row.get("decision"), dict):
            continue
        result[row["query_id"]] = row["decision"]
    return result


def selected_candidate_time(query, side, option):
    for candidate in query["{}_candidates".format(side)]:
        if candidate["option"] == option:
            return float(candidate["timestamp_sec"])
    raise ValueError("unknown {} option {}".format(side, option))


def option_permutation(query_id, side, count, seed):
    if count > len(string.ascii_uppercase):
        raise ValueError("too many fine candidates")
    digest = hashlib.sha256(
        "{}|{}|{}".format(seed, query_id, side).encode("utf-8")
    ).digest()
    keyed = sorted(range(count), key=lambda index: (digest[index % len(digest)], index))
    options = list(string.ascii_uppercase[:count])
    result = [None] * count
    for rank, index in enumerate(keyed):
        result[index] = options[rank]
    return result


def build_fine_rows(
    proposals,
    coarse_queries,
    coarse_predictions,
    candidate_count=7,
    maximum_step=0.3,
    minimum_step=0.1,
    transition_gap_frames=2,
    option_seed=20260810,
):
    proposal_map = {row["proposal_id"]: row for row in proposals}
    decisions = decision_by_query(coarse_predictions)
    rows = []
    for query in coarse_queries:
        decision = decisions.get(query["query_id"])
        boundary_only = query.get("task_mode") == "boundary-only"
        if not decision or (
            not boundary_only and decision.get("action_status") != "VALID_ACTION"
        ):
            continue
        proposal = proposal_map[query["proposal_id"]]
        step = fine_step_seconds(
            proposal["origin_stride_seconds"], maximum_step, minimum_step
        )
        for side in ("start", "end"):
            side_decision = decision.get("{}_decision".format(side))
            # KEEP_SADA is terminal: once the coarse audit finds no clear
            # improvement, Fine must not reopen that side and introduce a
            # lower-confidence replacement.
            if side_decision != "SELECT_CANDIDATE":
                continue
            side_index = 0 if side == "start" else 1
            sada_boundary = float(proposal["coarse_segment_seconds"][side_index])
            center = sada_boundary
            if side_decision == "SELECT_CANDIDATE":
                center = selected_candidate_time(
                    query, side, decision["selected_{}_option".format(side)]
                )
            fine_query_id = proposal["proposal_id"] + ":fine:" + side
            raw_candidates = _option_candidates(
                fine_query_id,
                side,
                transition_pair_grid(
                    center,
                    step,
                    int(candidate_count),
                    proposal["video_duration_seconds"],
                    proposal["fps"],
                    transition_gap_frames,
                ),
                option_seed,
            )
            coarse_frame_map = {
                frame["frame_id"]: frame for frame in query["frames"]
            }
            anchor_indices = {
                int(coarse_frame_map[frame_id]["frame_index"])
                for frame_id in query["instance_anchor_frame_ids"]
            }
            all_indices = set(anchor_indices)
            for candidate in raw_candidates:
                all_indices.add(candidate["before_frame_index"])
                all_indices.add(candidate["after_frame_index"])
            frames, frame_ids = _materialize_frames(all_indices, proposal["fps"])
            candidates = _named_candidates(raw_candidates, frame_ids)
            row = {
                "query_schema_version": QUERY_SCHEMA_VERSION,
                "stage": "fine",
                "task_mode": query.get("task_mode", "validity-and-boundary"),
                "boundary_side": side,
                "query_id": fine_query_id,
                "proposal_id": proposal["proposal_id"],
                "coarse_query_id": query["query_id"],
                "video_id": proposal["video_id"],
                "video_duration_seconds": float(proposal["video_duration_seconds"]),
                "predicted_class_id": int(proposal["predicted_class_id"]),
                "predicted_class_name": proposal["predicted_class_name"],
                "coarse_segment_seconds": list(
                    map(float, proposal["coarse_segment_seconds"])
                ),
                "fps": float(proposal["fps"]),
                "instance_description": decision["instance_description"],
                "instance_anchor_frame_ids": [
                    frame_ids[index] for index in sorted(anchor_indices)
                ],
                "sada_boundary_seconds": sada_boundary,
                "rough_boundary_seconds": center,
                "fine_step_seconds": step,
                "frames": frames,
                "candidates": candidates,
            }
            row["query_sha256"] = json_sha256(row)
            rows.append(row)
    return rows


def add_common_arguments(parser):
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--output", required=True)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    coarse = subparsers.add_parser("coarse")
    add_common_arguments(coarse)
    coarse.add_argument("--context-frame-count", type=int, default=6)
    coarse.add_argument("--boundary-candidate-count", type=int, default=5)
    coarse.add_argument("--inside-frame-count", type=int, default=4)
    coarse.add_argument("--transition-gap-frames", type=int, default=2)
    coarse.add_argument("--option-seed", type=int, default=20260810)
    coarse.add_argument("--context-ratio", type=float, default=1.0)
    coarse.add_argument("--stride-multiplier", type=float, default=4.0)
    coarse.add_argument("--minimum-margin", type=float, default=4.0)
    coarse.add_argument("--maximum-margin", type=float, default=8.0)
    coarse.add_argument(
        "--task-mode",
        choices=("validity-and-boundary", "boundary-only"),
        default="validity-and-boundary",
    )

    fine = subparsers.add_parser("fine")
    add_common_arguments(fine)
    fine.add_argument("--coarse-manifest", required=True)
    fine.add_argument("--coarse-predictions", required=True)
    fine.add_argument("--candidate-count", type=int, default=7)
    fine.add_argument("--maximum-step", type=float, default=0.3)
    fine.add_argument("--minimum-step", type=float, default=0.1)
    fine.add_argument("--transition-gap-frames", type=int, default=2)
    fine.add_argument("--option-seed", type=int, default=20260810)
    args = parser.parse_args()

    proposals = read_jsonl(args.proposals)
    if args.command == "coarse":
        rows = build_coarse_rows(
            proposals,
            context_frame_count=args.context_frame_count,
            boundary_candidate_count=args.boundary_candidate_count,
            inside_frame_count=args.inside_frame_count,
            transition_gap_frames=args.transition_gap_frames,
            option_seed=args.option_seed,
            task_mode=args.task_mode,
            context_ratio=args.context_ratio,
            stride_multiplier=args.stride_multiplier,
            minimum_margin=args.minimum_margin,
            maximum_margin=args.maximum_margin,
        )
    else:
        rows = build_fine_rows(
            proposals,
            read_jsonl(args.coarse_manifest),
            read_jsonl(args.coarse_predictions),
            candidate_count=args.candidate_count,
            maximum_step=args.maximum_step,
            minimum_step=args.minimum_step,
            transition_gap_frames=args.transition_gap_frames,
            option_seed=args.option_seed,
        )

    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("query IDs are not unique")
    write_jsonl(args.output, rows)
    metadata = {
        "query_schema_version": QUERY_SCHEMA_VERSION,
        "stage": args.command,
        "rows": len(rows),
        "source_proposals": str(Path(args.proposals).resolve()),
        "source_proposals_sha256": file_sha256(args.proposals),
        "selection_gt_use": "none",
    }
    if args.command == "coarse":
        metadata["task_mode"] = args.task_mode
        metadata["parameters"] = {
            "context_frame_count": args.context_frame_count,
            "boundary_candidate_count_per_side": args.boundary_candidate_count,
            "inside_frame_count": args.inside_frame_count,
            "transition_gap_frames": args.transition_gap_frames,
            "option_seed": args.option_seed,
            "context_ratio": args.context_ratio,
            "stride_multiplier": args.stride_multiplier,
            "minimum_margin": args.minimum_margin,
            "maximum_margin": args.maximum_margin,
        }
    else:
        metadata.update(
            {
                "coarse_manifest": str(Path(args.coarse_manifest).resolve()),
                "coarse_manifest_sha256": file_sha256(args.coarse_manifest),
                "coarse_predictions": str(
                    Path(args.coarse_predictions).resolve()
                ),
                "coarse_predictions_sha256": file_sha256(
                    args.coarse_predictions
                ),
                "parameters": {
                    "candidate_count": args.candidate_count,
                    "maximum_step": args.maximum_step,
                    "minimum_step": args.minimum_step,
                    "transition_gap_frames": args.transition_gap_frames,
                    "option_seed": args.option_seed,
                },
            }
        )
    Path(str(args.output) + ".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
