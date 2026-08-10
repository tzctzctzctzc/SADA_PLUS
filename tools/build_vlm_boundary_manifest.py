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
    transition_grid,
    write_jsonl,
)


def dynamic_context_window(
    segment,
    duration,
    origin_stride_seconds,
    context_ratio=0.5,
    stride_multiplier=2.0,
    minimum_margin=1.0,
    maximum_margin=6.0,
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


def adaptive_coarse_frames(
    segment,
    window,
    origin_stride_seconds,
    video_duration,
    context_count=6,
    boundary_count=5,
    inside_count=4,
):
    """Cover context, proposal interior, and both boundary neighborhoods."""
    if min(int(context_count), int(boundary_count), int(inside_count)) < 2:
        raise ValueError("all coarse sampling counts must be at least two")
    left, right = map(float, segment)
    safe_duration = max(0.0, float(video_duration) - 0.05)
    radius = max(0.15, float(origin_stride_seconds) / 2.0)
    groups = (
        np.linspace(float(window[0]), float(window[1]), int(context_count)),
        np.linspace(left - radius, left + radius, int(boundary_count)),
        np.linspace(left, right, int(inside_count)),
        np.linspace(right - radius, right + radius, int(boundary_count)),
    )
    timestamps = sorted(
        {
            round(min(max(float(timestamp), 0.0), safe_duration), 6)
            for group in groups
            for timestamp in group
        }
    )
    return [
        {"frame_id": "F{}".format(index + 1), "timestamp_sec": float(timestamp)}
        for index, timestamp in enumerate(timestamps)
    ]


def build_coarse_rows(
    proposals,
    context_frame_count=6,
    boundary_frame_count=5,
    inside_frame_count=4,
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
        row = {
            "query_schema_version": QUERY_SCHEMA_VERSION,
            "stage": "coarse",
            "task_mode": task_mode,
            "query_id": proposal["proposal_id"] + ":coarse",
            "proposal_id": proposal["proposal_id"],
            "video_id": proposal["video_id"],
            "video_duration_seconds": float(proposal["video_duration_seconds"]),
            "predicted_class_id": int(proposal["predicted_class_id"]),
            "predicted_class_name": proposal["predicted_class_name"],
            "coarse_segment_seconds": list(map(float, proposal["coarse_segment_seconds"])),
            "context_window_seconds": window,
            "sampling_protocol": "adaptive-context-boundary-interior-v2",
            "frames": adaptive_coarse_frames(
                proposal["coarse_segment_seconds"],
                window,
                proposal["origin_stride_seconds"],
                proposal["video_duration_seconds"],
                context_count=context_frame_count,
                boundary_count=boundary_frame_count,
                inside_count=inside_frame_count,
            ),
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


def transition_time(query, before_frame_id, after_frame_id):
    ids = [frame["frame_id"] for frame in query["frames"]]
    before_index = ids.index(before_frame_id)
    after_index = ids.index(after_frame_id)
    if after_index != before_index + 1:
        raise ValueError("boundary transition frames must be adjacent")
    before = float(query["frames"][before_index]["timestamp_sec"])
    after = float(query["frames"][after_index]["timestamp_sec"])
    return 0.5 * (before + after)


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
            if decision.get("{}_status".format(side)) != "VISIBLE":
                continue
            before_key = "rough_{}_before_frame".format(side)
            after_key = "rough_{}_after_frame".format(side)
            center = transition_time(
                query, decision.get(before_key), decision.get(after_key)
            )
            safe_duration = max(
                0.0, float(proposal["video_duration_seconds"]) - 0.05
            )
            frames = transition_grid(
                center, step, int(candidate_count), safe_duration
            )
            fine_query_id = proposal["proposal_id"] + ":fine:" + side
            options = option_permutation(
                fine_query_id, side, len(frames) - 1, int(option_seed)
            )
            candidates = [
                {
                    "option": options[index],
                    "before_frame_id": frames[index]["frame_id"],
                    "after_frame_id": frames[index + 1]["frame_id"],
                    "timestamp_sec": 0.5
                    * (
                        float(frames[index]["timestamp_sec"])
                        + float(frames[index + 1]["timestamp_sec"])
                    ),
                }
                for index in range(len(frames) - 1)
            ]
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
                "rough_boundary_transition": {
                    "before_frame_id": decision[before_key],
                    "after_frame_id": decision[after_key],
                },
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
    coarse.add_argument("--boundary-frame-count", type=int, default=5)
    coarse.add_argument("--inside-frame-count", type=int, default=4)
    coarse.add_argument("--context-ratio", type=float, default=0.5)
    coarse.add_argument("--stride-multiplier", type=float, default=2.0)
    coarse.add_argument("--minimum-margin", type=float, default=1.0)
    coarse.add_argument("--maximum-margin", type=float, default=6.0)
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
    fine.add_argument("--option-seed", type=int, default=20260810)
    args = parser.parse_args()

    proposals = read_jsonl(args.proposals)
    if args.command == "coarse":
        rows = build_coarse_rows(
            proposals,
            context_frame_count=args.context_frame_count,
            boundary_frame_count=args.boundary_frame_count,
            inside_frame_count=args.inside_frame_count,
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
            "boundary_frame_count_per_side": args.boundary_frame_count,
            "inside_frame_count": args.inside_frame_count,
            "maximum_frames_before_deduplication": (
                args.context_frame_count
                + 2 * args.boundary_frame_count
                + args.inside_frame_count
            ),
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
