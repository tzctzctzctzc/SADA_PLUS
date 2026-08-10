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
    centered_candidates,
    file_sha256,
    fine_step_seconds,
    json_sha256,
    read_jsonl,
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


def chronological_frames(window, count, video_duration=None):
    if int(count) < 2:
        raise ValueError("frame count must be at least two")
    right = float(window[1])
    if video_duration is not None:
        right = min(right, max(0.0, float(video_duration) - 0.05))
    timestamps = np.linspace(float(window[0]), right, int(count))
    return [
        {"frame_id": "F{}".format(index + 1), "timestamp_sec": float(timestamp)}
        for index, timestamp in enumerate(timestamps)
    ]


def build_coarse_rows(proposals, frame_count=16, **window_options):
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
            "query_id": proposal["proposal_id"] + ":coarse",
            "proposal_id": proposal["proposal_id"],
            "video_id": proposal["video_id"],
            "video_duration_seconds": float(proposal["video_duration_seconds"]),
            "predicted_class_id": int(proposal["predicted_class_id"]),
            "predicted_class_name": proposal["predicted_class_name"],
            "coarse_segment_seconds": list(map(float, proposal["coarse_segment_seconds"])),
            "context_window_seconds": window,
            "frames": chronological_frames(
                window, frame_count, proposal["video_duration_seconds"]
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


def frame_time(query, frame_id):
    matches = [
        float(frame["timestamp_sec"])
        for frame in query["frames"]
        if frame["frame_id"] == frame_id
    ]
    if len(matches) != 1:
        raise ValueError("unknown or duplicated frame_id {}".format(frame_id))
    return matches[0]


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
        if not decision or decision.get("status") != "VALID_ACTION":
            continue
        proposal = proposal_map[query["proposal_id"]]
        step = fine_step_seconds(
            proposal["origin_stride_seconds"], maximum_step, minimum_step
        )
        for side in ("start", "end"):
            rough_key = "rough_{}_frame".format(side)
            center = frame_time(query, decision.get(rough_key))
            safe_duration = max(
                0.0, float(proposal["video_duration_seconds"]) - 0.05
            )
            timestamps = centered_candidates(
                center,
                step,
                int(candidate_count),
                safe_duration,
            )
            fine_query_id = proposal["proposal_id"] + ":fine:" + side
            options = option_permutation(
                fine_query_id, side, len(timestamps), int(option_seed)
            )
            candidates = [
                {
                    "frame_id": "F{}".format(index + 1),
                    "timestamp_sec": float(timestamp),
                    "option": options[index],
                }
                for index, timestamp in enumerate(timestamps)
            ]
            row = {
                "query_schema_version": QUERY_SCHEMA_VERSION,
                "stage": "fine",
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
                "rough_boundary_frame": decision[rough_key],
                "rough_boundary_seconds": center,
                "fine_step_seconds": step,
                "frames": candidates,
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
    coarse.add_argument("--frame-count", type=int, default=16)
    coarse.add_argument("--context-ratio", type=float, default=0.5)
    coarse.add_argument("--stride-multiplier", type=float, default=2.0)
    coarse.add_argument("--minimum-margin", type=float, default=1.0)
    coarse.add_argument("--maximum-margin", type=float, default=6.0)

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
            frame_count=args.frame_count,
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
        metadata["parameters"] = {
            "frame_count": args.frame_count,
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
