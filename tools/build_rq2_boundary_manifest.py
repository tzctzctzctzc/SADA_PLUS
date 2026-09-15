"""Build deterministic context windows for the RQ2 boundary diagnostic."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from tools.build_rq1_semantic_manifest import load_memberships, midpoint_labels
from tools.run_clip_target_adapter import LABELS, proposal_timestamps


def context_ratios(candidate_id, seed=20260915, minimum=0.25, maximum=2.0):
    payload = "{}|{}".format(seed, candidate_id).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = random.Random(value)
    return rng.uniform(minimum, maximum), rng.uniform(minimum, maximum)


def overlaps(first, second):
    return max(float(first[0]), float(second[0])) < min(float(first[1]), float(second[1]))


def build_rows(annotation_path, repo_root, frame_count=16, seed=20260915,
               context_minimum=0.25, context_maximum=2.0):
    database = json.loads(Path(annotation_path).read_text(encoding="utf-8"))["database"]
    memberships, config_hashes = load_memberships(repo_root)
    rows = []
    for video_id in sorted(database):
        video = database[video_id]
        roles = memberships.get(str(video["subset"]).lower(), [])
        if not roles:
            continue
        duration = float(video["duration"])
        annotations = video.get("annotations", [])
        for annotation_index, action in enumerate(annotations):
            label_id = int(action["label_id"])
            if label_id >= len(LABELS):
                continue
            start, end = map(float, action["segment"])
            action_duration = end - start
            if not 0.0 <= start < end <= duration:
                raise ValueError("invalid segment {} {}".format(video_id, action["segment"]))
            candidate_id = "rq2-{}-{:05d}".format(video_id, annotation_index)
            left_ratio, right_ratio = context_ratios(
                candidate_id, seed, context_minimum, context_maximum
            )
            window_start = max(0.0, start - left_ratio * action_duration)
            window_end = min(duration, end + right_ratio * action_duration)
            window_duration = window_end - window_start
            acceptable = midpoint_labels(annotations, annotation_index)
            same_class_context = any(
                other_index != annotation_index
                and int(other["label_id"]) == label_id
                and overlaps(other["segment"], (window_start, window_end))
                for other_index, other in enumerate(annotations)
            )
            challenge = len(acceptable) > 1 or same_class_context
            row = {
                "candidate_id": candidate_id,
                "video_id": video_id,
                "annotation_index": annotation_index,
                "subset": str(video["subset"]).lower(),
                "gt_label_id": label_id,
                "gt_label": LABELS[label_id],
                "gt_start_sec": start,
                "gt_end_sec": end,
                "duration_sec": action_duration,
                "proposal_start_sec": window_start,
                "proposal_end_sec": window_end,
                "context_left_ratio": left_ratio,
                "context_right_ratio": right_ratio,
                "context_ratio_min": context_minimum,
                "context_ratio_max": context_maximum,
                "gt_start_position": frame_count * (start - window_start) / window_duration,
                "gt_end_position": frame_count * (end - window_start) / window_duration,
                "frame_count": frame_count,
                "frame_timestamps_sec": proposal_timestamps(
                    {"proposal_start_sec": window_start, "proposal_end_sec": window_end},
                    frame_count,
                ),
                "is_challenge": challenge,
                "challenge_reasons": [
                    reason for reason, enabled in (
                        ("overlapping_action_at_midpoint", len(acceptable) > 1),
                        ("same_class_instance_in_context", same_class_context),
                    ) if enabled
                ],
                "memberships": sorted(roles, key=lambda item: (item["shift"], item["domain"])),
            }
            rows.append(row)
    return rows, config_hashes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--annotations",
        default="data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--context-ratio-min", type=float, default=0.25)
    parser.add_argument("--context-ratio-max", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 1:
        raise ValueError("frame count must be greater than one")
    root = Path(args.repo_root).resolve()
    annotations = Path(args.annotations)
    if not annotations.is_absolute():
        annotations = root / annotations
    if not 0.0 <= args.context_ratio_min < args.context_ratio_max:
        raise ValueError("invalid context ratio range")
    rows, config_hashes = build_rows(
        annotations, root, args.frame_count, args.seed,
        args.context_ratio_min, args.context_ratio_max,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "protocol": "rq2-oracle-class-jittered-context-boundary-v2",
        "target_gt_role": "offline_evaluation_only",
        "frame_count": args.frame_count,
        "seed": args.seed,
        "context_ratio_range": [args.context_ratio_min, args.context_ratio_max],
        "unique_segments": len(rows),
        "unique_videos": len({row["video_id"] for row in rows}),
        "isolated_segments": sum(not row["is_challenge"] for row in rows),
        "challenge_segments": sum(row["is_challenge"] for row in rows),
        "challenge_reasons": dict(Counter(
            reason for row in rows for reason in row["challenge_reasons"]
        )),
        "config_sha256": config_hashes,
    }
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
