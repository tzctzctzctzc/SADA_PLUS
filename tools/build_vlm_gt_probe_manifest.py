"""Build a GT-balanced capability probe for the VLM actionness prompt.

This is not a UDA result: GT deliberately constructs a balanced diagnostic
set containing target actions, non-target actions, and idle background.
"""

import argparse
import json
import random
from pathlib import Path


def video_ids_in(root):
    return {
        path.stem
        for path in Path(root).iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    }


def midpoint(segment):
    return 0.5 * (float(segment[0]) + float(segment[1]))


def diverse_take(rows, count, rng):
    rng.shuffle(rows)
    chosen, used_videos = [], set()
    for row in rows:
        if row["video_id"] in used_videos:
            continue
        chosen.append(row)
        used_videos.add(row["video_id"])
        if len(chosen) == count:
            return chosen
    for row in rows:
        if row not in chosen:
            chosen.append(row)
        if len(chosen) == count:
            return chosen
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-group", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--idle-margin-sec", type=float, default=1.0)
    parser.add_argument(
        "--candidate-label-id",
        type=int,
        help="Build a candidate-specific verifier probe instead of the ten-class probe.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    database = json.loads(Path(args.annotations).read_text(encoding="utf-8"))["database"]
    available = video_ids_in(args.video_root)
    if args.candidate_label_id is None:
        groups = {"target_action": [], "other_action": [], "idle_background": []}
    else:
        groups = {
            "candidate_action": [],
            "other_target_action": [],
            "other_action": [],
            "idle_background": [],
        }

    for video_id in sorted(available):
        video = database.get(video_id)
        if not video:
            continue
        annotations = video.get("annotations", [])
        for index, action in enumerate(annotations):
            label = int(action["label_id"])
            center = midpoint(action["segment"])
            if args.candidate_label_id is None:
                group = "target_action" if label < 10 else "other_action"
                overlaps_excluded_label = group == "other_action" and any(
                    int(other["label_id"]) < 10
                    and float(other["segment"][0]) <= center <= float(other["segment"][1])
                    for other in annotations
                )
            else:
                if label == args.candidate_label_id:
                    group = "candidate_action"
                elif label < 10:
                    group = "other_target_action"
                else:
                    group = "other_action"
                excluded_ids = (
                    {args.candidate_label_id}
                    if group == "other_target_action"
                    else set(range(10)) if group == "other_action" else set()
                )
                overlaps_excluded_label = any(
                    int(other["label_id"]) in excluded_ids
                    and float(other["segment"][0]) <= center <= float(other["segment"][1])
                    for other in annotations
                )
            if overlaps_excluded_label:
                continue
            groups[group].append(
                {
                    "candidate_id": "probe-{}-{}-{:05d}".format(group, video_id, index),
                    "video_id": video_id,
                    "center_sec": center,
                    "gt_is_foreground": group
                    in {"target_action", "candidate_action", "other_target_action"},
                    "gt_matches_candidate": group == "candidate_action",
                    "probe_group": group,
                    "gt_label_id": label,
                    "target_label_id": label if label < 10 else None,
                    "candidate_label_id": args.candidate_label_id,
                }
            )

        duration = float(video["duration"])
        for trial in range(1000):
            if len([row for row in groups["idle_background"] if row["video_id"] == video_id]) >= 2:
                break
            center = rng.uniform(1.0, max(1.0, duration - 1.0))
            is_idle = all(
                center < float(action["segment"][0]) - args.idle_margin_sec
                or center > float(action["segment"][1]) + args.idle_margin_sec
                for action in annotations
            )
            if is_idle:
                groups["idle_background"].append(
                    {
                        "candidate_id": "probe-idle_background-{}-{:05d}".format(
                            video_id, trial
                        ),
                        "video_id": video_id,
                        "center_sec": center,
                        "gt_is_foreground": False,
                        "probe_group": "idle_background",
                        "target_label_id": None,
                    }
                )

    selected = []
    for name, rows in groups.items():
        chosen = diverse_take(rows, args.per_group, rng)
        if len(chosen) < args.per_group:
            raise RuntimeError("not enough {} examples: {}".format(name, len(chosen)))
        selected.extend(chosen)
    rng.shuffle(selected)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row) + "\n")
    Path(str(output) + ".probe.json").write_text(
        json.dumps(
            {
                "audit_scope": "GT-balanced prompt capability probe only",
                "valid_uda_result": False,
                "selection_uses_target_gt": True,
                "groups": {name: args.per_group for name in groups},
                "candidate_label_id": args.candidate_label_id,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"samples": len(selected), "groups": list(groups)}, indent=2))


if __name__ == "__main__":
    main()
