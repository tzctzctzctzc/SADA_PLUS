"""Select a deterministic, class-balanced source-train pilot from an RQ2 manifest."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from tools.run_clip_target_adapter import LABELS, read_jsonl


def score(instance_id, seed):
    value = "{}|{}".format(seed, instance_id).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def has_role(row, shift, role):
    return any(
        item["shift"] == shift and item["role"] == role
        for item in row["memberships"]
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shift", default="S1")
    parser.add_argument("--role", default="source_train")
    parser.add_argument("--instances-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.instances_per_class <= 0:
        raise ValueError("instances-per-class must be positive")
    rows = [
        row for row in read_jsonl(args.manifest)
        if has_role(row, args.shift, args.role)
    ]
    instances = defaultdict(dict)
    for row in rows:
        instances[int(row["gt_label_id"])].setdefault(row["instance_id"], []).append(row)
    selected = []
    counts = {}
    for label_id, label in enumerate(LABELS):
        candidates = sorted(
            instances[label_id].items(), key=lambda item: score(item[0], args.seed)
        )
        chosen = []
        used_videos = set()
        for candidate in candidates:
            video_id = candidate[1][0]["video_id"]
            if video_id in used_videos:
                continue
            chosen.append(candidate)
            used_videos.add(video_id)
            if len(chosen) == args.instances_per_class:
                break
        if len(chosen) < args.instances_per_class:
            chosen_ids = {item[0] for item in chosen}
            chosen.extend(
                item for item in candidates
                if item[0] not in chosen_ids
            )
            chosen = chosen[: args.instances_per_class]
        if len(chosen) < args.instances_per_class:
            raise RuntimeError(
                "{} has only {} eligible instances".format(label, len(chosen))
            )
        counts[label] = len(chosen)
        for _, triplet in chosen:
            if len(triplet) != 3:
                raise RuntimeError("counterfactual instance does not have three windows")
            selected.extend(sorted(triplet, key=lambda row: row["context_position"]))
    selected.sort(key=lambda row: row["candidate_id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    protocol = {
        "protocol": "rq2-source-probe-balanced-pilot-v1",
        "parent_manifest": str(Path(args.manifest).resolve()),
        "shift": args.shift,
        "role": args.role,
        "instances_per_class": args.instances_per_class,
        "seed": args.seed,
        "per_class_instances": counts,
        "unique_instances": len({row["instance_id"] for row in selected}),
        "unique_windows": len(selected),
        "unique_videos": len({row["video_id"] for row in selected}),
    }
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
