"""Build shortcut-controlled counterfactual windows for the RQ2 temporal probe."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS, file_sha256
from tools.run_clip_target_adapter import LABELS


POSITIONS = (0.25, 0.50, 0.75)
WINDOW_LENGTHS = (8.0, 16.0, 32.0)


def stable_index(value, size, seed):
    payload = "{}|{}".format(seed, value).encode("utf-8")
    number = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return number % size


def overlaps(first, second):
    return max(float(first[0]), float(second[0])) < min(float(first[1]), float(second[1]))


def load_role_memberships(repo_root):
    memberships = defaultdict(list)
    config_hashes = {}
    role_keys = (
        ("source_train", "src_train_split"),
        ("source_val", "src_val_split"),
        ("target_val", "tgt_val_split"),
    )
    for shift, relative_path in SHIFT_CONFIGS.items():
        path = Path(repo_root) / relative_path
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_hashes[shift] = file_sha256(path)
        for role, key in role_keys:
            for subset in config[key]:
                memberships[str(subset).lower()].append({"shift": shift, "role": role})
    return memberships, config_hashes


def windows_for(action, video_duration, window_length):
    start, end = map(float, action["segment"])
    center = 0.5 * (start + end)
    windows = []
    for position in POSITIONS:
        window_start = center - position * window_length
        window_end = window_start + window_length
        if window_start < 0.0 or window_end > video_duration:
            return None
        if start < window_start or end > window_end:
            return None
        windows.append((window_start, window_end))
    return windows


def contains_second_same_class(annotations, current_index, label_id, windows):
    for other_index, other in enumerate(annotations):
        if other_index == current_index or int(other["label_id"]) != label_id:
            continue
        if any(overlaps(other["segment"], window) for window in windows):
            return True
    return False


def sampled_timestamps(window_start, window_end, fps, max_frames):
    duration = window_end - window_start
    frame_count = min(max_frames, max(2, int(round(duration * fps))))
    step = duration / frame_count
    return [window_start + (index + 0.5) * step for index in range(frame_count)]


def build_rows(annotation_path, repo_root, fps=2.0, max_frames=64, seed=20260919):
    database = json.loads(Path(annotation_path).read_text(encoding="utf-8"))["database"]
    memberships, config_hashes = load_role_memberships(repo_root)
    rows = []
    rejected = Counter()
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
            if not 0.0 <= start < end <= duration:
                rejected["invalid_segment"] += 1
                continue
            feasible = []
            for window_length in WINDOW_LENGTHS:
                windows = windows_for(action, duration, window_length)
                if windows is None:
                    continue
                if contains_second_same_class(
                    annotations, annotation_index, label_id, windows
                ):
                    continue
                feasible.append((window_length, windows))
            if not feasible:
                rejected["no_clean_counterfactual_triplet"] += 1
                continue
            instance_id = "rq2p-{}-{:05d}".format(video_id, annotation_index)
            window_length, windows = feasible[
                stable_index(instance_id, len(feasible), seed)
            ]
            for position, (window_start, window_end) in zip(POSITIONS, windows):
                timestamps = sampled_timestamps(
                    window_start, window_end, fps=fps, max_frames=max_frames
                )
                tag = "p{:02d}".format(int(round(position * 100)))
                rows.append({
                    "candidate_id": "{}-{}".format(instance_id, tag),
                    "instance_id": instance_id,
                    "context_position": position,
                    "video_id": video_id,
                    "annotation_index": annotation_index,
                    "subset": str(video["subset"]).lower(),
                    "gt_label_id": label_id,
                    "gt_label": LABELS[label_id],
                    "gt_start_sec": start,
                    "gt_end_sec": end,
                    "proposal_start_sec": window_start,
                    "proposal_end_sec": window_end,
                    "window_length_sec": window_length,
                    "sampling_fps": fps,
                    "frame_count": len(timestamps),
                    "frame_timestamps_sec": timestamps,
                    "memberships": sorted(
                        roles, key=lambda item: (item["shift"], item["role"])
                    ),
                })
    return rows, config_hashes, rejected


def summarize(rows, annotations, config_hashes, rejected, fps, max_frames, seed):
    role_counts = Counter()
    class_counts = Counter()
    for row in rows:
        class_counts[row["gt_label"]] += 1
        for role in row["memberships"]:
            role_counts[(role["shift"], role["role"])] += 1
    return {
        "protocol": "rq2-source-probe-counterfactual-context-v1",
        "seed": seed,
        "positions": list(POSITIONS),
        "window_lengths_sec": list(WINDOW_LENGTHS),
        "sampling_fps": fps,
        "max_frames": max_frames,
        "target_gt_role": "offline_evaluation_only",
        "unique_instances": len({row["instance_id"] for row in rows}),
        "unique_windows": len(rows),
        "unique_videos": len({row["video_id"] for row in rows}),
        "per_class_windows": dict(sorted(class_counts.items())),
        "per_shift_role_windows": {
            "{}_{}".format(shift, role): role_counts[(shift, role)]
            for shift in SHIFT_CONFIGS
            for role in ("source_train", "source_val", "target_val")
        },
        "rejected_instances": dict(sorted(rejected.items())),
        "annotations_sha256": file_sha256(annotations),
        "config_sha256": config_hashes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--annotations",
        default="data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0 or args.max_frames <= 1:
        raise ValueError("fps must be positive and max_frames must exceed one")
    root = Path(args.repo_root).resolve()
    annotations = Path(args.annotations)
    if not annotations.is_absolute():
        annotations = root / annotations
    rows, config_hashes, rejected = build_rows(
        annotations, root, fps=args.fps, max_frames=args.max_frames, seed=args.seed
    )
    if not rows:
        raise RuntimeError("RQ2 probe manifest is empty")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    protocol = summarize(
        rows, annotations, config_hashes, rejected, args.fps, args.max_frames, args.seed
    )
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
