"""Build the union of source-train and validation GT segments for RQ1 probes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS, file_sha256
from tools.run_clip_target_adapter import LABELS


def load_split_memberships(repo_root):
    memberships = defaultdict(list)
    config_hashes = {}
    for shift, relative_path in SHIFT_CONFIGS.items():
        path = Path(repo_root) / relative_path
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_hashes[shift] = file_sha256(path)
        split_keys = (
            ("source_train", "source", "train", "src_train_split"),
            ("source_val", "source", "val", "src_val_split"),
            ("target_val", "target", "val", "tgt_val_split"),
        )
        for role, domain, split, key in split_keys:
            for subset in config[key]:
                memberships[str(subset).lower()].append({
                    "shift": shift,
                    "role": role,
                    "domain": domain,
                    "split": split,
                })
    return memberships, config_hashes


def build_rows(annotation_path, repo_root):
    database = json.loads(Path(annotation_path).read_text(encoding="utf-8"))["database"]
    memberships, config_hashes = load_split_memberships(repo_root)
    rows = []
    for video_id in sorted(database):
        video = database[video_id]
        subset = str(video["subset"]).lower()
        roles = memberships.get(subset, [])
        if not roles:
            continue
        for annotation_index, action in enumerate(video.get("annotations", [])):
            label_id = int(action["label_id"])
            if label_id >= len(LABELS):
                continue
            start, end = map(float, action["segment"])
            if not 0.0 <= start < end:
                raise ValueError("invalid segment {} {}".format(video_id, action["segment"]))
            rows.append({
                "candidate_id": "rq1-{}-{:05d}".format(video_id, annotation_index),
                "video_id": video_id,
                "annotation_index": annotation_index,
                "proposal_start_sec": start,
                "proposal_end_sec": end,
                "duration_sec": end - start,
                "subset": subset,
                "gt_label_id": label_id,
                "gt_label": LABELS[label_id],
                "memberships": sorted(
                    roles, key=lambda item: (item["shift"], item["role"])
                ),
            })
    return rows, config_hashes


def summarize(rows, annotation_path, config_hashes, frame_count):
    counts = Counter()
    for row in rows:
        for role in row["memberships"]:
            counts[(role["shift"], role["role"])] += 1
    return {
        "protocol": "rq1-source-only-frozen-probe-features-v1",
        "frame_count": int(frame_count),
        "classes": list(LABELS),
        "unique_videos": len({row["video_id"] for row in rows}),
        "unique_segments": len(rows),
        "per_shift_role": {
            "{}_{}".format(shift, role): counts[(shift, role)]
            for shift in SHIFT_CONFIGS
            for role in ("source_train", "source_val", "target_val")
        },
        "supervision_policy": {
            "classifier_fit": "source_train labels only",
            "source_val_labels": "evaluation only",
            "target_val_labels": "evaluation only",
        },
        "annotations_sha256": file_sha256(annotation_path),
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
    parser.add_argument("--frame-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0:
        raise ValueError("frame count must be positive")
    repo_root = Path(args.repo_root).resolve()
    annotation_path = Path(args.annotations)
    if not annotation_path.is_absolute():
        annotation_path = repo_root / annotation_path
    rows, config_hashes = build_rows(annotation_path, repo_root)
    if not rows:
        raise RuntimeError("RQ1 source-probe manifest is empty")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    protocol = summarize(rows, annotation_path, config_hashes, args.frame_count)
    Path(str(output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
