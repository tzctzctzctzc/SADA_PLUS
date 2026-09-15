"""Build the deduplicated GT-segment manifest for RQ1 semantic robustness.

Target annotations are used only to construct this offline diagnostic benchmark.
They must never be consumed by UDA training, prompt selection, or checkpoint selection.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tools.run_clip_target_adapter import LABELS


SHIFT_CONFIGS = {
    "S1": "configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml",
    "S2": "configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml",
    "S3": "configs/white_vs_all_new_videos_full/version_final_ckpt.yaml",
    "S4": "configs/white_vs_all_old_videos_full/version_final_ckpt.yaml",
    "S5": "configs/old_vs_new/version_final_ckpt.yaml",
    "S6": "configs/new_vs_old/version_paper_s6_dim512.yaml",
}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_memberships(repo_root):
    memberships = defaultdict(list)
    config_hashes = {}
    for shift, relative_path in SHIFT_CONFIGS.items():
        path = Path(repo_root) / relative_path
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config_hashes[shift] = file_sha256(path)
        for domain, key in (("source", "src_val_split"), ("target", "tgt_val_split")):
            for subset in config[key]:
                memberships[str(subset).lower()].append({
                    "shift": shift,
                    "domain": domain,
                })
    return memberships, config_hashes


def midpoint_labels(annotations, current_index):
    current = annotations[current_index]
    center = 0.5 * (float(current["segment"][0]) + float(current["segment"][1]))
    labels = {
        int(action["label_id"])
        for action in annotations
        if int(action["label_id"]) < len(LABELS)
        and float(action["segment"][0]) <= center <= float(action["segment"][1])
    }
    labels.add(int(current["label_id"]))
    return sorted(labels)


def build_rows(annotation_path, repo_root):
    database = json.loads(Path(annotation_path).read_text(encoding="utf-8"))["database"]
    memberships, config_hashes = load_memberships(repo_root)
    rows = []
    for video_id in sorted(database):
        video = database[video_id]
        subset = str(video["subset"]).lower()
        roles = memberships.get(subset, [])
        if not roles:
            continue
        annotations = video.get("annotations", [])
        for annotation_index, action in enumerate(annotations):
            label_id = int(action["label_id"])
            if label_id >= len(LABELS):
                continue
            start, end = map(float, action["segment"])
            if not 0.0 <= start < end:
                raise ValueError("invalid segment {} {}".format(video_id, action["segment"]))
            acceptable = midpoint_labels(annotations, annotation_index)
            rows.append({
                "candidate_id": "rq1-{}-{:05d}".format(video_id, annotation_index),
                "video_id": video_id,
                "annotation_index": annotation_index,
                "absolute_feature_index": annotation_index,
                "proposal_start_sec": start,
                "proposal_end_sec": end,
                "duration_sec": end - start,
                "subset": subset,
                "gt_label_id": label_id,
                "gt_label": LABELS[label_id],
                "gt_acceptable_class_ids": acceptable,
                "is_temporally_ambiguous": len(acceptable) > 1,
                "memberships": sorted(roles, key=lambda item: (item["shift"], item["domain"])),
            })
    return rows, config_hashes


def write_manifest(rows, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def summarize(rows, annotation_path, config_hashes, frame_count):
    per_split = Counter()
    per_class = Counter()
    for row in rows:
        per_class[row["gt_label"]] += 1
        for role in row["memberships"]:
            per_split[(role["shift"], role["domain"])] += 1
    return {
        "protocol": "rq1-semantic-robustness-gt-segments-v1",
        "target_gt_role": "offline_evaluation_only",
        "target_gt_forbidden_uses": [
            "uda_training", "prompt_selection", "threshold_selection", "checkpoint_selection"
        ],
        "frame_count": int(frame_count),
        "classes": list(LABELS),
        "unique_videos": len({row["video_id"] for row in rows}),
        "unique_segments": len(rows),
        "ambiguous_midpoint_segments": sum(row["is_temporally_ambiguous"] for row in rows),
        "per_class": dict(sorted(per_class.items())),
        "per_shift_domain": {
            "{}_{}".format(shift, domain): per_split[(shift, domain)]
            for shift in SHIFT_CONFIGS
            for domain in ("source", "target")
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
        raise RuntimeError("RQ1 manifest is empty")
    write_manifest(rows, args.output)
    protocol = summarize(rows, annotation_path, config_hashes, args.frame_count)
    Path(str(args.output) + ".protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
