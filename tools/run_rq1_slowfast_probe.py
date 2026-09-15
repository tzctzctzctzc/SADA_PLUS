"""Train a source-only linear probe on frozen SlowFast segment features for RQ1."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_clip_target_adapter import LABELS, read_jsonl


def segment_feature(features, segment, fps=30.0, feat_stride=16.0, num_frames=32.0):
    start = (float(segment[0]) * fps - 0.5 * num_frames) / feat_stride
    end = (float(segment[1]) * fps - 0.5 * num_frames) / feat_stride
    left = max(0, int(math.ceil(start)))
    right = min(len(features), int(math.floor(end)) + 1)
    if right <= left:
        center = int(round(0.5 * (start + end)))
        left = min(max(center, 0), len(features) - 1)
        right = left + 1
    pooled = features[left:right].astype(np.float32).mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    return pooled / max(norm, 1e-12)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_iter <= 0:
        raise ValueError("max iterations must be positive")
    from sklearn.linear_model import LogisticRegression

    repo_root = Path(args.repo_root).resolve()
    annotation_path = repo_root / "data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json"
    feature_root = repo_root / "data/ek100_slowfast/features"
    database = json.loads(annotation_path.read_text(encoding="utf-8"))["database"]
    val_rows = read_jsonl(args.manifest)
    val_by_key = {(row["video_id"], int(row["annotation_index"])): row for row in val_rows}

    configs = {}
    source_train_subsets = set()
    for shift, relative_path in SHIFT_CONFIGS.items():
        config = yaml.safe_load((repo_root / relative_path).read_text(encoding="utf-8"))
        configs[shift] = config
        source_train_subsets.update(str(value).lower() for value in config["src_train_split"])

    needed = defaultdict(list)
    for video_id, video in database.items():
        subset = str(video["subset"]).lower()
        for annotation_index, action in enumerate(video.get("annotations", [])):
            if int(action["label_id"]) >= len(LABELS):
                continue
            key = (video_id, annotation_index)
            if subset in source_train_subsets or key in val_by_key:
                needed[video_id].append((annotation_index, action))

    pooled = {}
    for number, video_id in enumerate(sorted(needed), 1):
        feature_path = feature_root / (video_id + ".npz")
        with np.load(feature_path) as data:
            features = data["feats"]
        for annotation_index, action in needed[video_id]:
            pooled[(video_id, annotation_index)] = segment_feature(
                features, action["segment"]
            )
        print("features {}/{} {}".format(number, len(needed), video_id), flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for shift, config in configs.items():
            source_subsets = {str(value).lower() for value in config["src_train_split"]}
            train_keys = []
            train_labels = []
            for video_id, video in database.items():
                if str(video["subset"]).lower() not in source_subsets:
                    continue
                for annotation_index, action in enumerate(video.get("annotations", [])):
                    label = int(action["label_id"])
                    if label < len(LABELS):
                        train_keys.append((video_id, annotation_index))
                        train_labels.append(label)
            x_train = np.stack([pooled[key] for key in train_keys])
            y_train = np.asarray(train_labels, dtype=np.int64)
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=args.max_iter,
                random_state=args.seed,
                solver="lbfgs",
            )
            classifier.fit(x_train, y_train)
            if list(classifier.classes_) != list(range(len(LABELS))):
                raise RuntimeError("linear probe did not observe all ten classes")
            shift_rows = [
                row for row in val_rows
                if any(role["shift"] == shift for role in row["memberships"])
            ]
            probabilities = classifier.predict_proba(
                np.stack([
                    pooled[(row["video_id"], int(row["annotation_index"]))]
                    for row in shift_rows
                ])
            )
            for row, probs in zip(shift_rows, probabilities):
                order = np.argsort(probs)[::-1]
                top1, top2 = map(int, order[:2])
                stream.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "video_id": row["video_id"],
                    "shift": shift,
                    "status": "ok",
                    "model": "slowfast-source-linear",
                    "source_supervision": True,
                    "class_id": top1,
                    "prediction": LABELS[top1],
                    "confidence": float(probs[top1]),
                    "margin": float(probs[top1] - probs[top2]),
                    "class_probabilities": probs.tolist(),
                }, sort_keys=True) + "\n")
            print(
                "{} train={} evaluate={}".format(shift, len(train_keys), len(shift_rows)),
                flush=True,
            )
    Path(str(output) + ".protocol.json").write_text(json.dumps({
        "model": "slowfast-source-linear",
        "feature_pooling": "L2-normalized mean over GT segment feature grids",
        "source_supervision_only": True,
        "target_labels_used_for_training": False,
        "classifier": {
            "type": "sklearn.linear_model.LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
