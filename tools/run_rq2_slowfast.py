"""Source-only SlowFast frame-wise semantic baseline for RQ2."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.run_clip_target_adapter import LABELS, read_jsonl
from tools.run_rq1_slowfast_probe import segment_feature
from tools.run_rq2_clip import localize_peak


def timestamp_features(features, timestamps, fps=30.0, feat_stride=16.0, num_frames=32.0):
    rows = []
    for timestamp in timestamps:
        position = (float(timestamp) * fps - 0.5 * num_frames) / feat_stride
        index = min(max(int(round(position)), 0), len(features) - 1)
        vector = features[index].astype(np.float32)
        rows.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    return np.stack(rows)


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

    root = Path(args.repo_root).resolve()
    annotation_path = root / "data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json"
    feature_root = root / "data/ek100_slowfast/features"
    database = json.loads(annotation_path.read_text(encoding="utf-8"))["database"]
    manifest = read_jsonl(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for shift, config_path in SHIFT_CONFIGS.items():
            config = yaml.safe_load((root / config_path).read_text(encoding="utf-8"))
            source_subsets = {str(value).lower() for value in config["src_train_split"]}
            train_features = []
            train_labels = []
            for video_id, video in database.items():
                if str(video["subset"]).lower() not in source_subsets:
                    continue
                selected = [
                    action for action in video.get("annotations", [])
                    if int(action["label_id"]) < len(LABELS)
                ]
                if not selected:
                    continue
                with np.load(feature_root / (video_id + ".npz")) as data:
                    features = data["feats"]
                    for action in selected:
                        train_features.append(segment_feature(features, action["segment"]))
                        train_labels.append(int(action["label_id"]))
            classifier = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=args.max_iter,
                random_state=args.seed, solver="lbfgs",
            ).fit(np.stack(train_features), np.asarray(train_labels))
            if list(classifier.classes_) != list(range(len(LABELS))):
                raise RuntimeError("linear probe did not observe all ten classes")
            shift_rows = [
                row for row in manifest if any(role["shift"] == shift for role in row["memberships"])
            ]
            grouped = defaultdict(list)
            for row in shift_rows:
                grouped[row["video_id"]].append(row)
            for video_id in sorted(grouped):
                with np.load(feature_root / (video_id + ".npz")) as data:
                    features = data["feats"]
                    for row in grouped[video_id]:
                        vectors = timestamp_features(features, row["frame_timestamps_sec"])
                        probabilities = classifier.predict_proba(vectors)
                        scores = probabilities[:, int(row["gt_label_id"])].tolist()
                        pred_start, pred_end, threshold = localize_peak(scores)
                        stream.write(json.dumps({
                            "candidate_id": row["candidate_id"], "video_id": video_id,
                            "shift": shift, "status": "ok",
                            "model": "slowfast-source-linear-framewise",
                            "source_supervision": True, "frame_count": row["frame_count"],
                            "class_scores": scores, "relative_threshold": threshold,
                            "pred_start_position": pred_start,
                            "pred_end_position": pred_end,
                        }, sort_keys=True) + "\n")
            print("{} train={} evaluate={}".format(
                shift, len(train_labels), len(shift_rows)
            ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
