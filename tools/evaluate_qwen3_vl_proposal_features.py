"""Evaluate Qwen visual features before integrating relational supervision."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.run_clip_then_llava import read_jsonl, write_json


PROTOCOL_VERSION = "qwen-visual-relation-source-gate-v1"


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-8)


def summarize_frames(frames):
    frames = normalize(frames)
    if frames.shape[0] == 1:
        static = frames[0]
        transition = np.zeros_like(static)
        return static, transition, np.concatenate((static, transition))
    middle = frames.shape[0] // 2
    static = normalize(frames.mean(axis=0, keepdims=True))[0]
    transition = normalize(
        (frames[middle:].mean(axis=0) - frames[:middle].mean(axis=0))[None]
    )[0]
    return static, transition, np.concatenate((static, transition))


def load_feature_variants(index_path, feature_root):
    variants = defaultdict(dict)
    for row in read_jsonl(index_path):
        if row.get("status") != "ok":
            continue
        path = Path(feature_root) / row["feature_file"]
        arrays = np.load(path)
        stacks = [("final", arrays["final_frames"])]
        stacks.extend(
            ("deepstack_{}".format(index), frames)
            for index, frames in enumerate(arrays["deepstack_frames"])
        )
        for name, frames in stacks:
            static, transition, combined = summarize_frames(frames)
            variants[name + "_static"][row["candidate_id"]] = static
            variants[name + "_transition"][row["candidate_id"]] = transition
            variants[name + "_combined"][row["candidate_id"]] = combined
    return variants


def slowfast_variants(rows, feature_root, fps, stride, num_frames):
    variants = defaultdict(dict)
    videos = {}
    for row in rows:
        video_id = row["video_id"]
        if video_id not in videos:
            videos[video_id] = np.load(
                Path(feature_root) / (video_id + ".npz"), mmap_mode="r"
            )["feats"]
        features = videos[video_id]
        start = int(np.floor((row["proposal_start_sec"] * fps - 0.5 * num_frames) / stride))
        end = int(np.ceil((row["proposal_end_sec"] * fps - 0.5 * num_frames) / stride))
        start = max(0, min(start, len(features) - 1))
        end = max(start + 1, min(end, len(features)))
        static, transition, combined = summarize_frames(features[start:end])
        candidate_id = row["candidate_id"]
        variants["slowfast_static"][candidate_id] = static
        variants["slowfast_transition"][candidate_id] = transition
        variants["slowfast_combined"][candidate_id] = combined
    return variants


def group_linear_probe(features, labels, groups, folds):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    predictions = np.full(len(labels), -1, dtype=np.int64)
    for train, test in splitter.split(features, labels, groups):
        train_features = features[train]
        train_labels = labels[train]
        classes = np.unique(train_labels)
        targets = np.empty((len(train), len(classes)), dtype=np.float32)
        for column, class_id in enumerate(classes):
            positive = train_labels == class_id
            positive_count = max(int(positive.sum()), 1)
            negative_count = max(int((~positive).sum()), 1)
            targets[:, column] = np.where(
                positive,
                len(train) / (2.0 * positive_count),
                -len(train) / (2.0 * negative_count),
            )
        kernel = train_features @ train_features.T
        dual_weights = np.linalg.solve(
            kernel + np.eye(len(train), dtype=np.float32), targets
        )
        scores = features[test] @ train_features.T @ dual_weights
        predictions[test] = classes[scores.argmax(axis=1)]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_recall": float(balanced_accuracy_score(labels, predictions)),
        "count": int(len(labels)),
    }


def relation_auc(rows, feature_by_id, max_gap_seconds):
    from sklearn.metrics import roc_auc_score

    by_video = defaultdict(list)
    for row in rows:
        if row["candidate_id"] in feature_by_id and row.get("gt_assigned_class_ids"):
            by_video[row["video_id"]].append(row)
    labels, scores = [], []
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: row["absolute_center_time"])
        for left, right in zip(video_rows, video_rows[1:]):
            gap = right["absolute_center_time"] - left["absolute_center_time"]
            if gap > max_gap_seconds:
                continue
            left_labels = set(left["gt_assigned_class_ids"])
            right_labels = set(right["gt_assigned_class_ids"])
            labels.append(int(not left_labels.intersection(right_labels)))
            one = normalize(feature_by_id[left["candidate_id"]][None])[0]
            two = normalize(feature_by_id[right["candidate_id"]][None])[0]
            scores.append(float(1.0 - np.dot(one, two)))
    auc = None
    if len(set(labels)) == 2:
        auc = float(roc_auc_score(labels, scores))
    return {"auc": auc, "pair_count": len(labels), "change_count": sum(labels)}


def temporal_iou(segment, reference):
    intersection = max(
        0.0, min(segment[1], reference[1]) - max(segment[0], reference[0])
    )
    union = max(segment[1], reference[1]) - min(segment[0], reference[0])
    return intersection / union if union > 0.0 else 0.0


def proposal_quality_targets(rows, database, num_known_classes=10):
    """Attach GT IoU targets for source-only proposal-quality diagnostics."""
    output = []
    for row in rows:
        proposal = (
            float(row["proposal_start_sec"]), float(row["proposal_end_sec"])
        )
        known = []
        for annotation in database[row["video_id"]].get("annotations", []):
            label = int(annotation["label_id"])
            if label < int(num_known_classes):
                known.append((
                    tuple(float(value) for value in annotation["segment"]), label
                ))
        scored = [(temporal_iou(proposal, segment), segment, label)
                  for segment, label in known]
        best_any = max(scored, default=(0.0, None, None), key=lambda item: item[0])
        predicted_class = int(row["teacher_top1_class"])
        same_class = [item for item in scored if item[2] == predicted_class]
        best_class = max(
            same_class, default=(0.0, None, None), key=lambda item: item[0]
        )
        enriched = dict(row)
        enriched["quality_iou_any"] = float(best_any[0])
        enriched["quality_iou_class"] = float(best_class[0])
        output.append(enriched)
    return output


def _rank_metrics(targets, predictions, threshold=0.5):
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score, roc_auc_score

    targets = np.asarray(targets, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=np.float32)
    binary = targets >= float(threshold)
    auc = float(roc_auc_score(binary, predictions)) if len(set(binary)) == 2 else None
    ap = float(average_precision_score(binary, predictions)) if binary.any() else None
    correlation = spearmanr(targets, predictions).statistic
    return {
        "roc_auc_iou_0.5": auc,
        "average_precision_iou_0.5": ap,
        "spearman_iou": (
            float(correlation) if np.isfinite(correlation) else None
        ),
        "mae_iou": float(np.abs(targets - predictions).mean()),
        "positive_count_iou_0.5": int(binary.sum()),
        "count": int(len(targets)),
    }


def group_ridge_predictions(blocks, targets, groups, folds=5, alpha=1.0):
    """Group-CV linear-kernel ridge with fold-local feature normalization."""
    from sklearn.model_selection import GroupKFold

    targets = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(groups)
    block_items = [(name, np.asarray(values, dtype=np.float32))
                   for name, values in blocks.items()]
    if not block_items or any(len(values) != len(targets) for _, values in block_items):
        raise ValueError("quality feature blocks have inconsistent lengths")
    split_count = min(int(folds), len(set(groups)))
    if split_count < 2:
        raise ValueError("quality gate requires at least two source videos")
    predictions = np.zeros_like(targets)
    splitter = GroupKFold(n_splits=split_count)
    dummy = np.zeros((len(targets), 1), dtype=np.float32)
    for train, test in splitter.split(dummy, targets, groups):
        train_parts, test_parts = [], []
        for name, values in block_items:
            train_values, test_values = values[train], values[test]
            if name == "structure":
                mean = train_values.mean(axis=0, keepdims=True)
                std = train_values.std(axis=0, keepdims=True)
                train_values = (train_values - mean) / np.maximum(std, 1e-6)
                test_values = (test_values - mean) / np.maximum(std, 1e-6)
            else:
                train_values = normalize(train_values)
                test_values = normalize(test_values)
            train_parts.append(train_values)
            test_parts.append(test_values)
        train_features = np.concatenate(train_parts, axis=1)
        test_features = np.concatenate(test_parts, axis=1)
        center = float(targets[train].mean())
        kernel = train_features @ train_features.T
        dual = np.linalg.solve(
            kernel + float(alpha) * np.eye(len(train), dtype=np.float32),
            targets[train] - center,
        )
        predictions[test] = center + test_features @ train_features.T @ dual
    return predictions


def proposal_quality_gate(rows, qwen, slowfast, folds, min_auc_gain,
                          min_spearman_gain):
    candidates = [
        row for row in rows
        if row["candidate_id"] in slowfast["slowfast_combined"]
        and row["candidate_id"] in qwen["deepstack_0_combined"]
    ]
    groups = np.asarray([row["video_id"] for row in candidates])
    structure = np.asarray([
        [
            float(row["teacher_score"]),
            np.log1p(float(row["proposal_end_sec"]) - float(row["proposal_start_sec"])),
            *[float(int(row["source_fpn_levels"][0]) == level) for level in range(3)],
        ]
        for row in candidates
    ], dtype=np.float32)
    candidate_ids = [row["candidate_id"] for row in candidates]
    feature_sets = {
        "structure": {"structure": structure},
        "structure_slowfast": {
            "structure": structure,
            "slowfast": np.asarray([
                slowfast["slowfast_combined"][key] for key in candidate_ids
            ]),
        },
        "structure_qwen_deepstack0": {
            "structure": structure,
            "qwen": np.asarray([
                qwen["deepstack_0_combined"][key] for key in candidate_ids
            ]),
        },
        "structure_slowfast_qwen_deepstack0": {
            "structure": structure,
            "slowfast": np.asarray([
                slowfast["slowfast_combined"][key] for key in candidate_ids
            ]),
            "qwen": np.asarray([
                qwen["deepstack_0_combined"][key] for key in candidate_ids
            ]),
        },
    }
    report = {"count": len(candidates), "targets": {}}
    for target_name in ("quality_iou_any", "quality_iou_class"):
        targets = np.asarray([row[target_name] for row in candidates], dtype=np.float32)
        methods = {
            "teacher_score": _rank_metrics(
                targets, [row["teacher_score"] for row in candidates]
            )
        }
        for name, blocks in feature_sets.items():
            predictions = group_ridge_predictions(blocks, targets, groups, folds)
            methods[name] = _rank_metrics(targets, predictions)
        baseline = methods["structure_slowfast"]
        fusion = methods["structure_slowfast_qwen_deepstack0"]
        auc_gain = (
            fusion["roc_auc_iou_0.5"] - baseline["roc_auc_iou_0.5"]
            if fusion["roc_auc_iou_0.5"] is not None
            and baseline["roc_auc_iou_0.5"] is not None else None
        )
        spearman_gain = (
            fusion["spearman_iou"] - baseline["spearman_iou"]
            if fusion["spearman_iou"] is not None
            and baseline["spearman_iou"] is not None else None
        )
        report["targets"][target_name] = {
            "methods": methods,
            "predeclared_fusion_vs_slowfast": {
                "auc_gain": auc_gain,
                "spearman_gain": spearman_gain,
            },
        }
    evidence = report["targets"]["quality_iou_any"][
        "predeclared_fusion_vs_slowfast"
    ]
    report["thresholds"] = {
        "min_auc_gain": float(min_auc_gain),
        "min_spearman_gain": float(min_spearman_gain),
    }
    report["source_gate_pass"] = bool(
        evidence["auc_gain"] is not None
        and evidence["auc_gain"] >= float(min_auc_gain)
        and evidence["spearman_gain"] is not None
        and evidence["spearman_gain"] >= float(min_spearman_gain)
    )
    return report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--slowfast-root", required=True)
    parser.add_argument("--logit-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-pair-gap", type=float, default=30.0)
    parser.add_argument("--min-probe-gain", type=float, default=0.03)
    parser.add_argument("--min-relation-gain", type=float, default=0.03)
    parser.add_argument("--min-relation-pairs", type=int, default=20)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--feat-stride", type=float, default=16.0)
    parser.add_argument("--feat-num-frames", type=float, default=32.0)
    parser.add_argument(
        "--annotation-json",
        default="data/ek100_slowfast/annotations/epic_kitchens_100_verb_detailed_split.json",
    )
    parser.add_argument("--min-quality-auc-gain", type=float, default=0.02)
    parser.add_argument("--min-quality-spearman-gain", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    all_rows = read_jsonl(args.manifest)
    database = json.loads(Path(args.annotation_json).read_text(encoding="utf-8"))[
        "database"
    ]
    quality_rows = proposal_quality_targets(all_rows, database)
    rows = [row for row in quality_rows if row.get("gt_assigned_class_ids")]
    qwen = load_feature_variants(args.feature_index, args.feature_root)
    slowfast = slowfast_variants(
        rows, args.slowfast_root, args.fps, args.feat_stride, args.feat_num_frames
    )
    candidates = [
        row for row in rows
        if all(row["candidate_id"] in values for values in qwen.values())
    ]
    labels = np.asarray([row["gt_assigned_class_ids"][0] for row in candidates])
    groups = np.asarray([row["video_id"] for row in candidates])
    probes = {}
    all_variants = {**slowfast, **qwen}
    for name, values in all_variants.items():
        matrix = normalize([values[row["candidate_id"]] for row in candidates])
        probes[name] = group_linear_probe(matrix, labels, groups, args.folds)
    for qwen_name, qwen_values in qwen.items():
        suffix = qwen_name.rsplit("_", 1)[-1]
        slowfast_name = "slowfast_" + suffix
        if slowfast_name not in slowfast:
            continue
        matrix = normalize([
            np.concatenate((
                slowfast[slowfast_name][row["candidate_id"]],
                qwen_values[row["candidate_id"]],
            ))
            for row in candidates
        ])
        probes["fusion_" + qwen_name] = group_linear_probe(
            matrix, labels, groups, args.folds
        )
    logits = {
        row["candidate_id"]: int(row["class_id"])
        for row in read_jsonl(args.logit_predictions)
        if row.get("status") == "ok"
    }
    logit_rows = [row for row in candidates if row["candidate_id"] in logits]
    logit_accuracy = sum(
        logits[row["candidate_id"]] in row["gt_assigned_class_ids"]
        for row in logit_rows
    ) / len(logit_rows)
    relation = {
        name: relation_auc(candidates, values, args.max_pair_gap)
        for name, values in all_variants.items()
        if name.endswith("_static")
    }
    best_slowfast = max(
        (value["accuracy"] for name, value in probes.items() if name.startswith("slowfast_")),
        default=0.0,
    )
    best_fusion_name, best_fusion = max(
        ((name, value) for name, value in probes.items() if name.startswith("fusion_")),
        key=lambda item: item[1]["accuracy"],
    )
    slowfast_relation = relation["slowfast_static"]
    qwen_relation_name, qwen_relation = max(
        ((name, value) for name, value in relation.items() if not name.startswith("slowfast_")),
        key=lambda item: item[1]["auc"] if item[1]["auc"] is not None else -1.0,
    )
    probe_gain = best_fusion["accuracy"] - best_slowfast
    relation_gain = (
        qwen_relation["auc"] - slowfast_relation["auc"]
        if qwen_relation["auc"] is not None and slowfast_relation["auc"] is not None
        else None
    )
    gate = bool(
        probe_gain >= args.min_probe_gain
        and relation_gain is not None
        and relation_gain >= args.min_relation_gain
        and qwen_relation["pair_count"] >= args.min_relation_pairs
    )
    quality_gate = proposal_quality_gate(
        quality_rows,
        qwen,
        slowfast_variants(
            quality_rows,
            args.slowfast_root,
            args.fps,
            args.feat_stride,
            args.feat_num_frames,
        ),
        args.folds,
        args.min_quality_auc_gain,
        args.min_quality_spearman_gain,
    )
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "selection_split": "source_val",
        "target_gt_usage": "forbidden",
        "known_action_count": len(candidates),
        "logit_accuracy": logit_accuracy,
        "probes": probes,
        "relations": relation,
        "best_slowfast_accuracy": best_slowfast,
        "best_fusion": {"name": best_fusion_name, **best_fusion},
        "probe_gain": probe_gain,
        "best_qwen_relation": {"name": qwen_relation_name, **qwen_relation},
        "slowfast_relation": slowfast_relation,
        "relation_gain": relation_gain,
        "thresholds": {
            "min_probe_gain": args.min_probe_gain,
            "min_relation_gain": args.min_relation_gain,
            "min_relation_pairs": args.min_relation_pairs,
            "max_pair_gap_seconds": args.max_pair_gap,
        },
        "source_gate_pass": gate,
        "proposal_quality_gate": quality_gate,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
