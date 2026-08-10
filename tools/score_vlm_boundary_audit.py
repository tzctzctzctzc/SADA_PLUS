"""Offline geometry ceilings and fixed-proposal scoring for VLM Stage2."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.vlm_boundary_utils import (
    centered_candidates,
    fine_step_seconds,
    read_jsonl,
    temporal_iou,
    valid_segment,
)


TIOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def kept_label(label, action_subsampling):
    if action_subsampling == "10_majority_classes":
        return int(label) < 10
    if action_subsampling == "20_random_classes":
        return int(label) < 20
    return True


def query_frame_time(query, frame_id):
    for frame in query["frames"]:
        if frame["frame_id"] == frame_id:
            return float(frame["timestamp_sec"])
    raise ValueError("unknown frame {} in {}".format(frame_id, query["query_id"]))


def selected_option_time(query, option):
    for frame in query["frames"]:
        if frame["option"] == option:
            return float(frame["timestamp_sec"])
    raise ValueError("unknown option {} in {}".format(option, query["query_id"]))


def prediction_decisions(rows):
    return {
        row["query_id"]: row["decision"]
        for row in rows
        if row.get("run_status") == "ok" and isinstance(row.get("decision"), dict)
    }


def best_joint_segment(starts, ends, target):
    candidates = [
        [float(left), float(right)]
        for left in starts
        for right in ends
        if valid_segment(left, right)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda segment: temporal_iou(segment, target))


def coarse_oracle_segment(query, target, baseline=None):
    times = [float(frame["timestamp_sec"]) for frame in query["frames"]]
    best = best_joint_segment(times, times, target)
    if baseline is not None and (
        best is None or temporal_iou(baseline, target) >= temporal_iou(best, target)
    ):
        return list(map(float, baseline))
    return best


def two_stage_oracle_segments(
    proposal,
    query,
    target,
    candidate_count=7,
    maximum_step=0.3,
    minimum_step=0.1,
):
    baseline = list(map(float, proposal["coarse_segment_seconds"]))
    rough = coarse_oracle_segment(query, target)
    if rough is None:
        return None, None
    step = fine_step_seconds(
        proposal["origin_stride_seconds"], maximum_step, minimum_step
    )
    duration = max(0.0, float(proposal["video_duration_seconds"]) - 0.05)
    starts = sorted(
        set(
            centered_candidates(rough[0], step, candidate_count, duration)
            + [baseline[0]]
        )
    )
    ends = sorted(
        set(
            centered_candidates(rough[1], step, candidate_count, duration)
            + [baseline[1]]
        )
    )
    independent = [
        min(starts, key=lambda value: abs(value - float(target[0]))),
        min(ends, key=lambda value: abs(value - float(target[1]))),
    ]
    if not valid_segment(*independent):
        independent = rough
    joint = best_joint_segment(starts, ends, target) or rough
    if temporal_iou(independent, target) < temporal_iou(baseline, target):
        independent = baseline
    if temporal_iou(joint, target) < temporal_iou(baseline, target):
        joint = baseline
    return independent, joint


def vlm_rough_segment(proposal, query, decision):
    coarse = list(map(float, proposal["coarse_segment_seconds"]))
    if not decision or decision.get("status") != "VALID_ACTION":
        return coarse
    candidate = [
        query_frame_time(query, decision["rough_start_frame"]),
        query_frame_time(query, decision["rough_end_frame"]),
    ]
    return candidate if valid_segment(*candidate) else coarse


def vlm_refined_segment(proposal, query, coarse_decision, fine_queries, fine_decisions):
    rough = vlm_rough_segment(proposal, query, coarse_decision)
    if not coarse_decision or coarse_decision.get("status") != "VALID_ACTION":
        return rough
    coarse = list(map(float, proposal["coarse_segment_seconds"]))
    candidate = list(coarse)
    for index, side in enumerate(("start", "end")):
        fine_query = fine_queries.get((proposal["proposal_id"], side))
        if fine_query is None:
            continue
        decision = fine_decisions.get(fine_query["query_id"])
        if decision and decision.get("status") == "VALID_ACTION":
            candidate[index] = selected_option_time(
                fine_query, decision["selected_option"]
            )
    return candidate if valid_segment(*candidate) else coarse


def segment_set(
    proposals,
    coarse_queries,
    coarse_predictions=None,
    fine_queries=None,
    fine_predictions=None,
    oracle_candidate_count=7,
    oracle_maximum_step=0.3,
    oracle_minimum_step=0.1,
):
    coarse_query_map = {row["proposal_id"]: row for row in coarse_queries}
    coarse_decisions = prediction_decisions(coarse_predictions or [])
    fine_query_map = {
        (row["proposal_id"], row["boundary_side"]): row
        for row in (fine_queries or [])
    }
    fine_decisions = prediction_decisions(fine_predictions or [])
    variants = {
        "baseline": {},
        "coarse_joint_oracle": {},
        "two_stage_independent_oracle": {},
        "two_stage_joint_oracle": {},
    }
    if coarse_predictions is not None:
        variants["vlm_rough"] = {}
        variants["vlm_refined"] = {}

    for proposal in proposals:
        proposal_id = proposal["proposal_id"]
        coarse = list(map(float, proposal["coarse_segment_seconds"]))
        query = coarse_query_map[proposal_id]
        for name in variants:
            variants[name][proposal_id] = coarse
        posthoc = proposal.get("posthoc_gt") or {}
        target = posthoc.get("matched_segment_seconds")
        if posthoc.get("kind") == "VALID_ACTION" and target:
            coarse_oracle = coarse_oracle_segment(query, target, coarse) or coarse
            independent, joint = two_stage_oracle_segments(
                proposal,
                query,
                target,
                oracle_candidate_count,
                oracle_maximum_step,
                oracle_minimum_step,
            )
            variants["coarse_joint_oracle"][proposal_id] = coarse_oracle
            variants["two_stage_independent_oracle"][proposal_id] = independent or coarse
            variants["two_stage_joint_oracle"][proposal_id] = joint or coarse
        if coarse_predictions is not None:
            decision = coarse_decisions.get(query["query_id"])
            variants["vlm_rough"][proposal_id] = vlm_rough_segment(
                proposal, query, decision
            )
            variants["vlm_refined"][proposal_id] = vlm_refined_segment(
                proposal,
                query,
                decision,
                fine_query_map,
                fine_decisions,
            )
    return variants


def proposal_boundary_metrics(proposals, segments):
    deltas = []
    start_before, start_after, end_before, end_after = [], [], [], []
    baseline_ious, refined_ious = [], []
    high_quality = []
    for proposal in proposals:
        posthoc = proposal.get("posthoc_gt") or {}
        target = posthoc.get("matched_segment_seconds")
        if posthoc.get("kind") != "VALID_ACTION" or not target:
            continue
        before_segment = proposal["coarse_segment_seconds"]
        after_segment = segments[proposal["proposal_id"]]
        before = temporal_iou(before_segment, target)
        after = temporal_iou(after_segment, target)
        baseline_ious.append(before)
        refined_ious.append(after)
        deltas.append(after - before)
        duration = max(float(target[1]) - float(target[0]), 1e-12)
        start_before.append(abs(float(before_segment[0]) - float(target[0])) / duration)
        start_after.append(abs(float(after_segment[0]) - float(target[0])) / duration)
        end_before.append(abs(float(before_segment[1]) - float(target[1])) / duration)
        end_after.append(abs(float(after_segment[1]) - float(target[1])) / duration)
        if before >= 0.7:
            high_quality.append((before, after))

    def mean(values):
        return float(np.mean(values)) if values else None

    def median(values):
        return float(np.median(values)) if values else None

    high_deltas = [after - before for before, after in high_quality]

    return {
        "count": len(deltas),
        "delta_iou_mean": mean(deltas),
        "delta_iou_median": median(deltas),
        "wins": sum(value > 1e-9 for value in deltas),
        "losses": sum(value < -1e-9 for value in deltas),
        "ties": sum(abs(value) <= 1e-9 for value in deltas),
        "win_rate": mean([value > 1e-9 for value in deltas]),
        "loss_rate": mean([value < -1e-9 for value in deltas]),
        "tie_rate": mean([abs(value) <= 1e-9 for value in deltas]),
        "iou_at_0.5_before": mean([value >= 0.5 for value in baseline_ious]),
        "iou_at_0.5_after": mean([value >= 0.5 for value in refined_ious]),
        "iou_at_0.7_before": mean([value >= 0.7 for value in baseline_ious]),
        "iou_at_0.7_after": mean([value >= 0.7 for value in refined_ious]),
        "start_normalized_error_before": mean(start_before),
        "start_normalized_error_after": mean(start_after),
        "start_normalized_error_delta": (
            mean(start_after) - mean(start_before) if start_before else None
        ),
        "end_normalized_error_before": mean(end_before),
        "end_normalized_error_after": mean(end_after),
        "end_normalized_error_delta": (
            mean(end_after) - mean(end_before) if end_before else None
        ),
        "high_quality_count": len(high_quality),
        "high_quality_iou_before_mean": mean([value[0] for value in high_quality]),
        "high_quality_iou_after_mean": mean([value[1] for value in high_quality]),
        "high_quality_delta_iou_mean": mean(high_deltas),
        "high_quality_wins": sum(value > 1e-9 for value in high_deltas),
        "high_quality_losses": sum(value < -1e-9 for value in high_deltas),
        "high_quality_ties": sum(abs(value) <= 1e-9 for value in high_deltas),
    }


def validity_metrics(proposals, coarse_queries, coarse_predictions):
    decisions = prediction_decisions(coarse_predictions)
    query_map = {row["proposal_id"]: row for row in coarse_queries}
    labels = ("VALID_ACTION", "WRONG_ACTION", "NO_ACTION", "UNCERTAIN")
    matrix = {truth: {prediction: 0 for prediction in labels} for truth in labels[:-1]}
    missing = 0
    for proposal in proposals:
        truth = proposal["posthoc_gt"]["kind"]
        decision = decisions.get(query_map[proposal["proposal_id"]]["query_id"])
        if decision is None:
            missing += 1
            continue
        matrix[truth][decision["status"]] += 1
    decided_total = sum(
        matrix[truth][prediction]
        for truth in labels[:-1]
        for prediction in labels[:-1]
    )
    correct = sum(matrix[label][label] for label in labels[:-1])
    uncertain = sum(matrix[truth]["UNCERTAIN"] for truth in labels[:-1])
    total = decided_total + uncertain
    return {
        "confusion": matrix,
        "missing": missing,
        "abstention_rate": uncertain / total if total else None,
        "decided_accuracy": correct / decided_total if decided_total else None,
    }


def load_ground_truth(annotation_json, video_ids, action_subsampling):
    database = json.loads(Path(annotation_json).read_text(encoding="utf-8"))["database"]
    ground_truth = []
    for video_id in video_ids:
        for annotation in database[video_id].get("annotations", []):
            label = int(annotation["label_id"])
            if kept_label(label, action_subsampling):
                ground_truth.append(
                    {
                        "video_id": video_id,
                        "label": label,
                        "segment": list(map(float, annotation["segment"])),
                    }
                )
    return ground_truth


def interpolated_ap(precision, recall):
    precision = np.concatenate(([0.0], np.asarray(precision), [0.0]))
    recall = np.concatenate(([0.0], np.asarray(recall), [1.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    changed = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum((recall[changed] - recall[changed - 1]) * precision[changed]))


def fixed_proposal_map(proposals, segments, ground_truth, thresholds=TIOU_THRESHOLDS):
    classes = sorted({row["label"] for row in ground_truth})
    by_class_gt = defaultdict(list)
    for index, row in enumerate(ground_truth):
        by_class_gt[row["label"]].append((index, row))
    predictions = [
        {
            "video_id": row["video_id"],
            "label": int(row["predicted_class_id"]),
            "score": float(row["score"]),
            "segment": segments[row["proposal_id"]],
            "proposal_id": row["proposal_id"],
        }
        for row in proposals
    ]
    result = {}
    for threshold in thresholds:
        aps = []
        for label in classes:
            class_gt = by_class_gt[label]
            class_predictions = sorted(
                (row for row in predictions if row["label"] == label),
                key=lambda row: (-row["score"], row["proposal_id"]),
            )
            matched = set()
            tp, fp = [], []
            for prediction in class_predictions:
                same_video = [
                    (index, temporal_iou(prediction["segment"], gt["segment"]))
                    for index, gt in class_gt
                    if gt["video_id"] == prediction["video_id"]
                ]
                best = max(same_video, key=lambda item: item[1]) if same_video else None
                if (
                    best is not None
                    and best[1] >= float(threshold)
                    and best[0] not in matched
                ):
                    matched.add(best[0])
                    tp.append(1.0)
                    fp.append(0.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            if not class_gt:
                continue
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / len(class_gt)
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
            aps.append(interpolated_ap(precision, recall))
        result["{:.1f}".format(threshold)] = float(np.mean(aps)) if aps else None
    available = [value for value in result.values() if value is not None]
    result["average_0.1_0.5"] = float(np.mean(available)) if available else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", required=True)
    parser.add_argument("--coarse-manifest", required=True)
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coarse-predictions")
    parser.add_argument("--fine-manifest")
    parser.add_argument("--fine-predictions")
    parser.add_argument("--action-subsampling", default="10_majority_classes")
    parser.add_argument("--oracle-candidate-count", type=int, default=7)
    parser.add_argument("--oracle-maximum-step", type=float, default=0.3)
    parser.add_argument("--oracle-minimum-step", type=float, default=0.1)
    args = parser.parse_args()

    proposals = read_jsonl(args.proposals)
    coarse_queries = read_jsonl(args.coarse_manifest)
    coarse_predictions = (
        read_jsonl(args.coarse_predictions) if args.coarse_predictions else None
    )
    fine_queries = read_jsonl(args.fine_manifest) if args.fine_manifest else None
    fine_predictions = (
        read_jsonl(args.fine_predictions) if args.fine_predictions else None
    )
    variants = segment_set(
        proposals,
        coarse_queries,
        coarse_predictions,
        fine_queries,
        fine_predictions,
        args.oracle_candidate_count,
        args.oracle_maximum_step,
        args.oracle_minimum_step,
    )
    ground_truth = load_ground_truth(
        args.annotation_json,
        sorted({row["video_id"] for row in proposals}),
        args.action_subsampling,
    )
    report = {
        "protocol": "fixed-proposal-vlm-boundary-audit-v1",
        "fixed_proposal_invariants": {
            "proposal_count": len(proposals),
            "labels_scores_and_proposal_set": "frozen",
            "renms_after_refinement": False,
            "invalid_wrong_uncertain_detection_policy": "retain_coarse_segment",
            "ground_truth_use": "offline_scoring_only",
            "oracle_candidate_count": args.oracle_candidate_count,
            "oracle_maximum_step": args.oracle_maximum_step,
            "oracle_minimum_step": args.oracle_minimum_step,
        },
        "oracle_definitions": {
            "coarse_joint_oracle": (
                "best legal coarse-frame pair or KEEP original segment"
            ),
            "two_stage_independent_oracle": (
                "best start and end independently from fine grid or KEEP each side"
            ),
            "two_stage_joint_oracle": (
                "best legal start/end pair from fine grids plus per-side KEEP"
            ),
        },
        "variants": {
            name: {
                "proposal_boundary": proposal_boundary_metrics(proposals, segments),
                "fixed_proposal_map": fixed_proposal_map(
                    proposals, segments, ground_truth
                ),
            }
            for name, segments in variants.items()
        },
    }
    if coarse_predictions is not None:
        report["validity"] = validity_metrics(
            proposals, coarse_queries, coarse_predictions
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
