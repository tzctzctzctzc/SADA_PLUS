"""Offline geometry ceilings and fixed-proposal scoring for VLM Stage2."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.vlm_boundary_utils import (
    fine_step_seconds,
    read_jsonl,
    temporal_iou,
    transition_grid,
    transition_midpoints,
    valid_segment,
)
from libs.evaluation.evaluator_anet import (
    ANETdetection,
    compute_average_precision_detection,
    remove_duplicate_annotations,
)


TIOU_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def kept_label(label, action_subsampling):
    if action_subsampling == "10_majority_classes":
        return int(label) < 10
    if action_subsampling == "20_majority_classes":
        return int(label) in set(range(14)) | {15, 17, 18, 19}
    return True


def query_transition_time(query, before_frame_id, after_frame_id):
    frame_ids = [frame["frame_id"] for frame in query["frames"]]
    before_index = frame_ids.index(before_frame_id)
    after_index = frame_ids.index(after_frame_id)
    if after_index != before_index + 1:
        raise ValueError("boundary transition frames must be adjacent")
    return 0.5 * (
        float(query["frames"][before_index]["timestamp_sec"])
        + float(query["frames"][after_index]["timestamp_sec"])
    )


def selected_option_time(query, option):
    for candidate in query["candidates"]:
        if candidate["option"] == option:
            return float(candidate["timestamp_sec"])
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
    times = transition_midpoints(query["frames"])
    starts = list(times)
    ends = list(times)
    if baseline is not None:
        starts.append(float(baseline[0]))
        ends.append(float(baseline[1]))
    best = best_joint_segment(starts, ends, target)
    if baseline is not None and (
        best is None or temporal_iou(baseline, target) >= temporal_iou(best, target)
    ):
        return list(map(float, baseline))
    return best


def two_stage_candidate_space(
    proposal,
    query,
    candidate_count=7,
    maximum_step=0.3,
    minimum_step=0.1,
):
    """Exact per-side values reachable through coarse -> fine, including KEEP."""
    baseline = list(map(float, proposal["coarse_segment_seconds"]))
    step = fine_step_seconds(
        proposal["origin_stride_seconds"], maximum_step, minimum_step
    )
    duration = max(0.0, float(proposal["video_duration_seconds"]) - 0.05)
    reachable = set()
    paths = []
    for rough in transition_midpoints(query["frames"]):
        fine_frames = transition_grid(rough, step, int(candidate_count), duration)
        values = set(transition_midpoints(fine_frames)) | {float(rough)}
        reachable.update(values)
        paths.append({"rough": float(rough), "values": sorted(values)})
    return {
        "start": sorted(reachable | {baseline[0]}),
        "end": sorted(reachable | {baseline[1]}),
        "visible_side_paths": paths,
        "coarse_transition_count": len(transition_midpoints(query["frames"])),
        "fine_candidate_count_per_visible_side": int(candidate_count),
        "fine_step_seconds": float(step),
        "includes_keep_coarse": True,
        "includes_side_abstention_detection_fallback": True,
    }


def two_stage_oracle_segments(
    proposal,
    query,
    target,
    candidate_count=7,
    maximum_step=0.3,
    minimum_step=0.1,
):
    baseline = list(map(float, proposal["coarse_segment_seconds"]))
    space = two_stage_candidate_space(
        proposal,
        query,
        candidate_count,
        maximum_step,
        minimum_step,
    )
    starts, ends = space["start"], space["end"]
    independent = [
        min(starts, key=lambda value: abs(value - float(target[0]))),
        min(ends, key=lambda value: abs(value - float(target[1]))),
    ]
    if not valid_segment(*independent):
        independent = baseline
    joint_candidates = [baseline]
    for path in space["visible_side_paths"]:
        joint_candidates.extend(
            [baseline[0], value]
            for value in path["values"]
            if valid_segment(baseline[0], value)
        )
        joint_candidates.extend(
            [value, baseline[1]]
            for value in path["values"]
            if valid_segment(value, baseline[1])
        )
    for start_path in space["visible_side_paths"]:
        for end_path in space["visible_side_paths"]:
            if start_path["rough"] >= end_path["rough"]:
                continue
            joint_candidates.extend(
                [left, right]
                for left in start_path["values"]
                for right in end_path["values"]
                if valid_segment(left, right)
            )
    joint = max(
        joint_candidates, key=lambda segment: temporal_iou(segment, target)
    )
    if temporal_iou(independent, target) < temporal_iou(baseline, target):
        independent = baseline
    if temporal_iou(joint, target) < temporal_iou(baseline, target):
        joint = baseline
    return independent, joint


def side_result(seconds, valid, source):
    return {
        "seconds": float(seconds),
        "valid": bool(valid),
        "source": str(source),
    }


def invalidate_result(coarse):
    return {
        "segment": list(map(float, coarse)),
        "sides": {
            "start": side_result(coarse[0], False, "INVALID"),
            "end": side_result(coarse[1], False, "INVALID"),
        },
    }


def is_boundary_only(query):
    return query.get("task_mode") == "boundary-only"


def vlm_rough_result(proposal, query, decision):
    coarse = list(map(float, proposal["coarse_segment_seconds"]))
    result = invalidate_result(coarse)
    if not decision or (
        not is_boundary_only(query)
        and decision.get("action_status") != "VALID_ACTION"
    ):
        return result
    candidate = list(coarse)
    for index, side in enumerate(("start", "end")):
        if decision.get("{}_status".format(side)) != "VISIBLE":
            continue
        value = query_transition_time(
            query,
            decision["rough_{}_before_frame".format(side)],
            decision["rough_{}_after_frame".format(side)],
        )
        candidate[index] = value
        result["sides"][side] = side_result(value, True, "VLM")
    if not valid_segment(*candidate):
        return invalidate_result(coarse)
    result["segment"] = candidate
    return result


def vlm_rough_segment(proposal, query, decision):
    return vlm_rough_result(proposal, query, decision)["segment"]


def vlm_refined_result(proposal, query, coarse_decision, fine_queries, fine_decisions):
    rough = vlm_rough_result(proposal, query, coarse_decision)
    if not coarse_decision or (
        not is_boundary_only(query)
        and coarse_decision.get("action_status") != "VALID_ACTION"
    ):
        return rough
    coarse = list(map(float, proposal["coarse_segment_seconds"]))
    candidate = list(coarse)
    sides = {
        side: dict(rough["sides"][side]) for side in ("start", "end")
    }
    for index, side in enumerate(("start", "end")):
        if coarse_decision.get("{}_status".format(side)) != "VISIBLE":
            continue
        fine_query = fine_queries.get((proposal["proposal_id"], side))
        if fine_query is None:
            candidate[index] = rough["sides"][side]["seconds"]
            continue
        decision = fine_decisions.get(fine_query["query_id"])
        if decision and decision.get("decision") == "SELECT_CANDIDATE":
            value = selected_option_time(
                fine_query, decision["selected_option"]
            )
            candidate[index] = value
            sides[side] = side_result(value, True, "VLM")
        elif decision and decision.get("decision") == "KEEP_COARSE":
            value = rough["sides"][side]["seconds"]
            candidate[index] = value
            sides[side] = side_result(value, True, "KEEP")
        else:
            sides[side] = side_result(coarse[index], False, "INVALID")
    if not valid_segment(*candidate):
        return invalidate_result(coarse)
    return {"segment": candidate, "sides": sides}


def vlm_refined_segment(proposal, query, coarse_decision, fine_queries, fine_decisions):
    return vlm_refined_result(
        proposal, query, coarse_decision, fine_queries, fine_decisions
    )["segment"]


def segment_set(
    proposals,
    coarse_queries,
    coarse_predictions=None,
    fine_queries=None,
    fine_predictions=None,
    oracle_candidate_count=7,
    oracle_maximum_step=0.3,
    oracle_minimum_step=0.1,
    return_details=False,
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
    details = {"vlm_rough": {}, "vlm_refined": {}, "oracle_spaces": {}}

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
            space = two_stage_candidate_space(
                proposal,
                query,
                oracle_candidate_count,
                oracle_maximum_step,
                oracle_minimum_step,
            )
            details["oracle_spaces"][proposal_id] = {
                "start_values_seconds": space["start"],
                "end_values_seconds": space["end"],
                "coarse_transition_count": space["coarse_transition_count"],
                "fine_candidate_count_per_visible_side": space[
                    "fine_candidate_count_per_visible_side"
                ],
                "fine_step_seconds": space["fine_step_seconds"],
                "choices": [
                    "SELECT_CANDIDATE",
                    "KEEP_COARSE",
                    "SIDE_ABSTENTION_TO_DETECTION_FALLBACK",
                ],
                "joint_path_constraint": (
                    "when both sides are valid, their coarse rough transitions "
                    "must satisfy rough_start < rough_end"
                ),
            }
        if coarse_predictions is not None:
            decision = coarse_decisions.get(query["query_id"])
            rough_result = vlm_rough_result(
                proposal, query, decision
            )
            refined_result = vlm_refined_result(
                proposal,
                query,
                decision,
                fine_query_map,
                fine_decisions,
            )
            variants["vlm_rough"][proposal_id] = rough_result["segment"]
            variants["vlm_refined"][proposal_id] = refined_result["segment"]
            details["vlm_rough"][proposal_id] = rough_result["sides"]
            details["vlm_refined"][proposal_id] = refined_result["sides"]
    return (variants, details) if return_details else variants


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
        "high_quality_degradation_rate": mean(
            [value < -1e-9 for value in high_deltas]
        ),
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
        matrix[truth][decision["action_status"]] += 1
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
        annotations = remove_duplicate_annotations(
            database[video_id].get("annotations", [])
        )
        for annotation in annotations:
            label = int(annotation["label_id"])
            segment = list(map(float, annotation["segment"]))
            if segment[0] != segment[1] and kept_label(label, action_subsampling):
                ground_truth.append(
                    {
                        "video_id": video_id,
                        "label": label,
                        "segment": segment,
                    }
                )
    return ground_truth


def prediction_dataframe(proposals, segments):
    return pd.DataFrame(
        {
            "video-id": [row["video_id"] for row in proposals],
            "t-start": [segments[row["proposal_id"]][0] for row in proposals],
            "t-end": [segments[row["proposal_id"]][1] for row in proposals],
            "label": [int(row["predicted_class_id"]) for row in proposals],
            "score": [float(row["score"]) for row in proposals],
        }
    )


def fixed_proposal_map(proposals, segments, ground_truth, thresholds=TIOU_THRESHOLDS):
    classes = sorted({row["label"] for row in ground_truth})
    threshold_array = np.asarray(thresholds, dtype=float)
    gt_frame = pd.DataFrame(
        {
            "video-id": [row["video_id"] for row in ground_truth],
            "t-start": [row["segment"][0] for row in ground_truth],
            "t-end": [row["segment"][1] for row in ground_truth],
            "label": [row["label"] for row in ground_truth],
        }
    )
    predictions = prediction_dataframe(proposals, segments)
    class_aps = []
    for label in classes:
        class_gt = gt_frame[gt_frame["label"] == label].reset_index(drop=True)
        class_predictions = predictions[
            predictions["label"] == label
        ].reset_index(drop=True)
        computed = compute_average_precision_detection(
            class_gt, class_predictions, threshold_array
        )
        class_aps.append(np.asarray(computed[0] if isinstance(computed, tuple) else computed))
    mean_ap = np.mean(np.stack(class_aps), axis=0) if class_aps else None
    result = {
        "{:.1f}".format(threshold): (
            float(mean_ap[index]) if mean_ap is not None else None
        )
        for index, threshold in enumerate(threshold_array)
    }
    result["average_0.1_0.5"] = (
        float(np.mean(mean_ap)) if mean_ap is not None else None
    )
    return result


def official_evaluator_map(
    proposals,
    segments,
    annotation_json,
    split,
    action_subsampling,
    thresholds=TIOU_THRESHOLDS,
):
    database = json.loads(Path(annotation_json).read_text(encoding="utf-8"))["database"]
    split_videos = {
        video_id
        for video_id, value in database.items()
        if value["subset"].lower() == str(split).lower()
    }
    proposal_videos = {row["video_id"] for row in proposals}
    if proposal_videos != split_videos:
        raise ValueError(
            "official parity requires the complete split: proposals={} split={}".format(
                len(proposal_videos), len(split_videos)
            )
        )
    evaluator = ANETdetection(
        annotation_json,
        split=[split],
        tiou_thresholds=np.asarray(thresholds, dtype=float),
        num_workers=1,
        action_subsampling=action_subsampling,
    )
    values = evaluator.evaluate(
        prediction_dataframe(proposals, segments), verbose=False
    )[0]
    result = {
        "{:.1f}".format(threshold): float(values[index])
        for index, threshold in enumerate(thresholds)
    }
    result["average_0.1_0.5"] = float(np.mean(values))
    return result


def evaluator_parity(fixed_map, official_map, tolerance=1e-12):
    keys = ["{:.1f}".format(value) for value in TIOU_THRESHOLDS]
    differences = [abs(float(fixed_map[key]) - float(official_map[key])) for key in keys]
    maximum = max(differences) if differences else 0.0
    return {
        "same_prediction_set": True,
        "maximum_absolute_difference": float(maximum),
        "tolerance": float(tolerance),
        "passed": bool(maximum <= float(tolerance)),
    }


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
    parser.add_argument("--official-split")
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
    task_modes = {row.get("task_mode", "validity-and-boundary") for row in coarse_queries}
    if len(task_modes) != 1:
        raise ValueError("coarse manifest mixes task modes")
    task_mode = next(iter(task_modes))
    variants, details = segment_set(
        proposals,
        coarse_queries,
        coarse_predictions,
        fine_queries,
        fine_predictions,
        args.oracle_candidate_count,
        args.oracle_maximum_step,
        args.oracle_minimum_step,
        True,
    )
    ground_truth = load_ground_truth(
        args.annotation_json,
        sorted({row["video_id"] for row in proposals}),
        args.action_subsampling,
    )
    variant_reports = {}
    for name, segments in variants.items():
        fixed_map = fixed_proposal_map(proposals, segments, ground_truth)
        variant_report = {
            "proposal_boundary": proposal_boundary_metrics(proposals, segments),
            "fixed_proposal_map": fixed_map,
        }
        if args.official_split:
            official_map = official_evaluator_map(
                proposals,
                segments,
                args.annotation_json,
                args.official_split,
                args.action_subsampling,
            )
            parity = evaluator_parity(fixed_map, official_map)
            variant_report["official_evaluator_map"] = official_map
            variant_report["evaluator_parity"] = parity
            if not parity["passed"]:
                raise RuntimeError("official evaluator parity failed for {}".format(name))
        variant_reports[name] = variant_report

    report = {
        "protocol": "fixed-proposal-vlm-boundary-transition-audit-v3",
        "task_mode": task_mode,
        "boundary_semantics": {
            "representation": "midpoint of an adjacent chronological frame pair",
            "start": "not-in-action -> in-action",
            "end": "in-action -> out-of-action",
        },
        "fixed_proposal_invariants": {
            "proposal_count": len(proposals),
            "labels_scores_and_proposal_set": "frozen",
            "sada_action_occurrence_and_class": (
                "frozen_not_judged_by_vlm"
                if task_mode == "boundary-only"
                else "judged_by_vlm"
            ),
            "renms_after_refinement": False,
            "invalid_or_uncertain_side_policy": "retain_coarse_boundary",
            "ground_truth_use": "offline_scoring_only",
            "oracle_candidate_count": args.oracle_candidate_count,
            "oracle_maximum_step": args.oracle_maximum_step,
            "oracle_minimum_step": args.oracle_minimum_step,
            "official_split": args.official_split,
        },
        "oracle_definitions": {
            "coarse_joint_oracle": (
                "best legal adjacent-pair midpoint combination or KEEP each side"
            ),
            "two_stage_independent_oracle": (
                "best start/end independently from every reachable fine transition, "
                "KEEP_COARSE, or side-abstention detection fallback"
            ),
            "two_stage_joint_oracle": (
                "best legal pair from the identical per-side reachable choice spaces"
            ),
        },
        "oracle_candidate_spaces": details["oracle_spaces"],
        "side_outputs": {
            key: value
            for key, value in details.items()
            if key in ("vlm_rough", "vlm_refined") and value
        },
        "variants": variant_reports,
    }
    if coarse_predictions is not None:
        report["validity"] = (
            {
                "applicable": False,
                "reason": "SADA action occurrence and class are frozen; VLM judges boundaries only",
            }
            if task_mode == "boundary-only"
            else {
                "applicable": True,
                **validity_metrics(proposals, coarse_queries, coarse_predictions),
            }
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
