import json

from tools.score_vlm_boundary_audit import (
    coarse_oracle_segment,
    fixed_proposal_map,
    load_ground_truth,
    proposal_boundary_metrics,
    segment_set,
    two_stage_candidate_space,
    two_stage_oracle_segments,
)
from tools.vlm_boundary_utils import temporal_iou


def proposal(kind="VALID_ACTION"):
    return {
        "proposal_id": "p1",
        "video_id": "v1",
        "video_duration_seconds": 10.0,
        "predicted_class_id": 1,
        "predicted_class_name": "PUT",
        "score": 0.9,
        "coarse_segment_seconds": [1.0, 9.0],
        "origin_stride_seconds": 0.6,
        "posthoc_gt": {
            "kind": kind,
            "matched_segment_seconds": [3.0, 7.0] if kind == "VALID_ACTION" else None,
        },
    }


def coarse_query():
    return {
        "query_id": "p1:coarse",
        "proposal_id": "p1",
        "frames": [
            {"frame_id": "F{}".format(index + 1), "timestamp_sec": float(value)}
            for index, value in enumerate((0, 2, 4, 6, 8, 9.95))
        ],
    }


def coarse_decision(start="VISIBLE", end="VISIBLE"):
    return {
        "action_status": "VALID_ACTION",
        "start_status": start,
        "rough_start_before_frame": "F2" if start == "VISIBLE" else None,
        "rough_start_after_frame": "F3" if start == "VISIBLE" else None,
        "end_status": end,
        "rough_end_before_frame": "F4" if end == "VISIBLE" else None,
        "rough_end_after_frame": "F5" if end == "VISIBLE" else None,
        "reason": "visible",
    }


def fine_query(side, candidates):
    frames = [
        {"frame_id": "F{}".format(index + 1), "timestamp_sec": value}
        for index, value in enumerate(range(len(candidates) + 1))
    ]
    return {
        "query_id": "p1:fine:{}".format(side),
        "proposal_id": "p1",
        "boundary_side": side,
        "frames": frames,
        "candidates": [
            {
                "option": chr(ord("A") + index),
                "before_frame_id": frames[index]["frame_id"],
                "after_frame_id": frames[index + 1]["frame_id"],
                "timestamp_sec": value,
            }
            for index, value in enumerate(candidates)
        ],
    }


def prediction(query_id, decision):
    return {"query_id": query_id, "run_status": "ok", "decision": decision}


def test_geometry_oracles_use_exact_transition_choice_space_without_degradation():
    item = proposal()
    query = coarse_query()
    target = [3.0, 7.0]
    baseline = temporal_iou(item["coarse_segment_seconds"], target)
    coarse = coarse_oracle_segment(query, target, item["coarse_segment_seconds"])
    independent, joint = two_stage_oracle_segments(item, query, target)
    space = two_stage_candidate_space(item, query)
    assert 1.0 in space["start"] and 9.0 in space["end"]
    assert space["coarse_transition_count"] == len(query["frames"]) - 1
    assert temporal_iou(coarse, target) >= baseline
    assert temporal_iou(independent, target) >= temporal_iou(coarse, target)
    assert temporal_iou(joint, target) >= temporal_iou(coarse, target)


def test_non_valid_vlm_decision_retains_coarse_detection_and_invalid_sides():
    item = proposal(kind="NO_ACTION")
    query = coarse_query()
    decision = {
        "action_status": "NO_ACTION",
        "start_status": "NOT_VISIBLE",
        "rough_start_before_frame": None,
        "rough_start_after_frame": None,
        "end_status": "NOT_VISIBLE",
        "rough_end_before_frame": None,
        "rough_end_after_frame": None,
        "reason": "none",
    }
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], decision)], return_details=True
    )
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"] == {
        "seconds": 1.0, "valid": False, "source": "INVALID"
    }


def test_refined_segment_accepts_one_side_and_invalid_side_cannot_supervise():
    item = proposal()
    query = coarse_query()
    coarse_predictions = [prediction(query["query_id"], coarse_decision())]
    fine_queries = [fine_query("start", [2.7, 3.0]), fine_query("end", [7.0, 7.3])]
    fine_predictions = [
        prediction("p1:fine:start", {
            "decision": "SELECT_CANDIDATE", "selected_option": "B", "reason": "edge"
        }),
        prediction("p1:fine:end", {
            "decision": "UNCERTAIN", "selected_option": None, "reason": "blur"
        }),
    ]
    variants, details = segment_set(
        [item], [query], coarse_predictions, fine_queries, fine_predictions,
        return_details=True,
    )
    assert variants["vlm_rough"]["p1"] == [3.0, 7.0]
    assert variants["vlm_refined"]["p1"] == [3.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"] == {
        "seconds": 3.0, "valid": True, "source": "VLM"
    }
    assert details["vlm_refined"]["p1"]["end"] == {
        "seconds": 9.0, "valid": False, "source": "INVALID"
    }


def test_keep_coarse_preserves_rough_transition_with_keep_source():
    item, query = proposal(), coarse_query()
    rows = [fine_query("start", [2.7, 3.0])]
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], coarse_decision(end="NOT_VISIBLE"))],
        rows,
        [prediction("p1:fine:start", {
            "decision": "KEEP_COARSE", "selected_option": None, "reason": "best"
        })],
        return_details=True,
    )
    assert variants["vlm_refined"]["p1"] == [3.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"]["source"] == "KEEP"


def test_start_not_before_end_safely_invalidates_both_sides():
    item, query = proposal(), coarse_query()
    bad_start = fine_query("start", [9.5])
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], coarse_decision(end="NOT_VISIBLE"))],
        [bad_start],
        [prediction("p1:fine:start", {
            "decision": "SELECT_CANDIDATE", "selected_option": "A", "reason": "bad"
        })],
        return_details=True,
    )
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]
    assert all(not value["valid"] for value in details["vlm_refined"]["p1"].values())


def test_fixed_proposal_map_uses_official_unlocked_second_best_matching():
    first = proposal()
    second = {**proposal(), "proposal_id": "p2", "score": 0.8}
    ground_truth = [
        {"video_id": "v1", "label": 1, "segment": [0.0, 2.0]},
        {"video_id": "v1", "label": 1, "segment": [1.0, 3.0]},
    ]
    result = fixed_proposal_map(
        [first, second],
        {"p1": [0.0, 2.0], "p2": [0.2, 2.2]},
        ground_truth,
    )
    assert result["0.4"] == 1.0


def test_ground_truth_loader_removes_official_duplicates(tmp_path):
    annotation = {
        "database": {
            "v1": {
                "subset": "synthetic",
                "annotations": [
                    {"segment": [1, 2], "label_id": 1},
                    {"segment": [1, 2], "label_id": 1},
                ],
            }
        }
    }
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    assert len(load_ground_truth(path, ["v1"], "10_majority_classes")) == 1


def test_high_quality_degradation_rate_is_reported():
    item = proposal()
    item["coarse_segment_seconds"] = [3.0, 7.0]
    metrics = proposal_boundary_metrics([item], {"p1": [2.0, 8.0]})
    assert metrics["high_quality_count"] == 1
    assert metrics["high_quality_degradation_rate"] == 1.0
