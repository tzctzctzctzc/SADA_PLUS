from tools.score_vlm_boundary_audit import (
    coarse_oracle_segment,
    fixed_proposal_map,
    segment_set,
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
            for index, value in enumerate((0, 2, 4, 6, 8, 10))
        ],
    }


def test_geometry_oracles_do_not_degrade_valid_proposal():
    item = proposal()
    query = coarse_query()
    target = [3.0, 7.0]
    baseline = temporal_iou(item["coarse_segment_seconds"], target)
    coarse = coarse_oracle_segment(query, target)
    independent, joint = two_stage_oracle_segments(item, query, target)
    assert temporal_iou(coarse, target) >= baseline
    assert temporal_iou(independent, target) >= temporal_iou(coarse, target)
    assert temporal_iou(joint, target) >= temporal_iou(coarse, target)
    assert independent[0] < independent[1]
    assert joint[0] < joint[1]


def test_non_valid_vlm_decision_retains_coarse_detection():
    item = proposal(kind="NO_ACTION")
    query = coarse_query()
    predictions = [{
        "query_id": query["query_id"],
        "run_status": "ok",
        "decision": {
            "status": "NO_ACTION",
            "rough_start_frame": None,
            "rough_end_frame": None,
            "reason": "none",
        },
    }]
    variants = segment_set([item], [query], predictions)
    assert variants["vlm_rough"]["p1"] == [1.0, 9.0]
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]


def test_refined_segment_changes_only_independently_accepted_side():
    item = proposal()
    query = coarse_query()
    coarse_predictions = [{
        "query_id": query["query_id"],
        "run_status": "ok",
        "decision": {
            "status": "VALID_ACTION",
            "rough_start_frame": "F2",
            "rough_end_frame": "F5",
            "reason": "visible",
        },
    }]
    fine_queries = [
        {
            "query_id": "p1:fine:start",
            "proposal_id": "p1",
            "boundary_side": "start",
            "frames": [
                {"option": "A", "timestamp_sec": 2.7},
                {"option": "B", "timestamp_sec": 3.0},
            ],
        },
        {
            "query_id": "p1:fine:end",
            "proposal_id": "p1",
            "boundary_side": "end",
            "frames": [
                {"option": "A", "timestamp_sec": 7.0},
                {"option": "B", "timestamp_sec": 7.3},
            ],
        },
    ]
    fine_predictions = [
        {
            "query_id": "p1:fine:start",
            "run_status": "ok",
            "decision": {"status": "VALID_ACTION", "selected_option": "B"},
        },
        {
            "query_id": "p1:fine:end",
            "run_status": "ok",
            "decision": {"status": "UNCERTAIN", "selected_option": None},
        },
    ]
    variants = segment_set(
        [item], [query], coarse_predictions, fine_queries, fine_predictions
    )
    assert variants["vlm_rough"]["p1"] == [2.0, 8.0]
    assert variants["vlm_refined"]["p1"] == [3.0, 9.0]


def test_fixed_proposal_map_changes_only_segments():
    proposals = [proposal()]
    ground_truth = [{"video_id": "v1", "label": 1, "segment": [3.0, 7.0]}]
    weak = fixed_proposal_map(proposals, {"p1": [0.0, 1.0]}, ground_truth)
    strong = fixed_proposal_map(proposals, {"p1": [3.0, 7.0]}, ground_truth)
    assert weak["0.5"] == 0.0
    assert strong["0.5"] == 1.0


def test_fixed_map_does_not_rematch_duplicate_detection_to_second_best_gt():
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
    assert result["0.4"] == 0.5
