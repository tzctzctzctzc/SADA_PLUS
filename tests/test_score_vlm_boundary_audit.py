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
        "fps": 30.0,
        "posthoc_gt": {
            "kind": kind,
            "matched_segment_seconds": [3.0, 7.0] if kind == "VALID_ACTION" else None,
        },
    }


def coarse_query():
    return {
        "query_id": "p1:coarse",
        "proposal_id": "p1",
        "start_candidates": [
            {"option": "A", "timestamp_sec": 3.0},
            {"option": "B", "timestamp_sec": 4.0},
        ],
        "end_candidates": [
            {"option": "C", "timestamp_sec": 6.0},
            {"option": "D", "timestamp_sec": 7.0},
        ],
    }


def coarse_decision(start="SELECT_CANDIDATE", end="SELECT_CANDIDATE"):
    return {
        "action_status": "VALID_ACTION",
        "instance_description": "right hand puts the same bowl on the counter",
        "start_decision": start,
        "selected_start_option": "A" if start == "SELECT_CANDIDATE" else None,
        "end_decision": end,
        "selected_end_option": "D" if end == "SELECT_CANDIDATE" else None,
        "reason": "residual comparison",
    }


def fine_query(side, candidates):
    return {
        "query_id": "p1:fine:{}".format(side),
        "proposal_id": "p1",
        "boundary_side": side,
        "candidates": [
            {"option": chr(ord("A") + index), "timestamp_sec": value}
            for index, value in enumerate(candidates)
        ],
    }


def prediction(query_id, decision):
    return {"query_id": query_id, "run_status": "ok", "decision": decision}


def test_geometry_oracles_use_side_specific_residual_space_without_degradation():
    item = proposal()
    query = coarse_query()
    target = [3.0, 7.0]
    baseline = temporal_iou(item["coarse_segment_seconds"], target)
    coarse = coarse_oracle_segment(query, target, item["coarse_segment_seconds"])
    independent, joint = two_stage_oracle_segments(item, query, target)
    space = two_stage_candidate_space(item, query)
    assert 1.0 in space["start"] and 9.0 in space["end"]
    assert space["coarse_transition_count"] == {"start": 2, "end": 2}
    assert space["includes_keep_sada"] is True
    assert temporal_iou(coarse, target) >= baseline
    assert temporal_iou(independent, target) >= temporal_iou(coarse, target)
    assert temporal_iou(joint, target) >= temporal_iou(coarse, target)


def test_non_valid_vlm_decision_retains_sada_detection_and_invalid_sides():
    item = proposal(kind="NO_ACTION")
    query = coarse_query()
    decision = coarse_decision(start="NOT_VISIBLE", end="NOT_VISIBLE")
    decision["action_status"] = "NO_ACTION"
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], decision)], return_details=True
    )
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"] == {
        "seconds": 1.0, "valid": False, "source": "INVALID"
    }


def test_refined_segment_accepts_one_side_and_abstained_side_cannot_supervise():
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


def test_boundary_only_decision_does_not_require_action_status():
    item = proposal()
    query = {**coarse_query(), "task_mode": "boundary-only"}
    decision = coarse_decision()
    decision.pop("action_status")
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], decision)],
        return_details=True,
    )
    assert variants["vlm_rough"]["p1"] == [3.0, 7.0]
    assert all(details["vlm_rough"]["p1"][side]["valid"] for side in ("start", "end"))


def test_keep_sada_can_revert_a_coarse_replacement():
    item, query = proposal(), coarse_query()
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], coarse_decision(end="NOT_VISIBLE"))],
        [fine_query("start", [2.7, 3.0])],
        [prediction("p1:fine:start", {
            "decision": "KEEP_SADA", "selected_option": None, "reason": "SADA best"
        })],
        return_details=True,
    )
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"]["source"] == "SADA_KEEP"


def test_keep_coarse_preserves_clear_coarse_improvement():
    item, query = proposal(), coarse_query()
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], coarse_decision(end="NOT_VISIBLE"))],
        [fine_query("start", [2.7, 3.0])],
        [prediction("p1:fine:start", {
            "decision": "KEEP_COARSE", "selected_option": None, "reason": "coarse best"
        })],
        return_details=True,
    )
    assert variants["vlm_refined"]["p1"] == [3.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"]["source"] == "COARSE_KEEP"


def test_coarse_keep_sada_is_terminal_even_if_legacy_fine_output_exists():
    item, query = proposal(), coarse_query()
    decision = coarse_decision(start="KEEP_SADA", end="NOT_VISIBLE")
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], decision)],
        [fine_query("start", [2.7, 3.0])],
        [prediction("p1:fine:start", {
            "decision": "SELECT_CANDIDATE", "selected_option": "B", "reason": "legacy"
        })],
        return_details=True,
    )
    assert variants["vlm_refined"]["p1"] == [1.0, 9.0]
    assert details["vlm_refined"]["p1"]["start"] == {
        "seconds": 1.0, "valid": True, "source": "SADA_KEEP"
    }


def test_start_not_before_end_safely_invalidates_both_sides():
    item, query = proposal(), coarse_query()
    variants, details = segment_set(
        [item], [query], [prediction(query["query_id"], coarse_decision(end="NOT_VISIBLE"))],
        [fine_query("start", [9.5])],
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
