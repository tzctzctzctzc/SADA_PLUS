import subprocess
import sys
from pathlib import Path

from tools.build_vlm_boundary_manifest import build_coarse_rows, build_fine_rows


def proposal(duration=20.0, segment=None):
    return {
        "proposal_id": "v:l0:x5:a5:c2",
        "video_id": "v",
        "video_duration_seconds": duration,
        "predicted_class_id": 2,
        "predicted_class_name": "WASH",
        "score": 0.93,
        "coarse_segment_seconds": segment or [5.0, 9.0],
        "origin_stride_seconds": 0.533,
        "fps": 30.0,
        "fpn_level": 0,
        "posthoc_gt": {"kind": "VALID_ACTION"},
    }


def residual_decision(row, start="KEEP_SADA", end="UNCERTAIN"):
    return {
        "action_status": "VALID_ACTION",
        "instance_description": "right hand washes the same white plate at the sink",
        "start_decision": start,
        "selected_start_option": (
            row["start_candidates"][2]["option"]
            if start == "SELECT_CANDIDATE" else None
        ),
        "end_decision": end,
        "selected_end_option": (
            row["end_candidates"][2]["option"]
            if end == "SELECT_CANDIDATE" else None
        ),
        "reason": "conservative residual choice",
    }


def test_coarse_manifest_separates_context_anchors_and_real_frame_pairs():
    row = build_coarse_rows([proposal()])[0]
    assert row["context_window_seconds"] == [1.0, 13.0]
    assert row["sampling_protocol"] == "separate-context-real-frame-transitions-v3"
    frame_indices = [frame["frame_index"] for frame in row["frames"]]
    assert frame_indices == sorted(set(frame_indices))
    assert all(
        abs(frame["timestamp_sec"] - frame["frame_index"] / 30.0) < 1e-9
        for frame in row["frames"]
    )
    frame_map = {frame["frame_id"]: frame for frame in row["frames"]}
    for side in ("start", "end"):
        assert len(row["{}_candidates".format(side)]) == 5
        for candidate in row["{}_candidates".format(side)]:
            before = frame_map[candidate["before_frame_id"]]
            after = frame_map[candidate["after_frame_id"]]
            assert after["frame_index"] - before["frame_index"] == 2
    assert set(row["context_frame_ids"]).issubset(frame_map)
    assert set(row["instance_anchor_frame_ids"]).issubset(frame_map)
    assert "score" not in row and "fpn_level" not in row and "posthoc_gt" not in row


def test_adaptive_window_expands_short_undercovered_proposal_by_four_seconds():
    row = build_coarse_rows([proposal(duration=10.0, segment=[4.0, 4.5])])[0]
    assert row["context_window_seconds"] == [0.0, 8.5]
    assert min(item["timestamp_sec"] for item in row["start_candidates"]) < 0.1
    assert max(item["timestamp_sec"] for item in row["end_candidates"]) > 8.4


def test_fine_manifest_compares_candidate_and_sada_for_same_instance():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": residual_decision(
            coarse[0], start="SELECT_CANDIDATE", end="KEEP_SADA"
        ),
    }]
    first = build_fine_rows(proposals, coarse, prediction)
    second = build_fine_rows(proposals, coarse, prediction)
    assert first == second
    assert {row["boundary_side"] for row in first} == {"start", "end"}
    for row in first:
        assert row["instance_description"].startswith("right hand washes")
        side_index = 0 if row["boundary_side"] == "start" else 1
        assert row["sada_boundary_seconds"] == proposal()["coarse_segment_seconds"][side_index]
        frame_map = {frame["frame_id"]: frame for frame in row["frames"]}
        for candidate in row["candidates"]:
            before = frame_map[candidate["before_frame_id"]]
            after = frame_map[candidate["after_frame_id"]]
            assert 1 <= after["frame_index"] - before["frame_index"] <= 3
        assert "score" not in row and "posthoc_gt" not in row


def test_fine_grid_is_safe_near_video_boundary_and_frame_deduplicated():
    proposals = [proposal(duration=1.0, segment=[0.05, 0.5])]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": residual_decision(coarse[0], start="KEEP_SADA"),
    }]
    rows = build_fine_rows(proposals, coarse, prediction)
    indices = [frame["frame_index"] for frame in rows[0]["frames"]]
    assert indices == sorted(set(indices))
    assert 0 <= indices[0] < indices[-1] <= 29


def test_fine_manifest_skips_abstained_side_and_non_valid_action():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    decision = residual_decision(coarse[0], start="UNCERTAIN", end="NOT_VISIBLE")
    prediction = [{
        "query_id": coarse[0]["query_id"], "run_status": "ok", "decision": decision
    }]
    assert build_fine_rows(proposals, coarse, prediction) == []
    decision["action_status"] = "NO_ACTION"
    decision["start_decision"] = "KEEP_SADA"
    assert build_fine_rows(proposals, coarse, prediction) == []


def test_boundary_only_manifest_does_not_require_action_status():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals, task_mode="boundary-only")
    decision = residual_decision(coarse[0], start="KEEP_SADA")
    decision.pop("action_status")
    prediction = [{
        "query_id": coarse[0]["query_id"], "run_status": "ok", "decision": decision
    }]
    fine = build_fine_rows(proposals, coarse, prediction)
    assert [row["boundary_side"] for row in fine] == ["start"]
    assert fine[0]["task_mode"] == "boundary-only"


def test_manifest_and_scorer_support_direct_script_entrypoints():
    root = Path(__file__).resolve().parents[1]
    for script in (
        root / "tools" / "build_vlm_boundary_manifest.py",
        root / "tools" / "score_vlm_boundary_audit.py",
    ):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
