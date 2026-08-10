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
        "fpn_level": 0,
        "posthoc_gt": {"kind": "VALID_ACTION"},
    }


def visible_pair(row, side, before_index):
    return {
        "action_status": "VALID_ACTION",
        "start_status": "VISIBLE" if side == "start" else "NOT_VISIBLE",
        "rough_start_before_frame": (
            row["frames"][before_index]["frame_id"] if side == "start" else None
        ),
        "rough_start_after_frame": (
            row["frames"][before_index + 1]["frame_id"] if side == "start" else None
        ),
        "end_status": "VISIBLE" if side == "end" else "NOT_VISIBLE",
        "rough_end_before_frame": (
            row["frames"][before_index]["frame_id"] if side == "end" else None
        ),
        "rough_end_after_frame": (
            row["frames"][before_index + 1]["frame_id"] if side == "end" else None
        ),
        "reason": "one side is visible",
    }


def test_coarse_manifest_adaptively_covers_context_interior_and_boundaries():
    row = build_coarse_rows([proposal()])[0]
    times = [frame["timestamp_sec"] for frame in row["frames"]]
    assert row["context_window_seconds"] == [3.0, 11.0]
    assert 2 <= len(times) <= 20
    assert times == sorted(set(times))
    assert any(value < 5.0 for value in times)
    assert any(5.0 < value < 9.0 for value in times)
    assert any(value > 9.0 for value in times)
    uniform_gap = 8.0 / 5.0
    assert min(abs(value - 5.0) for value in times if value != 5.0) < uniform_gap
    assert min(abs(value - 9.0) for value in times if value != 9.0) < uniform_gap
    assert "score" not in row and "fpn_level" not in row and "posthoc_gt" not in row


def test_fine_manifest_builds_only_visible_side_with_transition_candidates():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": visible_pair(coarse[0], "start", 4),
    }]
    first = build_fine_rows(proposals, coarse, prediction)
    second = build_fine_rows(proposals, coarse, prediction)
    assert first == second
    assert [row["boundary_side"] for row in first] == ["start"]
    row = first[0]
    times = [frame["timestamp_sec"] for frame in row["frames"]]
    assert times == sorted(set(times))
    assert 1 <= len(row["candidates"]) <= 7
    assert len({item["option"] for item in row["candidates"]}) == len(row["candidates"])
    frame_ids = [frame["frame_id"] for frame in row["frames"]]
    for candidate in row["candidates"]:
        before = frame_ids.index(candidate["before_frame_id"])
        assert frame_ids.index(candidate["after_frame_id"]) == before + 1
    assert "score" not in row and "posthoc_gt" not in row


def test_fine_grid_is_safe_near_video_boundary_and_strictly_increasing():
    proposals = [proposal(duration=1.0, segment=[0.05, 0.5])]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": visible_pair(coarse[0], "start", 0),
    }]
    rows = build_fine_rows(proposals, coarse, prediction)
    times = [frame["timestamp_sec"] for frame in rows[0]["frames"]]
    assert times == sorted(set(times))
    assert 0.0 <= times[0] < times[-1] <= 0.95


def test_fine_manifest_skips_non_valid_coarse_decision():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": {
            "action_status": "NO_ACTION",
            "start_status": "NOT_VISIBLE",
            "rough_start_before_frame": None,
            "rough_start_after_frame": None,
            "end_status": "NOT_VISIBLE",
            "rough_end_before_frame": None,
            "rough_end_after_frame": None,
            "reason": "empty",
        },
    }]
    assert build_fine_rows(proposals, coarse, prediction) == []


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
