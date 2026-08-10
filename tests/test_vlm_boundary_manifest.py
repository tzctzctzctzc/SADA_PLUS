from tools.build_vlm_boundary_manifest import build_coarse_rows, build_fine_rows


def proposal():
    return {
        "proposal_id": "v:l0:x5:a5:c2",
        "video_id": "v",
        "video_duration_seconds": 20.0,
        "predicted_class_id": 2,
        "predicted_class_name": "WASH",
        "score": 0.93,
        "coarse_segment_seconds": [5.0, 9.0],
        "origin_stride_seconds": 0.533,
        "fpn_level": 0,
        "posthoc_gt": {"kind": "VALID_ACTION"},
    }


def test_coarse_manifest_has_dynamic_context_and_no_privileged_fields():
    row = build_coarse_rows([proposal()])[0]
    assert row["context_window_seconds"] == [3.0, 11.0]
    assert len(row["frames"]) == 16
    assert row["frames"][0]["timestamp_sec"] == 3.0
    assert row["frames"][-1]["timestamp_sec"] == 11.0
    assert "score" not in row and "fpn_level" not in row and "posthoc_gt" not in row


def test_fine_manifest_builds_two_sides_with_deterministic_shuffled_options():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": {
            "status": "VALID_ACTION",
            "rough_start_frame": "F5",
            "rough_end_frame": "F12",
            "reason": "visible",
        },
    }]
    first = build_fine_rows(proposals, coarse, prediction)
    second = build_fine_rows(proposals, coarse, prediction)
    assert first == second
    assert [row["boundary_side"] for row in first] == ["start", "end"]
    assert all(3 <= len(row["frames"]) <= 7 for row in first)
    for row in first:
        times = [frame["timestamp_sec"] for frame in row["frames"]]
        options = [frame["option"] for frame in row["frames"]]
        assert times == sorted(times)
        assert len(options) == len(set(options))
        assert "score" not in row and "posthoc_gt" not in row


def test_fine_manifest_skips_non_valid_coarse_decision():
    proposals = [proposal()]
    coarse = build_coarse_rows(proposals)
    prediction = [{
        "query_id": coarse[0]["query_id"],
        "run_status": "ok",
        "decision": {
            "status": "NO_ACTION",
            "rough_start_frame": None,
            "rough_end_frame": None,
            "reason": "empty",
        },
    }]
    assert build_fine_rows(proposals, coarse, prediction) == []
