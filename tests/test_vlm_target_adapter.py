from tools.export_vlm_target_adapter_proposals import limit_per_video
from tools.run_vlm_target_adapter import proposal_timestamps
from tools.score_vlm_target_adapter import summarize


def test_proposal_budget_is_deterministic_and_video_balanced():
    rows = [
        {"candidate_id": "{}-{}".format(video, index), "video_id": video}
        for video in ("a", "b")
        for index in range(5)
    ]

    first = limit_per_video(rows, max_per_video=2, max_proposals=4, seed=7)
    second = limit_per_video(rows, max_per_video=2, max_proposals=4, seed=7)

    assert first == second
    assert len(first) == 4
    assert sum(row["video_id"] == "a" for row in first) == 2
    assert sum(row["video_id"] == "b" for row in first) == 2


def test_proposal_timestamps_have_context_and_seven_inside_frames():
    timestamps = proposal_timestamps({
        "proposal_start_sec": 2.0,
        "proposal_end_sec": 4.0,
    }, duration=10.0)

    assert len(timestamps) == 9
    assert timestamps[0] < 2.0
    assert timestamps[1] == 2.0
    assert timestamps[7] == 4.0
    assert timestamps[8] > 4.0
    assert timestamps == sorted(timestamps)


def test_source_gate_requires_precision_gain_and_coverage():
    rows = []
    for index in range(10):
        teacher_class = 0
        teacher_correct = index < 5
        prediction = "TAKE" if index < 4 else "ABSTAIN"
        rows.append({
            "teacher_top1_class": teacher_class,
            "gt_assigned_class_ids": [0] if teacher_correct else [],
            "semantic_gt_kind": "KNOWN_ACTION" if teacher_correct else "BACKGROUND",
            "prediction": prediction,
        })
    result = summarize(rows, 10, {
        "min_success_rate": 0.95,
        "min_consensus_coverage": 0.20,
        "min_precision_gain": 0.03,
        "min_background_precision": 0.80,
    })

    assert result["adapter_pass"] is True
    assert result["metrics"]["base_teacher_precision"] == 0.5
    assert result["metrics"]["consensus_precision"] == 1.0
    assert result["metrics"]["consensus_coverage"] == 0.4
