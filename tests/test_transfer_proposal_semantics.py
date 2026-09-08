from tools.transfer_proposal_semantics import transfer_semantics


def test_transfer_uses_best_temporal_match_and_new_teacher_geometry():
    source = {
        "prompt_protocol": {"model": "vlm"},
        "entries": [
            {
                "candidate_id": "old-a",
                "video_id": "video",
                "proposal_start_sec": 0.0,
                "proposal_end_sec": 2.0,
                "semantic_label": "TAKE",
                "class_probabilities": [0.9] + [0.1 / 9] * 9,
            },
            {
                "candidate_id": "old-b",
                "video_id": "video",
                "proposal_start_sec": 3.0,
                "proposal_end_sec": 5.0,
                "semantic_label": "PUT",
                "class_probabilities": [0.1 / 9, 0.9] + [0.1 / 9] * 8,
            },
        ],
    }
    manifest = [{
        "candidate_id": "new",
        "video_id": "video",
        "absolute_feature_index": 7,
        "absolute_center_time": 3.9,
        "proposal_start_sec": 3.2,
        "proposal_end_sec": 4.8,
        "source_fpn_levels": [1],
        "teacher_score": 0.8,
    }]
    metadata = {
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {"checkpoint": "new"},
    }

    payload = transfer_semantics(manifest, source, metadata, min_iou=0.5)

    assert payload["target_gt_usage"] == "forbidden"
    assert payload["selection_uses_gt"] is False
    assert len(payload["entries"]) == 1
    entry = payload["entries"][0]
    assert entry["semantic_label"] == "PUT"
    assert entry["semantic_source_candidate_id"] == "old-b"
    assert entry["candidate_id"] == "new"
    assert entry["proposal_start_sec"] == 3.2
    assert entry["source_fpn_levels"] == [1]


def test_transfer_filters_low_confidence_semantics_before_matching():
    source = {
        "prompt_protocol": {"model": "vlm"},
        "entries": [
            {
                "candidate_id": "low",
                "video_id": "video",
                "proposal_start_sec": 0.0,
                "proposal_end_sec": 2.0,
                "semantic_label": "TAKE",
                "class_probabilities": [0.6] + [0.4 / 9] * 9,
            },
            {
                "candidate_id": "high",
                "video_id": "video",
                "proposal_start_sec": 0.1,
                "proposal_end_sec": 2.1,
                "semantic_label": "PUT",
                "class_probabilities": [0.1 / 9, 0.9] + [0.1 / 9] * 8,
            },
        ],
    }
    manifest = [{
        "candidate_id": "new",
        "video_id": "video",
        "absolute_feature_index": 1,
        "absolute_center_time": 1.0,
        "proposal_start_sec": 0.0,
        "proposal_end_sec": 2.0,
        "source_fpn_levels": [0],
    }]
    metadata = {
        "domain": "target",
        "split": "train",
        "selection_teacher_provenance": {"checkpoint": "new"},
    }

    payload = transfer_semantics(
        manifest, source, metadata, min_iou=0.5, min_confidence=0.7
    )

    assert payload["entries"][0]["semantic_label"] == "PUT"
    assert payload["entries"][0]["semantic_source_candidate_id"] == "high"
