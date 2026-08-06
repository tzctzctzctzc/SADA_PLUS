from tools.build_vlm_training_cache import build_entries


def test_deduplicated_query_expands_to_all_member_anchors():
    manifest = [
        {
            "candidate_id": "query",
            "video_id": "video",
            "sada_top1_class": 3,
            "option_mapping": {
                "A": {"role": "top1", "class_id": 3},
                "B": {"role": "top2", "class_id": 4},
                "C": {"role": "near_confusion", "class_id": 31},
            },
            "member_anchors": [
                {"fpn_level": 0, "anchor_index": 10},
                {"fpn_level": 2, "anchor_index": 3},
            ],
        }
    ]
    predictions = [
        {"candidate_id": "query", "status": "ok", "prediction": "A"}
    ]

    entries = build_entries(manifest, predictions)
    assert entries == [
        {
            "video_id": "video",
            "fpn_level": 0,
            "anchor_index": 10,
            "decision": "action",
            "class_id": 3,
        },
        {
            "video_id": "video",
            "fpn_level": 2,
            "anchor_index": 3,
            "decision": "action",
            "class_id": 3,
        },
    ]
