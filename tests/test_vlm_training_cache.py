from tools.build_vlm_training_cache import build_entries
from tools.export_vlm_training_candidates import limit_queries


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


def test_query_budget_keeps_separated_highest_scores_per_video():
    rows = [
        {
            "candidate_id": "q{}".format(index),
            "video_id": "video",
            "center_sec": center,
            "sada_score_max": score,
        }
        for index, (center, score) in enumerate(
            [(1.0, 0.49), (1.4, 0.48), (3.0, 0.40), (5.0, 0.30)]
        )
    ]
    selected = limit_queries(rows, max_per_video=2, min_gap_sec=1.0)
    assert {row["center_sec"] for row in selected} == {1.0, 3.0}
