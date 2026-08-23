from tools.diagnose_vlm_target_adapter import attach_gt_views, diagnose


def make_row(candidate_id, teacher_class, prediction, score, start, end):
    return {
        "candidate_id": candidate_id,
        "video_id": "video",
        "teacher_top1_class": teacher_class,
        "teacher_score": score,
        "prediction": prediction,
        "gt_assigned_class_ids": [teacher_class],
        "center_all_gt_class_ids": [teacher_class],
        "proposal_start_sec": start,
        "proposal_end_sec": end,
        "source_fpn_levels": [0],
    }


def test_attach_gt_views_distinguishes_center_and_interval_dominance():
    rows = [make_row("a", 0, "TAKE", 0.5, 0.0, 4.0)]
    database = {
        "video": {
            "annotations": [
                {"label_id": 0, "segment": [0.0, 1.0]},
                {"label_id": 1, "segment": [1.0, 4.0]},
            ]
        }
    }

    enriched = attach_gt_views(rows, database)[0]

    assert enriched["gt_interval_any"] == [0, 1]
    assert enriched["gt_interval_dominant"] == [1]
    assert enriched["gt_interval_all"] == [0, 1]
    assert enriched["interval_annotation_count"] == 2
    assert enriched["proposal_duration_sec"] == 4.0


def test_diagnosis_compares_consensus_with_same_size_score_control():
    rows = [
        make_row("a", 0, "TAKE", 0.4, 0.0, 1.0),
        make_row("b", 0, "PUT", 0.9, 1.0, 2.0),
    ]
    database = {
        "video": {
            "annotations": [
                {"label_id": 0, "segment": [0.0, 2.0]},
            ]
        }
    }

    result = diagnose(attach_gt_views(rows, database), random_samples=10)
    anchor = result["views"]["anchor"]

    assert anchor["consensus_count"] == 1
    assert anchor["consensus_precision"] == 1.0
    assert anchor["same_size_top_score_precision"] == 1.0
    assert anchor["consensus_gain_vs_same_size_top_score"] == 0.0
