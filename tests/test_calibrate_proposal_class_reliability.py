import pytest

from tools.calibrate_proposal_class_reliability import calibrate


def test_source_reliability_shrinks_sparse_classes_to_global_rate():
    rows = [
        {
            "status": "ok",
            "prediction": prediction,
            "class_id": class_id,
            "confidence": 0.9,
            "proposal_start_sec": 0.0,
            "proposal_end_sec": 2.0,
            "video_id": "video",
        }
        for prediction, class_id in (
            [("TAKE", 0)] * 4 + [("PUT", 1)]
        )
    ]
    database = {"video": {"annotations": [{
        "label_id": 0,
        "segment": [0.5, 1.5],
    }]}}

    result = calibrate(rows, database, shrinkage=5.0)

    assert result["global_reliability"] == pytest.approx(0.8)
    assert result["smoothed_reliabilities"][0] > 0.8
    assert result["smoothed_reliabilities"][1] < 0.8
    assert result["smoothed_reliabilities"][2] == pytest.approx(0.8)
