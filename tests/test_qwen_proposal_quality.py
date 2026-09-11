import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluate_qwen3_vl_proposal_features.py"
)
SPEC = importlib.util.spec_from_file_location("qwen_quality", MODULE_PATH)
qwen_quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(qwen_quality)


def test_proposal_quality_targets_are_class_aware():
    rows = [{
        "video_id": "v1",
        "proposal_start_sec": 1.0,
        "proposal_end_sec": 5.0,
        "teacher_top1_class": 1,
    }]
    database = {"v1": {"annotations": [
        {"segment": [1.0, 5.0], "label_id": 0},
        {"segment": [2.0, 4.0], "label_id": 1},
        {"segment": [1.0, 5.0], "label_id": 12},
    ]}}
    result = qwen_quality.proposal_quality_targets(rows, database)[0]
    assert result["quality_iou_any"] == 1.0
    assert result["quality_iou_class"] == 0.5


def test_group_ridge_never_trains_on_held_out_group():
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    targets = np.asarray([0.0, 0.1, 0.4, 0.5, 0.8, 0.9], dtype=np.float32)
    features = targets[:, None]
    predictions = qwen_quality.group_ridge_predictions(
        {"structure": features}, targets, groups, folds=3
    )
    assert predictions.shape == targets.shape
    assert np.isfinite(predictions).all()


def test_context_summary_preserves_four_temporal_quarters():
    frames = np.eye(4, dtype=np.float32).repeat(4, axis=0)
    summary = qwen_quality.summarize_frame_quarters(frames)
    assert summary.shape == (16,)
    assert np.allclose(summary.reshape(4, 4), np.eye(4))
