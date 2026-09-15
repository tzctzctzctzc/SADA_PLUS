import numpy as np

from tools.build_rq1_semantic_manifest import midpoint_labels
from tools.evaluate_rq1_semantics import confusion_metrics, jensen_shannon
from tools.run_rq1_slowfast_probe import segment_feature


def test_midpoint_labels_marks_overlapping_top10_actions():
    annotations = [
        {"label_id": 0, "segment": [1.0, 3.0]},
        {"label_id": 1, "segment": [1.5, 2.5]},
        {"label_id": 15, "segment": [1.0, 3.0]},
    ]
    assert midpoint_labels(annotations, 0) == [0, 1]


def test_confusion_metrics_uses_macro_classes():
    confusion = np.zeros((10, 10), dtype=np.int64)
    confusion[0, 0] = 2
    confusion[1, 0] = 1
    metrics = confusion_metrics(confusion)
    assert metrics["accuracy"] == 2 / 3
    assert metrics["macro_recall"] == 0.1


def test_jensen_shannon_is_zero_for_equal_distributions():
    assert jensen_shannon([0.2, 0.8], [0.2, 0.8]) == 0.0


def test_segment_feature_pools_feature_grids_inside_interval():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    pooled = segment_feature(features, [0.0, 1.0], fps=1.0, feat_stride=1.0, num_frames=0.0)
    assert np.allclose(pooled, np.asarray([1.0, 1.0]) / np.sqrt(2.0))
