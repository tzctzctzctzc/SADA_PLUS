from libs.evaluation.metrics import MetricTracker


def test_metric_tracker_can_reset_after_updates():
    tracker = MetricTracker("loss")
    tracker.update("loss", 3.0, n=2)

    tracker.reset()

    assert tracker._data.loc["loss", "total"] == 0
    assert tracker._data.loc["loss", "counts"] == 0
    assert tracker._data.loc["loss", "average"] == 0
