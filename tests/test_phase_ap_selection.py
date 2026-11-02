import pytest
from tools.phase_ap_selection import source_choice, same_predictions


def test_selection_uses_source_ap_and_earliest_tie():
    rows = [dict(epoch=e, source_map=m) for e, m in [(5, .5), (10, .7), (15, .4), (20, .7)]]
    assert source_choice(rows)['epoch'] == 10
    assert rows[0]['source_map'] == .5


def test_missing_saved_epoch_is_not_silently_ignored():
    with pytest.raises(AssertionError):
        source_choice([dict(epoch=5, source_map=.9)])


def test_prediction_identity_checks_coordinates_labels_and_scores(tmp_path):
    import numpy as np
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    data = dict(score=np.array([.4, .7]), label=np.array([0, 1]), start=np.array([1., 2.]))
    np.savez(a, **data)
    np.savez(b, **data)
    same_predictions(a, b)
    np.savez(b, **dict(data, start=np.array([1., 2.1])))
    with pytest.raises(AssertionError):
        same_predictions(a, b)
