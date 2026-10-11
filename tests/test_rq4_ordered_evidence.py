import numpy as np
import pytest
from tools import rq4_ordered_evidence as e2
from tools import rq4_local_evidence as e1


def row():
    rng = np.random.default_rng(9)
    return dict(id='fixture', candidate=dict(proposal=[1., 2.], class_id=2, video_id='v'),
        times=np.arange(12)*.25, slowfast=rng.normal(size=(12, 8)).astype('float32'),
        vlm=rng.normal(size=(12, 10)).astype('float32'), labels=np.zeros((12, 2), int))


def test_real_seconds_and_edges():
    x = np.array([[0.], [1.], [3.], [10.]])
    left, right = e2.context(x, np.array([0., .25, .5, 1.5]))
    np.testing.assert_array_equal(left[:, 0], [0., 0., .5, 10.])
    np.testing.assert_array_equal(right[:, 0], [2., 3., 3., 10.])


def test_duplicate_time_not_ordered_neighbor():
    left, right = e2.context(np.array([[1.], [3.]]), [0., 0.])
    np.testing.assert_array_equal(left, [[1.], [3.]])
    np.testing.assert_array_equal(right, left)


def test_bad_times():
    with pytest.raises(ValueError): e2.context(np.ones((2, 3)), [1., 0.])


def test_static_exact_and_capacity():
    r = row(); x, y, videos, ids = e2.design([r]); old, oy, ov, oi = e1.design([r])
    for a, b in [('geometry','geometry'), ('static_sf','slowfast'), ('static_fusion','fusion')]:
        np.testing.assert_array_equal(x[a], old[b])
    assert all(v.shape == (12, 142) for k, v in x.items() if k != 'geometry')
    assert ids == oi and np.array_equal(y, oy) and np.array_equal(videos, ov)


def test_swap_changes_order_not_center_or_multiset():
    r = row(); a = e2.ordered(r['vlm'], r['times'], 64, 64)
    b = e2.ordered(r['vlm'], r['times'], 64, 64, np.ones(12, bool))
    np.testing.assert_array_equal(a[:, :16], b[:, :16])
    np.testing.assert_array_equal(a[:, 16:32], b[:, 32:48])
    np.testing.assert_array_equal(a[:, 48:], -b[:, 48:])


def test_labels_do_not_affect_features_and_deterministic():
    r = row(); a = e2.design([r])[0]; r['labels'][:] = 1; b = e2.design([r])[0]
    for k in a: np.testing.assert_array_equal(a[k], b[k])


def test_same_predictions_cannot_pass():
    rng = np.random.default_rng(12); y = np.tile([[0,1],[1,0]], (20,1))
    p = rng.random(y.shape); videos = np.repeat(np.arange(10), 4)
    result = e2.summarize(y, {k:p for k in e2.CONDITIONS}, videos, iterations=20)
    assert not result['positive_signal']
    assert all(v['delta'] == 0 and v['interval'] == [0.,0.] for v in result['comparisons'].values())


def test_shifted_preserves_local_vectors_and_common():
    x = e2.design([row()])[0]
    np.testing.assert_array_equal(x['fusion'][:, :64], x['shifted'][:, :64])
    np.testing.assert_array_equal(np.roll(x['fusion'][:, 64:128], 6, axis=0), x['shifted'][:, 64:128])
    np.testing.assert_array_equal(x['fusion'][:, 128:], x['shifted'][:, 128:])
