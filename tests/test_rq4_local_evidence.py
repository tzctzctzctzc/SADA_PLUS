import numpy as np
import pytest
from tools.rq4_local_evidence import folds, labels, temporal_pool, slowfast_at, design, crossfit, POLICY
from tools.rq4_local_evidence import ap, ap_order, ordered_ap


def test_physical_video_folds_are_global():
    videos = [f'v{i}' for i in range(20)]
    assert folds(videos) == folds(videos[::-1]+videos)
    assert set(folds(videos).values()) == set(range(5))


def test_labels_use_same_class_all_instances_and_keep_negatives():
    a = [dict(label_id=2, segment=[1, 2]), dict(label_id=2, segment=[4, 5]), dict(label_id=3, segment=[7, 8])]
    y = labels(np.array([1., 2., 4., 7.]), a, 2)
    assert y.tolist() == [[1, 0], [0, 1], [1, 0], [0, 0]]
    assert not labels([0., 1.], a, 9).any()


def test_pool_keeps_temporal_order():
    x = np.arange(24).reshape(6, 4)
    feature, times = temporal_pool(x, [3, 2, 4], [0, .1, .2, .3, .4, .5])
    assert np.allclose(feature, x.reshape(3, 2, 4).mean(1))
    assert np.allclose(times, [.05, .25, .45])
    with pytest.raises(ValueError): temporal_pool(x, [3, 2, 4], [0, .1])


def test_slowfast_centers_follow_dataset_formula():
    d = dict(feat_stride=16, downsample_rate=1, num_frames=32, default_fps=30)
    assert np.allclose(slowfast_at(np.arange(6).reshape(3, 2), np.array([16/30, 32/30]), d), [[0, 1], [2, 3]])


def test_equal_capacity_controls_and_shift():
    row = dict(id='one', candidate=dict(proposal=[1, 3], class_id=2, video_id='v'),
               times=np.arange(4.), slowfast=np.arange(24).reshape(4, 6), vlm=np.arange(32).reshape(4, 8), labels=np.zeros((4, 2)))
    x, y, videos, ids = design([row])
    assert all(x[k].shape == (4, 142) for k in POLICY['conditions'] if k != 'geometry')
    assert np.allclose(x['fusion'][:, :64], x['shifted'][:, :64])
    assert not np.allclose(x['fusion'][:, 64:128], x['shifted'][:, 64:128])
    assert all(np.allclose(x[k][:, -14:], x['geometry']) for k in x)


def test_crossfit_no_video_overlap():
    rng = np.random.default_rng(6); videos = np.repeat([f'v{i}' for i in range(15)], 8)
    x = rng.normal(size=(120, 3)); y = np.column_stack((x[:, 0] > 0, x[:, 1] > 0)).astype(int)
    p, reports = crossfit(x, y, videos, folds(videos))
    assert p.shape == y.shape and np.isfinite(p).all()
    assert all(not set(r['train_videos']) & set(r['test_videos']) for r in reports)


def test_fast_ap_exact_ties_zero_weights():
    rng = np.random.default_rng(4)
    y = rng.integers(2, size=(100, 2)); p = np.round(rng.random((100, 2)), 1)
    w = rng.integers(4, size=100).astype(float)
    assert ordered_ap(ap_order(y, p), w) == pytest.approx(np.mean(ap(y, p, w)), abs=1e-12)
