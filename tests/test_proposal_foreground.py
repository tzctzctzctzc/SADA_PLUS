from copy import deepcopy
import torch
from tools.proposal_foreground_math import (ProposalForeground, sample_regions, interval_labels,
    seconds_to_grid, grid_to_seconds, balanced_loss, calibration_threshold, filter_result, true_positive_flags)
from tools.proposal_foreground import source_candidates, split_source


def test_wrong_predicted_class_not_used_as_background_truth():
    boxes = torch.tensor([[2., 4.], [2.5, 3.], [8., 9.]])
    y, _ = interval_labels(boxes, torch.tensor([[2., 4.]]))
    assert y.tolist() == [2, 1, 0]


def test_old_oracle_point_zero_one_is_foreground():
    y, _ = interval_labels(torch.tensor([[0., 100.], [0., 101.]]), torch.tensor([[0., 1.]]))
    assert y.tolist() == [1, 0]


def test_subgrid_short_actions_retained():
    b = torch.tensor([[-.30, -.24], [2., 2.07]])
    assert interval_labels(b, b)[0].tolist() == [2, 2]


def test_seconds_grid_center_offset_roundtrip():
    m = dict(use_temporal_grid_feats=True, fps=60., feat_stride=16., feat_num_frames=32.)
    seconds = torch.tensor([[0., .04], [.1, .2]], dtype=torch.double)
    grid = seconds_to_grid(seconds, m)
    assert grid[0, 0] == -1.
    torch.testing.assert_close(grid_to_seconds(grid, m), seconds, rtol=0, atol=1e-12)


def test_real_linear_samples_and_no_context_replication():
    feature = torch.arange(12.).reshape(1, 12)
    valid = torch.ones(12, dtype=torch.bool)
    # 内部4..8的16个bin中最大样本7.875；左上下文最大3.875。
    x = sample_regions(feature, valid, torch.tensor([[4., 8.]]))
    torch.testing.assert_close(x[0, :3], torch.tensor([4., 7.875, -1.]))
    edge = sample_regions(feature, valid, torch.tensor([[0., 1.]]))
    assert edge[0, -3] == 0. and edge[0, -2] == 1.


def test_missing_mask_is_not_filled_with_valid_neighbor():
    f = torch.arange(8.).reshape(1, 8)
    m = torch.tensor([1, 1, 1, 0, 0, 1, 1, 1], dtype=torch.bool)
    x = sample_regions(f, m, torch.tensor([[3., 4.]]))
    assert torch.isfinite(x).all() and x[0, -2] == 0.


def test_balanced_background_partial_complete_all_receive_gradient():
    logits = torch.zeros(7, 3, requires_grad=True)
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 2])
    balanced_loss(logits, labels).backward()
    assert (logits.grad.abs().sum(1) > 0).all()
    torch.testing.assert_close(logits.grad[:5].sum(0).norm(), logits.grad[5].norm())


def test_labels_and_proposals_detached_but_fpn_has_real_gradient():
    torch.manual_seed(3)
    head = ProposalForeground(5)
    feature = torch.randn(5, 30, requires_grad=True)
    boxes = torch.tensor([[4., 10.], [15., 19.]], requires_grad=True)
    truth = torch.tensor([[4., 10.]], requires_grad=True)
    logits = head(feature, torch.ones(30, dtype=torch.bool), boxes)
    y, _ = interval_labels(boxes, truth)
    balanced_loss(logits, y).backward()
    assert feature.grad.abs().sum() > 0
    assert boxes.grad is None and truth.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_chunking_equivalent_outputs_and_gradients():
    torch.manual_seed(8)
    a = ProposalForeground(8); b = deepcopy(a)
    x = torch.randn(8, 70, requires_grad=True); z = x.detach().clone().requires_grad_()
    boxes = torch.stack([torch.arange(20.) + 5, torch.arange(20.) + 12], 1)
    m = torch.ones(70, dtype=torch.bool)
    p = a(x, m, boxes, chunk=100); q = b(z, m, boxes, chunk=4)
    torch.testing.assert_close(p, q)
    p.sum().backward(); q.sum().backward()
    torch.testing.assert_close(x.grad, z.grad)
    for u, v in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(u.grad, v.grad)


def test_delete_only_no_second_nms_or_rescore():
    result = dict(segments=torch.tensor([[1., 2.], [3., 4.], [5., 6.]]),
                  scores=torch.tensor([.8, .9, .7]), labels=torch.tensor([1, 2, 3]), video_id='v')
    out = filter_result(result, torch.tensor([.2, .95, .3]), .9)
    for key in ('segments', 'scores', 'labels'):
        assert torch.equal(out[key], result[key][[0, 2]])
    assert out['video_id'] == 'v' and len(result['scores']) == 3


def test_disabled_filter_exact_identity():
    r = dict(segments=torch.randn(6, 2), scores=torch.rand(6), labels=torch.arange(6))
    out = filter_result(r, torch.ones(6), 1.01)
    assert all(torch.equal(r[k], out[k]) for k in r)


def test_calibration_video_macro_not_dense_count():
    rows = [dict(background_probability=torch.tensor([.8]), true_positive=torch.tensor([True])),
            dict(background_probability=torch.full((1000,), .2), true_positive=torch.ones(1000, dtype=torch.bool))]
    assert calibration_threshold(rows)['threshold'] > .79


def test_no_calibration_tp_disables_filter():
    row = dict(background_probability=torch.tensor([.99]), true_positive=torch.tensor([False]))
    assert calibration_threshold([row])['threshold'] == 1.01


def test_calibration_true_positive_matching_one_to_one():
    b = torch.tensor([[2., 4.], [2., 4.], [6., 7.]])
    labels = torch.tensor([1, 1, 0])
    flag = true_positive_flags(b, labels, torch.tensor([.7, .9, .8]), torch.tensor([[2., 4.]]), torch.tensor([1]))
    assert flag.tolist() == [False, True, False]


def test_raw_candidates_never_gt_filtered_and_crop_truncation_not_pseudo_gt():
    raw = torch.tensor([[8., 9.], [1., 2.]])
    gt = torch.tensor([[-2., 3.], [2., 4.]])
    boxes = source_candidates(raw, gt, 10)
    assert torch.equal(boxes[:2], raw) and len(boxes) == 5
    assert not (boxes == torch.tensor([0., 3.])).all(1).any()


def test_video_split_disjoint_order_invariant():
    ids = ['v%d' % i for i in range(30)]
    r = split_source(ids)
    assert r == split_source(ids[::-1])
    assert not set(r['fit']) & set(r['calibration'])
    assert set(r['fit']) | set(r['calibration']) == set(ids)


def test_empty_candidates_have_valid_shape():
    head = ProposalForeground(4)
    out = head(torch.randn(4, 10), torch.ones(10, dtype=torch.bool), torch.empty(0, 2))
    assert out.shape == (0, 3)
