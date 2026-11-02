from types import SimpleNamespace
import torch
from tools.phase_boundary import PhaseBoundary, sequence_loss, map_coordinates
from tools.phase_candidate_components import OrderedTemporalConv, refine, candidate_coordinates, source_candidates_loss, paired_source_error_loss, endpoint_mask, _device_tree, _cpu_tree


def test_foreground_joint_probability_changes_and_has_real_rejection_gradient():
    torch.manual_seed(123)
    head = PhaseBoundary(torch.randn(10, 4, 512));tokens = torch.randn(2, 24, 512)
    loss, _, scores = sequence_loss(head, tokens, torch.tensor([0, 10]))
    assert scores.shape == (2, 11)
    assert torch.allclose(scores.exp().sum(-1), torch.ones(2), atol=1e-6)
    loss.backward()
    assert head.foreground.weight.grad.norm() > 1e-5
    before = scores[:, 10].exp().detach()
    with torch.no_grad():
        head.foreground.bias.add_(5)
        after = head.sequence(tokens)[1][:, 10].exp()
    assert (after < before).all()


def test_background_sequences_train_without_fake_ordered_stage_gt():
    head = PhaseBoundary(torch.randn(10, 4, 512))
    loss, chosen, _ = sequence_loss(head, torch.randn(2, 24, 512), torch.tensor([10, 10]))
    assert chosen.shape[0] == 0 and torch.isfinite(loss)
    loss.backward()
    assert head.foreground.bias.grad.abs().sum() > 0


def test_zero_error_output_weights_choose_exact_no_change_for_both_endpoints():
    head = PhaseBoundary(torch.randn(10, 4, 512));features = torch.randn(512, 32)
    box = torch.tensor([6., 18.])
    refined, _, _, moved = refine(head, features, torch.ones(32, dtype=torch.bool), box, 0, 1, 32)
    assert torch.equal(refined, box) and moved == [0, 0]
    for side in range(2):
        points = candidate_coordinates(box, side)
        assert len(points) == 17 and points[0] == box[side]


def test_local_endpoint_model_updates_then_backpropagates_to_real_features():
    torch.manual_seed(123);head = PhaseBoundary(torch.randn(10, 4, 512));x = torch.randn(512, 32, requires_grad=True)
    box = torch.tensor([6., 18.]);points = candidate_coordinates(box, 0)
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad();x.grad = None
        risk = head.endpoint_error(x, box, points, 0, 0, 1, 32)
        loss = (torch.log1p(risk) - torch.log1p((points - 7.).abs() / 12)).square().mean()
        loss.backward();optimizer.step()
    assert x.grad.norm() > 0 and head.error_model[-1].weight.grad.norm() > 0
    clone = PhaseBoundary(torch.randn(10, 4, 512));clone.load_state_dict(head.state_dict(), strict=True)
    assert torch.equal(clone.boundary_position, head.boundary_position)


def test_source_curriculum_includes_actual_gt_gaps_and_stops_error_label_regression_gradient():
    torch.manual_seed(123);head = PhaseBoundary(torch.randn(10, 4, 512))
    f = [torch.randn(1, 512, 32, requires_grad=True)];mask = [torch.ones(1, 1, 32, dtype=torch.bool)]
    reg = [torch.full((1, 2, 32), 4., requires_grad=True)]
    videos = [dict(feats=torch.zeros(512, 32), segments=torch.tensor([[6., 16.]]), labels=torch.tensor([0]))]
    cls = [torch.full((1, 97, 32), -20.)];cls[0][0, 0, 10] = 5.
    loss, stats = source_candidates_loss(SimpleNamespace(fpn_strides=[1]), head, f, mask, reg, videos, cls)
    loss.backward()
    assert stats['candidate_background'] >= 2 and stats['candidate_error_endpoints'] > 0
    assert reg[0].grad is None
    assert head.error_model[-1].weight.grad.norm() > 0 and head.foreground.weight.grad.norm() > 0


def test_common_reference_duration_normalization_is_exact_across_crop_and_scale():
    ref = torch.tensor([4., 14.]);gt = torch.tensor([5., 13.]);pred = torch.tensor([6., 12.])
    view = dict(lo=2., ratio=1.1)
    a = (pred - gt).abs() / (ref[1] - ref[0])
    r, g, p = [map_coordinates(x, view) for x in [ref, gt, pred]]
    b = (p - g).abs() / (r[1] - r[0])
    assert torch.allclose(a, b, atol=1e-7)


def test_nested_cache_moves_preserve_real_tensors_and_metadata():
    x = dict(weak=([torch.ones(2)], [torch.tensor([True])]), views=[dict(ratio=1.1)], videos=[dict(labels=torch.tensor([2]))])
    y = _device_tree(_cpu_tree(x), 'cpu')
    assert isinstance(y['weak'], tuple) and torch.equal(y['weak'][0][0], x['weak'][0][0])
    assert y['views'] == x['views'] and torch.equal(y['videos'][0]['labels'], torch.tensor([2]))


def test_paired_calibration_uses_both_real_views_and_detaches_regression_labels():
    head = PhaseBoundary(torch.randn(10, 4, 512));model = SimpleNamespace(fpn_strides=[1])
    f = [torch.randn(1, 512, 32, requires_grad=True)];mask = [torch.ones(1, 1, 32, dtype=torch.bool)]
    cls = [torch.full((1, 97, 32), -20.)];cls[0][0, 0, 12] = 5.
    weak_reg = [torch.full((1, 2, 32), 4., requires_grad=True)]
    strong_reg = [torch.full((1, 2, 32), 3., requires_grad=True)]
    v = [dict(feats=torch.zeros(512, 32), segments=torch.tensor([[7., 17.]]), labels=torch.tensor([0]))]
    loss, stats = paired_source_error_loss(model, head, (f, mask, cls, weak_reg), (f, mask, cls, strong_reg),
                        [dict(lo=0, width=32, ratio=1., length=32, original_length=32)], v)
    assert stats['source_paired_error_endpoints'] == 2
    loss.backward()
    assert weak_reg[0].grad is None and strong_reg[0].grad is None
    assert head.error_model[-1].weight.grad.norm() > 0


def test_endpoint_masks_reject_unrepresentable_offsets_without_discarding_other_side():
    teacher = torch.tensor([.1, .1]);student = torch.tensor([.4, .4])
    assert endpoint_mask(teacher, student, torch.tensor([6., 12.]), 20, 5).tolist() == [False, True]
    assert endpoint_mask(teacher, student, torch.tensor([2., 4.]), 20, 5).tolist() == [True, False]


def test_explicit_temporal_convolution_matches_reference_values_and_gradients():
    torch.manual_seed(123)
    reference = torch.nn.Conv1d(3, 4, 3, padding=1).double()
    actual = OrderedTemporalConv(3, 4, 3, padding=1).double()
    actual.load_state_dict(reference.state_dict())
    a = torch.randn(2, 3, 7, dtype=torch.double, requires_grad=True);b = a.detach().clone().requires_grad_()
    with torch.backends.mkldnn.flags(enabled=False):
        expected = reference(a);expected.square().sum().backward()
    observed = actual(b);observed.square().sum().backward()
    assert torch.allclose(expected, observed, atol=1e-12)
    assert torch.allclose(a.grad, b.grad, atol=1e-12)
    assert torch.allclose(reference.weight.grad, actual.weight.grad, atol=1e-12)


def test_zero_initialized_error_output_cannot_invent_feature_gradients():
    for _ in range(3):
        head = PhaseBoundary(torch.randn(10, 4, 512));x = torch.randn(512, 32, requires_grad=True)
        box = torch.tensor([6., 18.]);points = candidate_coordinates(box, 0)
        head.endpoint_error(x, box, points, 0, 0, 1, 32).sum().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() == 0
