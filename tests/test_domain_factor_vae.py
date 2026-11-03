from copy import deepcopy
import io
import torch
import pytest
from tools.domain_factor_vae import DomainFactor, SPEC, Reverse, build_factor, class_distribution, gaussian_kl
from tools.run_domain_factor_vae import configuration, pair_check


def example(variant):
    torch.manual_seed(71)
    model = DomainFactor(16, 3, variant)
    source = [torch.randn(2, 16, 8, requires_grad=True)]
    target = [torch.randn(2, 16, 8, requires_grad=True)]
    mask = [torch.ones(2, 1, 8, dtype=torch.bool)]
    mask[0][:, :, -2:] = False
    labels = [torch.zeros(2, 8, 3)]
    labels[0][:, :3, 0] = 1.
    logits = [torch.full((2, 8, 3), -4.)]
    logits[0][:, :3, 0] = 4.
    return model, source, target, mask, labels, logits


@pytest.mark.parametrize('variant', ['ae', 'vae'])
def test_task_style_losses_and_gradient_paths(variant):
    model, source, target, masks, labels, logits = example(variant)
    out, sc = model.transform(source, masks, True)
    _, tc = model.transform(target, masks, True)
    losses = model.auxiliary(sc, tc, labels, logits, 5, 5)
    assert losses['source_fg'] == 6 and losses['target_fg'] == 6
    assert losses['aligned_classes'] == 1
    assert out[0][:, :, -2:].count_nonzero() == 0
    (losses['loss']+out[0].square().mean()).backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert source[0].grad.norm() > 0 and target[0].grad.norm() > 0
    assert losses['kl'] > 0 if variant == 'vae' else losses['kl'] == 0


def test_vae_noise_is_separate_and_checkpoint_replays_next_update():
    model, source, target, masks, labels, logits = example('vae')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    before = torch.get_rng_state().clone()
    model.transform(source, masks, True)
    assert torch.equal(before, torch.get_rng_state())
    stream = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=False)
    def update():
        optimizer.zero_grad(set_to_none=True)
        output, sc = model.transform(source, masks, True)
        _, tc = model.transform(target, masks, True)
        loss = model.auxiliary(sc, tc, labels, logits, 5, 5)['loss']+output[0].square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach().clone(), deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    first = update()
    model.load_state_dict(saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    second = update()
    assert torch.equal(first[0], second[0])
    assert all(torch.equal(value, second[1][name]) for name, value in first[1].items())
    for index, item in first[2]['state'].items():
        assert all(torch.equal(value, second[2]['state'][index][name]) for name, value in item.items())


@pytest.mark.parametrize('variant', ['ae', 'vae'])
def test_inference_ignores_domain_code_and_is_deterministic(variant):
    model, source, _, masks, _, _ = example(variant)
    before = model.sampling_state.clone()
    out, cache = model.transform(source, masks, False)
    with torch.no_grad():
        model.domain_mean.weight.add_(100.)
        model.reconstruction[0].weight.mul_(0.)
    after, _ = model.transform(source, masks, False)
    assert torch.equal(out[0], after[0]) and not cache
    assert torch.equal(before, model.sampling_state)
    assert out[0].shape == source[0].shape


def test_ae_vae_common_initialization_and_original_random_stream():
    torch.manual_seed(81)
    before = torch.get_rng_state().clone()
    ae = build_factor(dict(variant='ae'), 16, 3, 'cpu')
    vae = build_factor(dict(variant='vae'), 16, 3, 'cpu')
    assert torch.equal(before, torch.get_rng_state())
    for name, value in ae.state_dict().items():
        assert torch.equal(value, vae.state_dict()[name]), name


def test_padded_values_never_affect_auxiliary_loss():
    model, source, target, masks, labels, logits = example('ae')
    _, sc = model.transform(source, masks, True)
    _, tc = model.transform(target, masks, True)
    first = model.auxiliary(sc, tc, labels, logits, 5, 5)['loss']
    changed = source[0].detach().clone()
    changed[:, :, -2:] = 1e6
    _, sc2 = model.transform([changed], masks, True)
    second = model.auxiliary(sc2, tc, labels, logits, 5, 5)['loss']
    assert torch.equal(first, second)


def test_domain_gradient_reversal_and_conditional_kl():
    value = torch.tensor([2.], requires_grad=True)
    (Reverse.apply(value, .25)*3).sum().backward()
    assert value.grad.item() == -.75
    assert gaussian_kl(torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(4)).count_nonzero() == 0
    labels = torch.tensor([[1., 1., 0.], [0., 0., 0.]])
    assert torch.equal(class_distribution(labels), torch.tensor([[.5, .5, 0., 0.], [0., 0., 0., 1.]]))


def test_pair_preserves_source_protocol_and_target_is_unlabeled():
    original = dict(baseline=dict(code={}, tmpdir='old'),
                    model=dict(adv_method=['conditional_rev'], loss_weights=dict(weight_adv=.1)),
                    loader=dict(batch_size=2), train_cfg=dict(warmup_adv=5),
                    prototype_capacity=dict(variant='control'))
    untouched = deepcopy(original)
    ae = configuration(original, 'ae', dict(commit='new'), 'tmp')
    vae = configuration(original, 'vae', dict(commit='new'), 'tmp')
    pair_check(ae, vae)
    assert original == untouched
    assert ae['model']['adv_method'] == ['None']
    assert ae['prototype_capacity'] == original['prototype_capacity']
    vae['loader']['batch_size'] = 4
    with pytest.raises(AssertionError):
        pair_check(ae, vae)


def test_empty_foreground_cannot_create_fake_alignment():
    model, source, target, masks, labels, logits = example('vae')
    labels[0].zero_()
    logits[0].fill_(-9.)
    _, sc = model.transform(source, masks, True)
    _, tc = model.transform(target, masks, True)
    value = model.auxiliary(sc, tc, labels, logits, 1, 5)
    assert value['aligned_classes'] == 0 and value['conditional_domain'] == 0
    assert torch.isfinite(value['loss']) and value['grl_strength'] == 0
