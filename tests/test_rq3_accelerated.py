from copy import deepcopy
import pytest
from tools.rq3_accelerated import apply_batch, valid_batch
from tools.rq3_protocol import REVISION, audit_baseline


def test_only_batch_and_derived_budget_can_differ(monkeypatch):
    baseline = dict(seed=6, loader=dict(batch_size=2), opt=dict(epochs=50, learning_rate=.001),
                    rq2_formal=dict(regime='uda', epoch_budget=dict(batch_size=2, iterations_per_epoch=40)))
    candidate = deepcopy(baseline)
    candidate['rq3_auxiliary'] = dict(schema=REVISION, strategy='prototype')
    monkeypatch.setattr('tools.rq2_formal.build_epoch_budget', lambda cfg: dict(batch_size=cfg['loader']['batch_size'], iterations_per_epoch=10))
    apply_batch(candidate, 8)
    assert valid_batch(candidate)
    result = audit_baseline(baseline, candidate)
    assert result['batch_matched'] is False and result['only_auxiliary_changed'] is False
    candidate['opt']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='beyond'):
        audit_baseline(baseline, candidate)


def test_large_batch_requires_explicit_provenance():
    assert not valid_batch(dict(loader=dict(batch_size=8)))
    cfg = dict(loader=dict(batch_size=2), rq3_auxiliary=dict(schema=REVISION, strategy='hard'))
    with pytest.raises(ValueError):
        apply_batch(cfg, 4)


def test_original_batch_is_untouched():
    cfg = dict(loader=dict(batch_size=2))
    expected = deepcopy(cfg)
    apply_batch(cfg, 2)
    assert cfg == expected and valid_batch(cfg)


def test_evaluation_normalizes_only_authorized_batch(monkeypatch):
    from tools.rq3_accelerated import evaluation_audit_config
    from tools.rq2_formal import check_frozen_runtime_values
    frozen = dict(loader=dict(batch_size=2), opt=dict(learning_rate=.001))
    cfg = deepcopy(frozen)
    cfg.update(rq3_auxiliary=dict(schema=REVISION, strategy='prototype'), rq2_formal={})
    monkeypatch.setattr('tools.rq2_formal.build_epoch_budget', lambda cfg: {})
    apply_batch(cfg, 8)
    check_frozen_runtime_values(frozen, evaluation_audit_config(cfg))
    assert cfg['loader']['batch_size'] == 8
    cfg['opt']['learning_rate'] = .002
    with pytest.raises(ValueError, match='learning_rate'):
        check_frozen_runtime_values(frozen, evaluation_audit_config(cfg))
    cfg['rq3_auxiliary']['acceleration']['batch_size'] = 4
    with pytest.raises(ValueError, match='provenance'):
        evaluation_audit_config(cfg)
