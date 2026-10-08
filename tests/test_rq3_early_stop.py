from copy import deepcopy
import pytest
from tools.rq3_early_stop import apply_policy, enabled, decision, audit_termination
from tools.rq3_protocol import REVISION, audit_baseline


def config():
    return dict(rq3_auxiliary=dict(schema=REVISION, strategy='prototype'),
                loader=dict(batch_size=2), opt=dict(epochs=50),
                train_cfg=dict(early_stop=-1), rq2_formal=dict(regime='uda'))


def test_disabled_is_unchanged():
    cfg = config(); old = deepcopy(cfg)
    apply_policy(cfg, 0)
    assert cfg == old and not enabled(cfg)


def test_patience_counts_epochs_not_validations():
    assert decision([[5, 2.], [10, 1.], [15, 1.], [20, 1.]])['reason'] == 'running'
    assert decision([[5, 2.], [10, 1.], [15, 1.], [20, 1.], [25, 1.]])['reason'] == 'source_patience_exhausted'


def test_improvement_resets_patience():
    assert decision([[10, 1.], [20, .9], [25, 1.]])['best_epoch'] == 20
    assert decision([[10, 1.], [20, .9], [25, 1.]])['reason'] == 'running'


def test_maximum_and_nonfinite():
    assert decision([[40, 1.], [45, .9], [50, .8]])['reason'] == 'max_epochs'
    with pytest.raises(ValueError):
        decision([[10, float('nan')]])


def test_only_authorized_protocol():
    cfg = config(); apply_policy(cfg, 15); assert enabled(cfg)
    cfg['rq3_auxiliary']['early_stopping']['patience_epochs'] = 3
    with pytest.raises(ValueError): enabled(cfg)
    cfg = config(); cfg['rq3_auxiliary']['strategy'] = 'hard'
    with pytest.raises(ValueError): apply_policy(cfg, 15)


def test_termination_is_audited_not_just_short_checkpoint():
    history = [[5, 2.], [10, 1.], [15, 1.], [20, 1.], [25, 1.]]
    c = dict(epoch=25, selection_provenance=dict(selected_epoch=10, split='source_validation', metric_value=1.),
             rq3_termination=dict(source_validation_history=history, decision=decision(history)))
    assert audit_termination(c)
    c['epoch'] = 20
    with pytest.raises(ValueError): audit_termination(c)
    c['epoch'] = 25; c['selection_provenance']['split'] = 'target_validation'
    with pytest.raises(ValueError): audit_termination(c)


def test_pairing_marks_changed_stop_budget():
    cfg = config(); baseline = deepcopy(cfg); baseline.pop('rq3_auxiliary')
    apply_policy(cfg, 15)
    pairing = audit_baseline(baseline, cfg)
    assert pairing['stopping_budget_matched'] is False
    assert pairing['only_auxiliary_changed'] is False
