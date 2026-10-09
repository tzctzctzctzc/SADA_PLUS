from copy import deepcopy
import json
import pytest
from tools.sada_unified import apply_policy, validate, selection_loss, audit_termination
from tools.rq3_early_stop import enabled, decision


def config(regime='so'):
    return dict(seed=6, init_rand_seed=6, rq2_formal=dict(regime=regime),
                loader=dict(batch_size=2), opt=dict(epochs=50),
                train_cfg=dict(valid_freq=5, early_stop=-1, ema_active=True))


@pytest.mark.parametrize('regime', ['so', 'uda'])
def test_same_policy_for_both_regimes(regime):
    cfg = config(regime)
    before = deepcopy(cfg)
    apply_policy(cfg)
    assert enabled(cfg)
    cfg.pop('sada_unified')
    assert cfg == before  # 不修改优化器、训练日程或对齐目标。


@pytest.mark.parametrize('section,key,value', [
    ('loader', 'batch_size', 8), ('opt', 'epochs', 25),
    ('train_cfg', 'valid_freq', 1), ('train_cfg', 'early_stop', 15),
    ('train_cfg', 'ema_active', False), ('train_cfg', 'evaluate_student', True),
    ('rq2_formal', 'regime', 'target'),
])
def test_reject_protocol_drift(section, key, value):
    cfg = config(); cfg[section][key] = value
    with pytest.raises(ValueError): apply_policy(cfg)


def test_no_legacy_or_different_seed():
    cfg = config(); cfg['seed'] = 7
    with pytest.raises(ValueError): apply_policy(cfg)
    cfg = config(); cfg['rq3_auxiliary'] = dict(early_stopping={'schema': 'old'})
    with pytest.raises(ValueError): apply_policy(cfg)


def test_selection_excludes_auxiliary_and_target():
    losses = dict(source_gt_downstream_loss=2., downstream_loss=900.)
    assert selection_loss(losses, 'src') == 2.
    with pytest.raises(ValueError): selection_loss(losses, 'tgt')


def checkpoint(history=None):
    history = history or [[5, 2.], [10, 1.], [15, 1.1], [20, 1.2], [25, 1.3]]
    cfg = config(); apply_policy(cfg)
    d = decision(history)
    return dict(config=cfg, epoch=history[-1][0],
                sada_termination=dict(source_validation_history=history, decision=d),
                selection_provenance=dict(split='source_validation', metric='validation_loss',
                    state_key='ema_state_dict', selected_epoch=d['best_epoch'], metric_value=d['best_loss']))


def test_full_first_trigger_and_maximum():
    assert audit_termination(checkpoint())
    assert audit_termination(checkpoint([[i, 1./i] for i in range(5, 51, 5)]))
    from tools.rq3_early_stop import audit_termination as dispatch
    assert dispatch(checkpoint())


@pytest.mark.parametrize('fault', ['missing', 'late', 'early', 'target', 'student', 'best', 'nonfinite'])
def test_no_false_completion(fault):
    c = checkpoint()
    if fault == 'missing': c['sada_termination']['source_validation_history'].pop(0)
    if fault == 'late': c = checkpoint([[i, 1.] for i in range(5, 31, 5)])
    if fault == 'early': c = checkpoint([[5, 2.], [10, 1.]])
    if fault == 'target': c['selection_provenance']['split'] = 'target_validation'
    if fault == 'student': c['selection_provenance']['state_key'] = 'state_dict'
    if fault == 'best': c['selection_provenance']['selected_epoch'] = 15
    if fault == 'nonfinite': c['sada_termination']['source_validation_history'][-1][1] = float('nan')
    with pytest.raises(ValueError): audit_termination(c)


def test_pairing_rejects_policy_mismatch(tmp_path):
    from tools.audit_rq2_vlm_benchmark import audit_so_uda_pair
    a, b = config('so'), config('uda')
    apply_policy(a); apply_policy(b)
    p, q = tmp_path/'so.json', tmp_path/'uda.json'
    p.write_text(json.dumps(a)); q.write_text(json.dumps(b))
    assert audit_so_uda_pair(p, q)
    b['sada_unified']['patience_epochs'] = 10
    q.write_text(json.dumps(b))
    with pytest.raises(RuntimeError): audit_so_uda_pair(p, q)


def test_gpu_guard_environment_and_cpu_isolation(tmp_path, monkeypatch):
    from tools.run_rq2_matched_control import run_command
    calls = []
    monkeypatch.setattr('tools.run_rq2_matched_control.subprocess.run', lambda *a, **kw: calls.append(kw))
    monkeypatch.setenv('RQ3_GPU_INDEX', '0'); monkeypatch.setenv('RQ3_GPU_UUID', 'external')
    monkeypatch.setenv('PYTHONPATH', '/dependency')
    run_command(['test'], tmp_path/'cpu.log', tmp_path)
    assert calls[-1]['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert 'RQ3_GPU_INDEX' not in calls[-1]['env']
    run_command(['test'], tmp_path/'gpu.log', tmp_path, 'GPU-test', 2)
    env = calls[-1]['env']
    assert env['CUDA_VISIBLE_DEVICES'] == env['RQ3_GPU_UUID'] == 'GPU-test'
    assert env['RQ3_GPU_INDEX'] == '2' and '/dependency' in env['PYTHONPATH']


def test_exact_finite_shards():
    from tools.run_sada_baseline_queue import SHARDS
    assert set(SHARDS) == {2, 3}
    assert sorted(SHARDS[2] + SHARDS[3]) == ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']


def test_output_batches_do_not_share_training_names(tmp_path):
    from tools.run_rq2_matched_control import unique_run_name
    a = unique_run_name('slowfast', 'S1', 6, 'so', tmp_path/'batch_a'/'S1')
    b = unique_run_name('slowfast', 'S1', 6, 'so', tmp_path/'batch_b'/'S1')
    assert a != b
    assert a == unique_run_name('slowfast', 'S1', 6, 'so', tmp_path/'batch_a'/'S1')
