from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools import rq3_full_launch as launch
from tools.rq3_full_participation import VARIANT


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def rq4_fixture(tmp_path):
    entries = []
    for teacher, strategy, shift in sorted(launch.RQ4_KEYS):
        root = tmp_path/'formal'/teacher/shift/strategy
        cache = write(root/'cache.json', dict(teacher=teacher, shift=shift))
        write(root/'preflight.json', dict(rq4_boundary=dict(strategy=strategy, cache_path=str(cache))))
        write(root/'complete.json', dict(schema='rq4-complete-v1', teacher=teacher,
            strategy=strategy, shift=shift, code_commit='rq4-code', seed=6, batch_size=2,
            artifacts={str(cache): launch.bind(cache)['sha256']}))
        entries.append(dict(teacher=teacher, strategy=strategy, shift=shift, root=str(root)))
    return write(tmp_path/'final_audit.json', dict(status='passed', count=24, entries=entries))


def test_rq4_requires_exact_unique_24_and_real_identity(tmp_path):
    final = rq4_fixture(tmp_path)
    assert len(launch.rq4_entries(final, 'rq4-code')) == 24
    data = launch.read(final)
    bad = deepcopy(data); bad['entries'][-1] = bad['entries'][0]
    write(final, bad)
    with pytest.raises(ValueError, match='exact 24'):
        launch.rq4_entries(final, 'rq4-code')
    write(final, data)
    path = Path(data['entries'][0]['root'])/'complete.json'
    item = launch.read(path); item['teacher'] = 'other'
    write(path, item)
    with pytest.raises(ValueError, match='identity'):
        launch.rq4_entries(final, 'rq4-code')


def test_predecessor_artifact_change_invalidates_proof(tmp_path):
    final = rq4_fixture(tmp_path/'rq4')
    ref = launch.bind(write(tmp_path/'ref.json', {}))
    proof = dict(schema='rq3-full-launch-v1', status='passed', code_commit='new',
                 rq4_head='rq4-code', predecessor=launch.bind(final),
                 rq4_completed=launch.rq4_entries(final, 'rq4-code'),
                 references={s: {'complete': ref} for s in launch.SHIFTS})
    path = write(tmp_path/'proof.json', proof)
    assert launch.validate_launch_proof(path, 'new') == proof
    with pytest.raises(ValueError, match='invalid'):
        launch.validate_launch_proof(path, 'different-code')
    cache = Path(launch.read(final)['entries'][0]['root'])/'cache.json'
    cache.write_text(cache.read_text()+' ')
    with pytest.raises(ValueError, match='artifact changed'):
        launch.validate_launch_proof(path, 'new')


def smoke_fixture():
    cfg = dict(rq2_formal=dict(shift='S1', run_signature={'sha': 'signature', 'shift': 'S1'}),
               rq3_auxiliary=dict(schema=VARIANT, paired_reference={'complete':'old'}))
    values = {'target_gt_'+k+'_loss': 0. for k in ('cls', 'reg', 'downstream')}
    values.update({'rq3_'+k: 1. for k in ('candidate_count', 'fully_visible_count',
        'eligible', 'selected', 'source_loss', 'alignment_loss', 'gradient_norm', 'target_gradient_norm')})
    report = dict(status='passed', formal_result=False, real_video_features=True,
                  diagnostic_epoch=50, seed=6, batch_size=2, variant=VARIANT,
                  run_signature=cfg['rq2_formal']['run_signature'],
                  auxiliary=cfg['rq3_auxiliary'], metrics=values)
    return cfg, report


@pytest.mark.parametrize('key,value', [('target_gt_reg_loss', .1),
    ('rq3_target_gradient_norm', 0), ('rq3_source_loss', float('nan')), ('rq3_selected', 2)])
def test_real_acceptance_rejects_leakage_detach_nan_and_impossible_counts(key, value):
    cfg, report = smoke_fixture()
    launch.validate_smoke(report, cfg)
    report['metrics'][key] = value
    with pytest.raises(ValueError):
        launch.validate_smoke(report, cfg)


def test_formal_launch_cannot_use_environment_boolean_or_missing_acceptance(tmp_path, monkeypatch):
    from tools import run_rq2_matched_control as control
    from libs.trainer.da_trainer import Trainer
    cfg, _ = smoke_fixture()
    proof = dict(code_commit='new', references={'S1': cfg['rq3_auxiliary']['paired_reference']})
    monkeypatch.setattr(launch, 'validate_launch_proof', lambda _: proof)
    monkeypatch.setattr(control, 'check_code', lambda _: None)
    monkeypatch.setenv('RQ3_FULL_LAUNCH_PROOF', str(tmp_path/'proof.json'))
    monkeypatch.delenv('RQ3_FULL_ACCEPTANCE', raising=False)
    monkeypatch.setenv('RQ3_FULL_DIAGNOSTIC', '1')  # 没有该布尔逃生门。
    with pytest.raises(ValueError, match='acceptance pending'):
        launch.assert_runtime_launch(cfg)
    def diagnostic(self):
        pass
    diagnostic._rq3_full_diagnostic = True
    monkeypatch.setattr(Trainer, 'train', diagnostic)
    launch.assert_runtime_launch(cfg)
    cfg['rq3_auxiliary']['paired_reference'] = {'complete': 'another'}
    with pytest.raises(ValueError, match='another paired'):
        launch.assert_runtime_launch(cfg)


def test_acceptance_needs_six_bound_reports_and_same_config(tmp_path):
    cfg, smoke = smoke_fixture()
    proof = write(tmp_path/'proof.json', dict(code_commit='new', references={
        s: cfg['rq3_auxiliary']['paired_reference'] for s in launch.SHIFTS}))
    reports = {}
    for shift in launch.SHIFTS:
        row = deepcopy(smoke); row['run_signature']['shift'] = shift
        reports[shift] = launch.bind(write(tmp_path/(shift+'.json'), row))
    data = dict(status='passed', code_commit='new', launch_proof=launch.bind(proof),
                reports=reports)
    path = write(tmp_path/'acceptance.json', data)
    launch.acceptance_ready(path, cfg, proof)
    cfg['rq2_formal']['run_signature'] = {'sha': 'wrong'}
    with pytest.raises(ValueError, match='identity'):
        launch.acceptance_ready(path, cfg, proof)
    data['reports'].pop('S6'); write(path, data)
    with pytest.raises(ValueError, match='six-shift'):
        launch.acceptance_ready(path, cfg, proof)


def test_entry_audit_only_is_not_an_execute_flag_and_gpu_scope_is_fixed():
    # CLI层限定卡号；训练器仍要求proof+真实验收，不依赖此静态检查。
    path = Path(__file__).parents[1]/'tools/run_rq3_full_entry.py'
    text = path.read_text()
    assert "choices=('0', '1')" in text
    assert "if args.audit_only:" in text
    assert "audit_launch_reference(root, cfg, args.expected_head)" in text


def test_matrix_resolves_cli_paths_before_using_data_cwd(tmp_path, monkeypatch):
    """从独立代码目录启动时，真实前序审计仍按原数据目录读冻结相对路径。"""
    import builtins
    import sys
    from tools import run_rq3_full_matrix as matrix

    caller, data = tmp_path/'caller', tmp_path/'data_repo'
    caller.mkdir(); data.mkdir()
    annotation = data/'data/annotations.json'
    write(annotation, {'data': 'original'})
    references = {s: str(caller/(s+'.json')) for s in launch.SHIFTS}
    write(caller/'refs.json', references)
    write(caller/'rq4/final_audit.json', {'status': 'passed'})
    monkeypatch.chdir(caller)
    monkeypatch.setattr(matrix, 'check_code', lambda _: None)
    real_open = builtins.open
    monkeypatch.setattr(matrix, 'open', lambda p, *a, **k: real_open(
        tmp_path/'matrix.lock' if str(p).endswith('.rq3_qwen100_matrix.lock') else p, *a, **k),
        raising=False)
    monkeypatch.setattr(matrix.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(sys, 'argv', ['matrix', '--output', 'out', '--references', 'refs.json',
        '--rq4-root', 'rq4', '--expected-head', 'new', '--rq4-head', 'old',
        '--data-root', str(data)])

    def checked_audit(final_path, rq4_head, actual_refs, expected_head):
        assert Path.cwd() == data
        assert launch.read('data/annotations.json') == {'data': 'original'}
        assert final_path == caller/'rq4/final_audit.json'
        assert actual_refs == references
        raise RuntimeError('stop after verified CPU audit cwd')

    monkeypatch.setattr(matrix, 'create_launch_proof', checked_audit)
    with pytest.raises(RuntimeError, match='verified CPU audit cwd'):
        matrix.main()
    assert launch.read(caller/'out/state.json')['active'] == {}
    assert not (data/'out').exists()
