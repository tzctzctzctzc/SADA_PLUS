from copy import deepcopy
import json
import math
import os
from pathlib import Path

import pytest

from tools.rq2_formal import build_formal_run_signature
from tools.rq3_confidence import select_budget
from tools.rq3_early_stop import POLICY
from tools.rq3_full_participation import select_all, planned_config, audit_pair, VARIANT
from tools.rq3_protocol import REVISION, REVISED_FIXED, validate_config


def fixture(shift='S1'):
    entries = [dict(video_id='v', absolute_feature_index=i, class_probabilities=[.8, .2])
               for i in range(200)]
    aux = dict(REVISED_FIXED, schema=REVISION, strategy='prototype', teacher_model='qwen3_vl_8b',
               confidence_threshold=0.0, selected_coordinates=select_budget(entries),
               calibration_path='same-source-calibration', cache_path='same-target-cache')
    if shift in ('S4', 'S5', 'S6'):
        aux['early_stopping'] = deepcopy(POLICY)
    cfg = dict(seed=6, init_rand_seed=6, dataset=dict(input_dim=2304),
               loader=dict(batch_size=2), opt=dict(epochs=50, learning_rate=.001),
               train_cfg=dict(early_stop=0), rq2_formal=dict(regime='uda', shift=shift),
               rq3_auxiliary=aux)
    return cfg, entries


def test_full_selection_preserves_old_ranking_and_does_not_read_gt():
    cfg, entries = fixture()
    snapshot = deepcopy(cfg)
    assert select_all(entries)[:40] == select_budget(entries)
    assert select_all(list(reversed(entries))) == select_all(entries)
    for entry in entries:
        entry['unused_target_gt'] = object()
    candidate = planned_config(cfg, entries)
    assert cfg == snapshot
    assert candidate['rq3_auxiliary']['schema'] == VARIANT
    assert audit_pair(cfg, candidate, entries)['new_selected_count'] == 200
    assert candidate['rq3_auxiliary']['calibration_path'] == 'same-source-calibration'
    with pytest.raises(ValueError, match='fraction'):
        select_budget(entries, 1.0)  # 旧20%接口仍然拒绝100%。
    with pytest.raises(ValueError, match='bound reference'):
        validate_config(candidate, 'uda')  # 裸计划不能冒充绑定原结果的正式配置。


def test_full_selection_keeps_tie_abstention_and_rejects_duplicate():
    _, entries = fixture()
    entries[0]['class_probabilities'] = [.5, .5]
    assert len(select_all(entries)) == 199
    for entry in entries:
        entry['class_probabilities'] = [.5, .5]
    assert select_all(entries) == []
    with pytest.raises(ValueError, match='duplicate'):
        select_all(entries + [entries[0]])


@pytest.mark.parametrize('shift', ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'])
def test_each_historical_stopping_rule_is_preserved(shift):
    cfg, entries = fixture(shift)
    candidate = planned_config(cfg, entries)
    assert candidate['rq3_auxiliary'].get('early_stopping') == cfg['rq3_auxiliary'].get('early_stopping')
    assert audit_pair(cfg, candidate, entries)['stopping_rule_matched']
    cfg['rq3_auxiliary']['early_stopping'] = None if shift >= 'S4' else deepcopy(POLICY)
    with pytest.raises(ValueError, match='stopping policy'):
        planned_config(cfg, entries)


@pytest.mark.parametrize('section,key,value', [
    ('loader', 'batch_size', 8), ('opt', 'learning_rate', .01),
    ('rq3_auxiliary', 'prototype_alignment_weight', .5),
    ('rq3_auxiliary', 'calibration_path', 'expanded-source'),
    ('rq3_auxiliary', 'cache_path', 'expanded-pool'),
    ('rq3_auxiliary', 'early_stopping', POLICY),
    ('rq3_auxiliary', 'selected_coordinates', []),
])
def test_pair_rejects_other_changes_even_with_recomputed_signature(section, key, value):
    cfg, entries = fixture()
    candidate = planned_config(cfg, entries)
    candidate[section][key] = deepcopy(value)
    candidate['rq2_formal']['run_signature'] = build_formal_run_signature(candidate)
    with pytest.raises(ValueError, match='differs'):
        audit_pair(cfg, candidate, entries)


def test_pool_truncation_and_changed_reference_selection_are_rejected():
    cfg, entries = fixture()
    cfg['rq3_auxiliary']['selected_coordinates'] = []
    with pytest.raises(ValueError, match='selection'):
        planned_config(cfg, entries)
    entries.extend(dict(video_id='v', absolute_feature_index=i, class_probabilities=[.8, .2])
                   for i in range(200, 257))
    cfg['rq3_auxiliary']['selected_coordinates'] = select_budget(entries)
    with pytest.raises(ValueError, match='truncate'):
        planned_config(cfg, entries)


def test_unready_launch_stays_blocked_and_old_disabled_path_is_unchanged():
    from tools.rq3_full_participation import assert_launch_ready, apply_full_participation
    cfg, entries = fixture()
    before = deepcopy(cfg)
    apply_full_participation(cfg, None)
    assert cfg == before
    assert_launch_ready(cfg)
    with pytest.raises(ValueError, match='acceptance pending'):
        assert_launch_ready(planned_config(cfg, entries))


def test_reference_binding_requires_same_old_config_before_conversion(tmp_path, monkeypatch):
    import json
    from tools import rq3_full_participation as full
    cfg, entries = fixture('S4')
    path = tmp_path/'preflight.json'; path.write_text(json.dumps(cfg))
    report = dict(reference_artifacts=dict(preflight=dict(path=str(path))),
                  planned_config=planned_config(cfg, entries))
    monkeypatch.setattr(full, 'prepare', lambda _: deepcopy(report))
    runtime = deepcopy(cfg); runtime['name'] = 'independent-run'
    full.apply_full_participation(runtime, 'original-complete')
    assert runtime['rq3_auxiliary']['paired_reference'] == report['reference_artifacts']
    assert audit_pair(cfg, runtime, entries)['stopping_rule_matched']
    cfg['opt']['learning_rate'] *= 2
    with pytest.raises(ValueError, match='does not match'):
        full.apply_full_participation(cfg, 'original-complete')


def test_runtime_rejects_cross_regime_unbound_or_changed_reference(tmp_path):
    from tools.rq3_full_participation import validate_runtime
    cfg, entries = fixture()
    candidate = planned_config(cfg, entries)
    with pytest.raises(ValueError, match='standalone'):
        validate_runtime(candidate, 'so')
    with pytest.raises(ValueError, match='bound reference'):
        validate_runtime(candidate, 'uda')
    path = tmp_path/'bound.json'; path.write_text('{}')
    candidate['rq3_auxiliary']['paired_reference'] = {
        key: dict(path=str(path), sha256='0'*64)
        for key in ('complete', 'preflight', 'cache', 'manifest', 'calibration')}
    with pytest.raises(ValueError, match='artifact changed'):
        validate_runtime(candidate, 'uda')


def test_real_cli_exposes_reference_without_enabling_unapproved_launch():
    import train_da
    parser = train_da.create_argparser()
    assert any(action.dest == 'rq3_full_reference' for action in parser._actions)


@pytest.mark.skipif(not os.environ.get('RQ3_FULL_REFERENCE'), reason='requires real audited reference cache')
def test_actual_sada_routes_full_schema_and_backpropagates_without_target_gt():
    import torch
    from libs.models.SADA import make_meta_arch
    from tools.rq3_full_participation import prepare
    from tools.rq3_protocol import load_cache, proposal_evidence
    from libs.models.SADA.rq3_revised import feature_interval, RQ3ProposalPrototype
    cfg = prepare(os.environ['RQ3_FULL_REFERENCE'])['planned_config']
    aux = cfg['rq3_auxiliary']
    payload, _ = load_cache(aux['cache_path'])
    evidence = proposal_evidence(payload, aux['manifest_path'], aux['selected_coordinates'])
    meta = dict(fps=cfg['dataset']['default_fps'], feat_stride=cfg['dataset']['feat_stride'],
                feat_num_frames=cfg['dataset']['num_frames'])
    candidates = [row for rows in evidence.values() for row in rows if row['participation_selected']]
    chosen = next(row for row in candidates if 0 <= feature_interval(row, meta)[0]
                  and feature_interval(row, meta)[1] - feature_interval(row, meta)[0] < 100)
    start, end = feature_interval(chosen, meta)
    # 真实cache、真实SADA类；仅feature张量合成，不能算真实视频验收。
    cfg['dataset']['max_seq_len'] = cfg['model']['max_seq_len'] = 128
    torch.manual_seed(6)
    model = make_meta_arch(cfg['model_name'], cfg, device='cpu')
    assert isinstance(model.rq3_prototype, RQ3ProposalPrototype)
    assert model.rq3_auxiliary is None and model.rq3_verify_cache is None
    model.train()
    source = dict(feats=torch.randn(2304, 128), segments=torch.tensor([[2., 6.]]),
        labels=torch.tensor([chosen['class_id']]), metadata=dict(meta, video_id='source',
        duration=10000., crop_start=0, crop_feat_len=128))
    target = dict(feats=torch.randn(2304, 128), segments=torch.empty(0, 2),
        labels=torch.empty(0, dtype=torch.long), metadata=dict(meta, video_id=chosen['video_id'],
        duration=10000., crop_start=math.floor(start), crop_feat_len=128))
    losses = model([source, deepcopy(source)], [target, deepcopy(target)], mode='train',
                   curr_epoch=max(1, model.warmup_adv))[0]
    for key in ('target_gt_cls_loss', 'target_gt_reg_loss', 'target_gt_downstream_loss'):
        assert losses[key] == 0
    assert losses['rq3_selected'] > 0 and losses['rq3_target_gradient_norm'] > 0
    assert torch.isfinite(losses['downstream_loss'])
    losses['downstream_loss'].backward()
    # 正式评估使用同一schema构建分支，保留buffer/参数以strict恢复EMA。
    clone = make_meta_arch(cfg['model_name'], cfg, device='cpu')
    clone.load_state_dict(model.state_dict(), strict=True)
    assert isinstance(clone.rq3_prototype, RQ3ProposalPrototype)


def test_old_completed_monitor_cannot_count_the_new_variant(tmp_path):
    from tools.run_rq3_trusted_monitor import audit_complete
    cfg, entries = fixture()
    (tmp_path/'preflight.json').write_text(json.dumps(planned_config(cfg, entries)))
    (tmp_path/'complete.json').write_text('{}')
    with pytest.raises(ValueError, match='another RQ3 protocol'):
        audit_complete(tmp_path/'complete.json', {})
