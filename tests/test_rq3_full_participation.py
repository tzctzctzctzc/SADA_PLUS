from copy import deepcopy

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
    with pytest.raises(ValueError, match='original matched'):
        validate_config(candidate, 'uda')  # 尚未接线的计划不能启动旧入口。


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
