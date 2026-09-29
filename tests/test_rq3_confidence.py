import copy
import json

import pytest

from tools.rq3_confidence import (choose_threshold, check_time_evidence, _source_correct,
                                 calibrate, LOCAL_SCOPE, select_budget)


def test_budget_is_exact_deterministic_and_independent_of_labels():
    entries = [dict(video_id='v', absolute_feature_index=i, class_probabilities=[.8, .2],
                    labels=object()) for i in range(200)]
    expected = [['v', i] for i in range(40)]
    assert select_budget(entries) == expected
    assert select_budget(list(reversed(entries))) == expected
    assert len(select_budget(entries[:95])) == 19
    entries[0]['class_probabilities'] = [.5, .5]
    assert select_budget(entries) == [['v', i] for i in range(1, 41)]
    with pytest.raises(ValueError, match='fraction'):
        select_budget(entries, .5)


def test_budget_does_not_invent_a_class_for_top1_ties():
    entries = [dict(video_id='v', absolute_feature_index=i, class_probabilities=[.5, .5])
               for i in range(200)]
    assert select_budget(entries) == []


def test_source_precision_selects_largest_qualified_coverage_and_can_abstain():
    result = choose_threshold([.3, .5, .9, .95], [False, False, True, True], .9, 2)
    assert result == dict(threshold=.9, selected_count=2, empirical_precision=1.)
    assert choose_threshold([.9, .95], [False, True], .9, 2)['threshold'] is None
    assert choose_threshold([], [], .9, 2)['threshold'] is None


def fixture():
    cfg = {'dataset': {'default_fps': 30, 'feat_stride': 16, 'num_frames': 32}}
    row = dict(video_id='source', absolute_feature_index=10,
               proposal_start_sec=160/30, proposal_end_sec=192/30)
    manifest = dict(evidence_scope=LOCAL_SCOPE, reference_grid=dict(fps=30, feat_stride=16, num_frames=32),
                    candidates=[row])
    return cfg, row, manifest


def test_verify_time_scope_cannot_relabel_old_whole_proposal_cache():
    cfg, _, manifest = fixture()
    check_time_evidence(manifest, cfg, 'verify')
    old = copy.deepcopy(manifest)
    del old['evidence_scope']
    with pytest.raises(ValueError, match='scope'):
        check_time_evidence(old, cfg, 'verify')
    manifest['candidates'][0]['proposal_end_sec'] += 1
    with pytest.raises(ValueError, match='interval'):
        check_time_evidence(manifest, cfg, 'verify')


def test_verify_calibration_requires_single_class_not_just_top1_present():
    cfg, row, _ = fixture()
    database = {'source': {'annotations': [dict(label_id=0, segment=[0, 10])]}}
    assert _source_correct(row, {'class_id': 0}, database, cfg, 'verify')
    database['source']['annotations'].append(dict(label_id=1, segment=[0, 10]))
    assert not _source_correct(row, {'class_id': 0}, database, cfg, 'verify')


def test_prototype_calibration_counts_interval_union_not_any_overlap():
    cfg, row, _ = fixture()
    row.update(proposal_start_sec=0, proposal_end_sec=10)
    database = {'source': {'annotations': [dict(label_id=0, segment=[0, 3]), dict(label_id=0, segment=[0, 3])]}}
    assert not _source_correct(row, {'class_id': 0}, database, cfg, 'prototype')
    database['source']['annotations'].append(dict(label_id=0, segment=[3, 6]))
    assert _source_correct(row, {'class_id': 0}, database, cfg, 'prototype')


def test_target_data_cannot_be_used_for_calibration(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'calibration_domain': 'target-validation'}))
    with pytest.raises(ValueError, match='source validation'):
        calibrate({}, tmp_path/'missing.json', manifest, 'verify', .9, 20)


def test_source_membership_checked_before_loading_predictions(tmp_path):
    database = tmp_path/'annotations.json'
    database.write_text(json.dumps({'database': {
        's': {'subset': 'source_val'}, 't': {'subset': 'target_val'}}}))
    cfg = {'dataset': {'json_file': str(database)}, 'src_val_split': ['source_val'],
           'tgt_train_split': ['target_train'], 'tgt_val_split': ['target_val']}
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'calibration_domain': 'source-validation', 'target_video_ids': ['t']}))
    with pytest.raises(ValueError, match='members'):
        calibrate(cfg, tmp_path/'missing.json', manifest, 'prototype', .9, 20)


def test_calibration_replay_and_teacher_binding(tmp_path, monkeypatch):
    from tools import rq3_confidence as confidence
    protocol = dict(checkpoint_sha256={'model': 'a'}, processor_sha256='p',
                    readout='cosine', frame_protocol='eight', input_frames=8)
    report = dict(schema=confidence.SCHEMA, strategy='prototype', teacher_model='egovideo_hod',
                  shift='S1', source_cache_path='source.json', source_manifest_path='manifest.json',
                  min_precision=.9, min_support=20, threshold=.7, proposal_teacher_sha256='so',
                  input_protocol=protocol)
    path = tmp_path/'calibration.json'
    path.write_text(json.dumps(report))
    monkeypatch.setattr(confidence, 'calibrate', lambda *args: copy.deepcopy(report))
    target = dict(teacher_model='egovideo_hod', shift='S1', teacher_checkpoint_sha256='so',
                  input_protocol=copy.deepcopy(protocol))
    assert confidence.load_calibration(path, {}, target, {}, 'prototype') == report
    target['teacher_checkpoint_sha256'] = 'another-so'
    with pytest.raises(ValueError, match='SO checkpoints'):
        confidence.load_calibration(path, {}, target, {}, 'prototype')
    target['teacher_checkpoint_sha256'] = 'so'
    target['input_protocol']['processor_sha256'] = 'other-processor'
    def reject_processor(*args):
        raise ValueError('source/target scoring interface differs: processor_sha256')
    monkeypatch.setattr(confidence, 'check_relocated_processor', reject_processor)
    with pytest.raises(ValueError, match='interface differs'):
        confidence.load_calibration(path, {}, target, {}, 'prototype')
    altered = dict(report, threshold=.1)
    path.write_text(json.dumps(altered))
    with pytest.raises(ValueError, match='reproduced'):
        confidence.load_calibration(path, {}, target, {}, 'prototype')


def test_processor_relocation_requires_same_content_settings_and_runtime(monkeypatch):
    from tools import rq3_protocol as protocol
    from tools.rq3_confidence import check_relocated_processor
    names = ('score_rq3_vlm_candidates.py', 'run_qwen3_vl_classification.py',
             'extract_rq2_dense_vlm_features.py', 'extract_rq1_egovideo_features.py')
    settings = {'model_path': '/weights/official'}
    audits = {}
    def create(root, changed_code=False, versions=None):
        assets = {root + '/tools/' + name: name for name in names}
        assets['/official/tokenizer.json'] = 'tokenizer'
        if changed_code:
            assets[root + '/tools/score_rq3_vlm_candidates.py'] = 'different'
        versions = versions or {'torch': 'same'}
        digest = protocol.content_hash(dict(settings=settings, processor_files=assets,
                      runtime_versions=versions, frozen_protocol=protocol.FROZEN_PROTOCOLS['egovideo_hod'],
                      qwen_classifier='run_qwen3_vl_classification.py'))
        audits[digest] = dict(processor_files=assets, runtime_versions=versions,
                             frozen_protocol_sha256=protocol.FROZEN_PROTOCOLS['egovideo_hod'])
        return {'processor_sha256': digest}
    monkeypatch.setattr(protocol, 'validate_interface_audit', lambda value, model: audits[value['processor_sha256']])
    source, target = create('/old'), create('/new')
    assert source != target
    check_relocated_processor(source, target, 'egovideo_hod', settings)
    for changed in (create('/other', True), create('/other', versions={'torch': 'changed'})):
        with pytest.raises(ValueError, match='content or runtime'):
            check_relocated_processor(source, changed, 'egovideo_hod', settings)
    with pytest.raises(ValueError, match='settings/signature'):
        check_relocated_processor(source, target, 'egovideo_hod', {'model_path': '/other'})
