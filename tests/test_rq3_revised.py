import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import pytest
import torch

PACKAGE = types.ModuleType('rq3_revised_cpu')
PACKAGE.__path__ = [str(Path(__file__).resolve().parents[1] / 'libs/models/SADA')]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location(PACKAGE.__name__ + '.rq3_revised',
                                             Path(PACKAGE.__path__[0]) / 'rq3_revised.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def video():
    return {'metadata': {'video_id': 'v', 'crop_start': 2, 'crop_feat_len': 6,
                         'feat_stride': 2, 'feat_num_frames': 2, 'fps': 2},
            'feats': torch.zeros(4, 6)}


def entry(label=1, start=2.5, end=4.5):
    return {'class_id': label, 'proposal_start_sec': start, 'proposal_end_sec': end,
            'source_fpn_levels': [0], 'candidate_id': 'v-p0000003',
            'semantic_label': ('TAKE', 'PUT')[label],
            'class_probabilities': [1. if i == label else 0. for i in range(10)]}


def local_entry(label=1, index=3):
    return dict(entry(label, index, index + 1), absolute_feature_index=index,
                evidence_scope='anchor-local-center-only-v1')


def verify(cache=None, level=0):
    pseudo = torch.tensor([[0., 1.], [1., 0.], [0., 0.], [1., 0.], [0., 1.]])
    grid = torch.tensor([[i, 0, 9, 1] for i in range(6)], dtype=torch.float)
    valid = torch.tensor([[True, True, True, True, True, False]])
    keep, stats = MODULE.verify_sada_targets(pseudo, [video()], grid, valid,
                                             cache if cache is not None else {'v': [local_entry(), local_entry(index=6)]},
                                             level, .8)
    return pseudo, keep, stats


def test_verify_rejects_actual_foreground_without_relabel_or_background_conversion():
    pseudo, keep, stats = verify()
    # 局部证据只作用于其绝对中心3/6，即本地1/4，不向整个区间广播。
    assert keep.tolist() == [True, False, True, True, True]
    assert stats['verify_eligible'] == 2 and stats['verify_rejected'] == 1
    assert (pseudo.sum(-1).eq(0) & keep).tolist() == [False, False, True, False, False]
    features = torch.randn(5, 4, requires_grad=True)
    # 与SADA相同：按实际pseudo class拆分，在进入判别器前AND keep。
    discriminator = torch.nn.Linear(4, 1)
    loss = sum(discriminator(features[pseudo[:, c].eq(1) & keep]).square().sum() for c in range(2))
    loss.backward()
    assert features.grad[1].abs().sum() == 0
    assert features.grad[[0, 3, 4]].abs().sum() > 0


@pytest.mark.parametrize('cache,level', [({}, 0), ({'v': [local_entry()]}, 1),
                                       ({'v': [local_entry(0), local_entry(1)]}, 0)])
def test_uncovered_other_level_or_conflicting_evidence_does_not_change_baseline(cache, level):
    _, keep, _ = verify(cache, level)
    assert keep.all()


def test_verify_never_reads_target_gt():
    v = video()
    v['segments'], v['labels'] = object(), object()
    pseudo = torch.tensor([[1., 0.]])
    keep, _ = MODULE.verify_sada_targets(pseudo, [v], torch.tensor([[1., 0, 9, 1]]),
                                         torch.ones(1, 1, dtype=torch.bool), {'v': [local_entry()]}, 0, .8)
    assert not keep.item()


def test_budget_mask_controls_real_verify_rejection_and_prototype_gradient():
    _, keep, stats = verify({'v': [dict(local_entry(), participation_selected=False)]})
    assert keep.all() and stats['verify_eligible'] == 0
    _, keep, stats = verify({'v': [dict(local_entry(), participation_selected=True)]})
    assert not keep[1] and stats['verify_rejected'] == 1
    for selected in (False, True):
        model = MODULE.RQ3ProposalPrototype(4, 10,
            {'v': [dict(entry(), participation_selected=selected)]}, [1], [(0, 99)], 0.)
        out = model(*prototype_data(), epoch=5)
        assert (out['target_gradient_norm'] > 0) == selected
        assert out['selected'] == int(selected)


def test_fixed_budget_audit_requires_participation_but_not_forced_rejection(tmp_path):
    from tools.rq3_protocol import REVISION
    from tools.run_rq3_matrix import supervision_audit
    path = tmp_path/'metrics.jsonl'
    cfg = {'rq3_auxiliary': {'schema': REVISION, 'strategy': 'verify', 'participation_fraction': .2}}
    path.write_text(json.dumps({'t_rq3_verify_observed': 5}))
    assert not supervision_audit(path, cfg)['effective']
    path.write_text(json.dumps({'t_rq3_verify_observed': 5, 't_rq3_verify_eligible': 2,
                                't_rq3_verify_accepted': 2}))
    assert supervision_audit(path, cfg)['effective']
    assert not supervision_audit(path, cfg)['intervened']
    cfg['rq3_auxiliary']['strategy'] = 'prototype'
    path.write_text(json.dumps({'t_rq3_source_loss': 1, 't_rq3_gradient_norm': 1,
                                't_rq3_fully_visible_count': 2}))
    assert not supervision_audit(path, cfg)['effective']


def prototype_data():
    source = video()
    source.update(segments=torch.tensor([[0., 3.], [3., 5.]]), labels=torch.tensor([0, 1]))
    target = video()  # 没有target GT字段，完整forward必须成功。
    source_features = [torch.randn(1, 4, 6, requires_grad=True)]
    target_features = [torch.randn(1, 4, 6, requires_grad=True)]
    return [source], [target], source_features, target_features


def test_prototype_old_embedding_branch_updates_source_and_target_and_roundtrips():
    torch.manual_seed(6)
    state = torch.random.get_rng_state().clone()
    model = MODULE.RQ3ProposalPrototype(4, 10, {'v': [entry()]}, [1], [(0, 99)], .8)
    assert torch.equal(state, torch.random.get_rng_state())
    data = prototype_data()
    before = model.branch.input_projection.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    out = model(*data, epoch=5)
    assert out['selected'] == 1 and out['aligned_classes'] == 1
    assert out['source_loss'] > 0 and out['alignment_loss'] > 0
    assert out['target_gradient_norm'] > 0
    out['loss'].backward()
    assert data[2][0].grad.abs().sum() > 0 and data[3][0].grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.branch.input_projection.weight)
    restored = MODULE.RQ3ProposalPrototype(4, 10, {}, [1], [(0, 99)], .8)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.branch.source_prototype_ready[:2].all()
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_prototype_numerically_matches_historical_cotrain_objective():
    model = MODULE.RQ3ProposalPrototype(4, 10, {'v': [entry()]}, [1], [(0, 99)], .8)
    original = copy.deepcopy(model.branch)
    src, tgt, sf, tf = prototype_data()
    source = MODULE.source_gt_proposals(src, 10, 1, [(0, 99)])
    target = MODULE.cached_proposals_for_videos(model.cache, tgt, 10, 1, tf[0].device, tf[0].dtype)
    es, et = original.encode(sf, source, [1]), original.encode(tf, target, [1])
    alignment, _, _ = original.prototype_alignment(es, original.flatten_field(source, 'targets'),
                                                  et, original.select_vlm_target(target))
    expected = original.source_loss(es, source) + 0.1 * alignment
    actual = model(src, tgt, sf, tf, epoch=1)['loss']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_revised_verify_audit_does_not_require_fictitious_auxiliary_loss(tmp_path):
    from tools.rq3_protocol import REVISION
    from tools.run_rq3_matrix import supervision_audit
    path = tmp_path / 'metrics.jsonl'
    path.write_text(json.dumps({'t_rq3_verify_observed': 3, 't_rq3_verify_eligible': 3, 't_rq3_verify_accepted': 3}))
    result = supervision_audit(path, {'rq3_auxiliary': {'schema': REVISION, 'strategy': 'verify'}})
    assert result['effective'] and not result['intervened']


def test_verify_abstains_on_uncertain_evidence_and_never_broadcasts():
    weak = local_entry()
    weak['class_probabilities'] = [.099] * 10
    weak['class_probabilities'][1] = .109
    _, keep, stats = verify({'v': [weak]})
    assert keep.all() and stats['verify_uncertain'] == 1
    wide = local_entry()
    wide['proposal_end_sec'] = 8
    with pytest.raises(ValueError, match='window'):
        verify({'v': [wide]})
    with pytest.raises(ValueError, match='freshly scored'):
        verify({'v': [entry()]})


def test_prototype_rejects_partial_crop_or_low_confidence_without_target_gradient():
    for rows, threshold in (([entry(start=1.5)], .8), ([entry()], None),
                            ([dict(entry(), class_probabilities=[.1] * 10)], .1)):
        model = MODULE.RQ3ProposalPrototype(4, 10, {'v': rows}, [1], [(0, 99)], threshold)
        src, tgt, sf, tf = prototype_data()
        out = model(src, tgt, sf, tf, epoch=5)
        assert out['selected'] == 0 and out['target_gradient_norm'] == 0
        assert out['source_loss'] > 0  # 源域辅助和SADA主体不受目标拒绝影响。


def test_same_video_different_crops_do_not_share_filtered_evidence():
    first, second = video(), video()
    second['metadata']['crop_start'] = 4
    rows, total, visible = MODULE.fully_visible_cache({'v': [entry()]}, [first, second], .8)
    assert total == 2 and visible == 1
    assert len(rows[0]) == 1 and not rows[1]


def test_audit_accepts_valid_abstention_but_not_a_broken_target_gradient(tmp_path):
    from tools.rq3_protocol import REVISION
    from tools.run_rq3_matrix import supervision_audit
    path = tmp_path/'metrics.jsonl'
    row = {'t_rq3_source_loss': 1, 't_rq3_gradient_norm': .1,
           't_rq3_fully_visible_count': 3, 't_rq3_candidate_count': 4}
    cfg = {'rq3_auxiliary': {'schema': REVISION, 'strategy': 'prototype'}}
    path.write_text(json.dumps(row))
    report = supervision_audit(path, cfg)
    assert report['effective'] and not report['intervened']
    row['t_rq3_selected'] = 1
    path.write_text(json.dumps(row))
    assert not supervision_audit(path, cfg)['effective']


def test_rerun_plan_excludes_hard_and_changed_original(tmp_path):
    from tools.run_rq3_matrix import load_entry_plan
    from tools.rq2_dense_protocol import sha256_file
    original = tmp_path / 'complete.json'
    original.write_text(json.dumps(dict(teacher='egovideo_hod', strategy='verify', shift='S1', seed=6)))
    row = dict(teacher='egovideo_hod', strategy='verify', shift='S1',
               original_complete=str(original), original_complete_sha256=sha256_file(original))
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps(dict(schema='rq3-revised-completed-only-v1', entries=[row])))
    assert load_entry_plan(plan) == [('egovideo_hod', 'verify', 'S1')]
    original.write_text('{}')
    with pytest.raises(ValueError, match='identity changed'):
        load_entry_plan(plan)
    row['strategy'] = 'hard'
    plan.write_text(json.dumps(dict(schema='rq3-revised-completed-only-v1', entries=[row])))
    with pytest.raises(ValueError, match='Hard'):
        load_entry_plan(plan)


def test_proposal_evidence_uses_only_hash_bound_original_interval(tmp_path):
    from tools.rq3_protocol import content_hash, proposal_evidence
    manifest = dict(target_gt_used=False, target_video_ids=['v'], feature_lengths={'v': 6},
                    candidates=[dict(video_id='v', absolute_feature_index=3,
                                     source_fpn_levels=[0], candidate_id='v-p0000003',
                                     proposal_start_sec=2.5, proposal_end_sec=4.5)])
    payload = dict(candidate_sha256=content_hash(manifest), target_video_ids=['v'],
                   feature_lengths={'v': 6}, entries=[dict(video_id='v', absolute_feature_index=3,
                                                          source_fpn_levels=[0], class_id=1,
                                                          class_probabilities=[0., 1.] + [0.] * 8)])
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    result = proposal_evidence(payload, path)['v'][0]
    assert result['semantic_label'] == 'PUT'
    assert result['proposal_start_sec'] == 2.5 and result['proposal_end_sec'] == 4.5
    assert proposal_evidence(payload, path, [['v', 3]])['v'][0]['participation_selected']
    assert not proposal_evidence(payload, path, [])['v'][0]['participation_selected']
    with pytest.raises(ValueError, match='selected candidate'):
        proposal_evidence(payload, path, [['v', 4]])
    manifest['candidates'][0]['proposal_end_sec'] = 5.0
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='identity differs'):
        proposal_evidence(payload, path)
