"""三轮验收：协议/污染，数值/梯度，入口/回归；均为CPU测试。"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import pytest
import torch
from tools import rq4_protocol as p
from tools.rq4_teachers import TimePLEAdapter, score_requests

PACKAGE = types.ModuleType('rq4_cpu')
PACKAGE.__path__ = [str(Path(__file__).resolve().parents[1] / 'libs/models/SADA')]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = importlib.util.spec_from_file_location('rq4_cpu.rq4_boundary', Path(PACKAGE.__path__[0]) / 'rq4_boundary.py')
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def candidate(**updates):
    return dict(dict(candidate_id='p0', video_id='v', class_id=1, proposal=[2., 6.],
                     window=[1., 8.], video_duration=10., anchor_sec=4., fpn_level=0), **updates)


def identity():
    return {k: 'a'*64 for k in ('proposal_checkpoint_sha256', 'teacher_weights_sha256',
                               'processor_sha256', 'input_manifest_sha256', 'class_names_sha256')}


def payload(rows=None, teacher='timelens2_8b', text='[[1.5, 4.5]]'):
    rows = rows if rows is not None else [candidate()]
    responses = [dict(candidate_id=r['candidate_id'], video_id=r['video_id'], teacher=teacher,
                      candidate_sha256=p.digest(r), raw_text=text,
                      metadata={'codec_predictions': [[1.5, 4.5]]}) for r in rows]
    return p.build_cache(rows, responses, teacher=teacher, shift='S1', identity=identity(),
                         min_iou=.1, ambiguity_margin=.05)


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return str(path)


def spec(tmp_path, cache=None, strategy='direct', threshold=.5, scores=None):
    cache = payload() if cache is None else cache
    path = tmp_path / ('cache_' + strategy + '.json')
    write(path, cache)
    result = dict(schema='rq4-boundary-training-v1', strategy=strategy, cache_path=str(path),
                  cache_file_sha256=p.file_sha(path), weight=.1, normalizer='common-routable-candidates',
                  activation='sada-adaptation-phase', early_stop_patience=15, protocol_frozen=False)
    if strategy == 'gated':
        gate = dict(schema='rq4-source-gate-v1', fit_split='source_validation', target_gt_used=False,
                    teacher_identity=cache['identity'], cache_sha256=cache['payload_sha256'],
                    source_calibration_sha256='a'*64, scorer_code_sha256='b'*64,
                    score_name='TEST_ONLY', source_video_ids=['source'], threshold=threshold,
                    scores={p.digest(r['candidate']): 1. for r in cache['records']})
        if scores is not None:
            gate['scores'] = scores
        gp = tmp_path / 'gate.json'; write(gp, gate)
        result.update(gate_path=str(gp), gate_file_sha256=p.file_sha(gp))
    return result


def inputs(crop=0, length=10, anchor_stride=1):
    v = dict(metadata=dict(video_id='v', fps=2, feat_stride=2, feat_num_frames=2,
                           crop_start=crop, crop_feat_len=length), feats=torch.zeros(2, length))
    grid = torch.tensor([[i * anchor_stride, 0, 100, anchor_stride] for i in range(length)]).float()
    offsets = torch.full((1, length, 2), .6, requires_grad=True)
    return [v], [grid], [torch.ones(1, 1, length, dtype=torch.bool)], [offsets]


# 第一轮：协议、原生解码、实例匹配、缓存身份与数据隔离。
@pytest.mark.parametrize('teacher', ['timelens2_8b', 'timeple_8b', 'sada_self'])
def test_native_cache_roundtrip(teacher):
    result = payload(teacher=teacher)
    assert p.validate_cache(result) is result
    assert result['records'][0]['target'] == ([2., 6.] if teacher == 'sada_self' else [2.5, 5.5])


@pytest.mark.parametrize('bad', ['[]', 'null', '[[-1,2]]', '[[2,2]]', '[[1,99]]', '[[NaN,3]]', 'words'])
def test_invalid_or_empty_native_output_never_fabricates_boundary(bad):
    record = payload(text=bad)['records'][0]
    assert record['target'] is None
    assert record['reason'] in ('invalid_output', 'no_match')


def test_timeple_never_falls_back_to_text():
    with pytest.raises(ValueError, match='codec'):
        p.decode_response('timeple_8b', '[[1,2]]', {}, 8)


def test_ambiguous_instance_abstains():
    assert p.match_response(candidate(), [[1, 3], [3, 5]], 0, .1) == (None, 'ambiguous')


@pytest.mark.parametrize('extra', ['target_gt', 'labels', 'segments', 'description'])
def test_candidate_annotations_forbidden(extra):
    r = candidate(); r[extra] = [1]
    with pytest.raises(ValueError): p.validate_candidate(r)


def test_budget_is_deterministic_and_teacher_independent():
    rows = [candidate(candidate_id=str(i)) for i in range(10)]
    assert p.select_candidates(rows, .2) == p.select_candidates(rows[::-1], .2)
    assert len(p.select_candidates(rows, .2)) == 2
    assert len(p.select_candidates(rows[:1], .2)) == 0


def test_cache_tampering_detected_even_with_rehashed_envelope():
    cache = payload(); cache['records'][0]['target'][0] += .1
    with pytest.raises(ValueError): p.validate_cache(cache)
    cache['payload_sha256'] = p.digest({k: v for k, v in cache.items() if k != 'payload_sha256'})
    with pytest.raises(ValueError, match='decode'): p.validate_cache(cache)


def test_cache_file_swap_detected(tmp_path):
    s = spec(tmp_path)
    Path(s['cache_path']).write_text('{}')
    with pytest.raises(ValueError, match='SHA'): M.RQ4Boundary(s)


@pytest.mark.parametrize('field,value', [('fit_split', 'target_validation'), ('source_video_ids', ['v']),
                                        ('target_gt_used', True), ('cache_sha256', 'c'*64)])
def test_gate_rejects_target_calibration_and_wrong_identity(tmp_path, field, value):
    s = spec(tmp_path, strategy='gated'); gate = json.loads(Path(s['gate_path']).read_text())
    gate[field] = value
    with pytest.raises(ValueError): p.validate_gate(gate, payload())


# 第二轮：解析数值、裁剪、分母、反向传播和无操作等价。
def test_coordinate_crop_stride_and_real_gradient(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path))
    args = inputs(crop=1)
    result = module(*args, active=True)
    assert result['accepted'] == result['routable'] == 1
    # 秒4 -> 绝对feature3.5 -> crop内2.5，梯度只能进入2/3。
    result['loss'].backward()
    gradient = args[-1][0].grad
    assert gradient[:, 2:4].abs().sum() > 0
    assert gradient[:, :2].abs().sum() == gradient[:, 4:].abs().sum() == 0
    assert result['gradient_norm'] > 0
    assert list(module.parameters()) == []


def test_perfect_boundary_has_zero_loss(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path)); args = inputs()
    args[-1][0] = torch.full((1, 10, 2), 1.5, requires_grad=True)
    assert module(*args, active=True)['loss'].item() == 0


def test_fpn_stride_scaling(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path)); args = inputs(anchor_stride=2)
    args[-1][0] = torch.full((1, 10, 2), .75, requires_grad=True)
    assert module(*args, active=True)['loss'].item() == 0


@pytest.mark.parametrize('mode', ['warmup', 'crop', 'mask', 'missing', 'anchor'])
def test_inapplicable_teacher_is_noop(tmp_path, mode):
    row = candidate(anchor_sec=2.) if mode == 'anchor' else candidate()
    module = M.RQ4Boundary(spec(tmp_path, payload([row]))); args = inputs()
    if mode == 'crop': args[0][0]['metadata']['crop_start'] = 3
    if mode == 'mask': args[2][0].fill_(False)
    if mode == 'missing': args[0][0]['metadata']['video_id'] = 'absent'
    result = module(*args, active=mode != 'warmup')
    result['loss'].backward()
    assert result['loss'] == 0 and args[-1][0].grad.abs().sum() == 0


def test_gating_does_not_renormalize_accepted_subset(tmp_path):
    rows = [candidate(), candidate(candidate_id='p1')]; cache = payload(rows)
    direct = M.RQ4Boundary(spec(tmp_path, cache))
    gate = M.RQ4Boundary(spec(tmp_path, cache, 'gated', scores={p.digest(rows[0]): 1., p.digest(rows[1]): 0.}))
    d, g = direct(*inputs(), active=True), gate(*inputs(), active=True)
    assert g['routable'] == d['routable'] == 2 and g['accepted'] == 1
    assert torch.allclose(g['loss'] * 2, d['loss'])


def test_no_source_threshold_means_no_intervention(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path, strategy='gated', threshold=None))
    result = module(*inputs(), active=True)
    assert result['routable'] == result['gate_rejected'] == 1 and result['loss'] == 0


def test_target_gt_and_logits_not_read(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path)); args = inputs()
    before = module(*args, active=True)['loss'].detach()
    args[0][0].update(labels=object(), segments=object(), logits=object())
    assert torch.equal(before, module(*args, active=True)['loss'].detach())


def test_student_optimizer_updates_localization(tmp_path):
    module = M.RQ4Boundary(spec(tmp_path)); args = inputs()
    head = torch.nn.Conv1d(3, 2, 1)
    x = torch.ones(1, 3, 10)
    optimizer = torch.optim.SGD(head.parameters(), lr=.1)
    before = head.weight.detach().clone()
    offsets = torch.nn.functional.softplus(head(x)).transpose(1, 2)
    result = module(*args[:3], [offsets], active=True)
    result['loss'].backward(); optimizer.step()
    assert not torch.equal(before, head.weight)


# 第三轮：CLI缓存构建、官方接口桥接和训练入口接线。
def test_cpu_cache_builder_cli(tmp_path):
    r = candidate(); input_path = tmp_path/'input.json'; output_path = tmp_path/'output.json'
    write(input_path, dict(candidates=[r], responses=[dict(candidate_id='p0', video_id='v',
          candidate_sha256=p.digest(r), teacher='timelens2_8b', raw_text='[[1,5]]', metadata={})],
          teacher='timelens2_8b', shift='S1', identity=identity(), min_iou=.1, ambiguity_margin=.05))
    command = [sys.executable, '-B', '-m', 'tools.build_rq4_cache', '--input', str(input_path), '--output', str(output_path)]
    subprocess.run(command, check=True, capture_output=True)
    assert p.load_cache(output_path)['records'][0]['target'] == [2., 6.]
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_official_timeple_bridge_has_empty_gt(tmp_path):
    class QwenTimePLERunner:
        prompt_cfg = {'user_prompt': '{query}'}
        generation_cfg = {'temperature': 0}
        def predict_batch(self, samples):
            assert samples[0].gt_timestamps == [] and samples[0].raw_sample == {}
            assert samples[0].query == 'PUT'
            return [types.SimpleNamespace(sample_id='p0', raw_text='<TIMESPAN>', error='',
                                          metadata={'codec_predictions': [[1, 2]]})]
    adapter = TimePLEAdapter(QwenTimePLERunner(), types.SimpleNamespace)
    clip = tmp_path/'clip.mp4'; clip.write_bytes(b'TEST_ONLY')
    request = dict(candidate=candidate(), clip_path=str(clip), clip_sha256=p.file_sha(clip),
                   clip_start_sec=1., clip_end_sec=8.)
    result = score_requests(adapter, [request], ['TAKE', 'PUT'])
    assert result[0]['teacher'] == 'timeple_8b'
    assert result[0]['metadata']['codec_predictions'] == [[1, 2]]


def test_draft_cannot_start_formal_training():
    with pytest.raises(ValueError, match='draft'):
        p.assert_launch_ready({'rq4_boundary': {'protocol_frozen': False}})
    p.assert_launch_ready({})


def test_disabled_configuration_unchanged():
    cfg = {'seed': 6}; old = copy.deepcopy(cfg); p.configure(cfg, None)
    assert cfg == old


def test_training_entry_wiring_and_legacy_untouched():
    root = Path(__file__).resolve().parents[1]
    code = (root/'libs/models/SADA/meta_archs.py').read_text()
    assert "[value[bs:] for value in out_offsets], curr_epoch >= self.warmup_adv" in code
    assert "losses['downstream_loss'] + rq4_output['loss']" in code
    assert "losses['source_gt_downstream_loss'] + rq4_output" not in code
    trainer = (root/'libs/trainer/base_trainer_da.py').read_text()
    assert "'rq4_termination' if self.cfg.get('rq4_boundary') else 'rq3_termination'" in trainer


def test_rq4_termination_uses_existing_source_only_epoch_decision():
    from tools.rq3_early_stop import decision
    assert decision([[5, 2.], [10, 1.], [15, 1.], [20, 1.], [25, 1.]])['reason'] == 'source_patience_exhausted'


def test_baseline_requires_identical_nonauxiliary_protocol():
    cfg = dict(seed=123, rq2_formal=dict(regime='uda'), loader=dict(batch_size=2),
               rq4_boundary=dict(strategy='off', early_stop_patience=15))
    auxiliary = copy.deepcopy(cfg); auxiliary['rq4_boundary']['strategy'] = 'direct'
    assert p.audit_baseline_pair(cfg, auxiliary)['stopping_budget_matched']
    auxiliary['loader']['batch_size'] = 8
    with pytest.raises(ValueError): p.audit_baseline_pair(cfg, auxiliary)
    auxiliary['loader']['batch_size'] = 2
    auxiliary['rq4_boundary']['early_stop_patience'] = 0
    with pytest.raises(ValueError): p.audit_baseline_pair(cfg, auxiliary)


def test_checkpoint_early_stop_evidence_is_not_optional():
    from tools.rq3_early_stop import decision
    history = [[5, 2.], [10, 1.], [15, 1.], [20, 1.], [25, 1.]]
    ckpt = dict(epoch=25, selection_provenance=dict(selected_epoch=10, split='source_validation', metric_value=1.),
                rq4_termination=dict(source_validation_history=history, decision=decision(history)))
    p.audit_termination(ckpt, dict(early_stop_patience=15))
    with pytest.raises(ValueError): p.audit_termination(ckpt, dict(early_stop_patience=0))
    ckpt['epoch'] = 20
    with pytest.raises(ValueError): p.audit_termination(ckpt, dict(early_stop_patience=15))


@pytest.mark.skipif(not os.environ.get('RQ4_BASE_CONFIG'), reason='requires existing SADA CPU environment/config')
def test_actual_sada_full_forward_backward_and_config_gate(tmp_path):
    from libs.models.SADA import make_meta_arch
    from tools.rq2_formal import check_formal_config
    cfg = json.loads(Path(os.environ['RQ4_BASE_CONFIG']).read_text())
    annotation = Path(cfg['dataset']['json_file'])
    if not annotation.is_absolute():
        cfg['dataset']['json_file'] = str(Path(os.environ['RQ4_DATA_ROOT']) / annotation)
    import inspect
    from libs.models.SADA.meta_archs import SADA
    assert Path(inspect.getfile(SADA)).resolve().parents[3] == Path(__file__).resolve().parents[1]
    database = json.loads(Path(cfg['dataset']['json_file']).read_text())['database']
    target_splits = {s.lower() for s in cfg['tgt_train_split']}
    target_id = next(k for k, v in database.items() if v['subset'].lower() in target_splits)
    cache = payload([candidate(video_id=target_id)])
    s = spec(tmp_path, cache)
    cfg['rq4_boundary'] = s
    p.validate_config(cfg, 'uda')
    target_manifest = cfg['rq2_formal']['target_manifest']
    check_formal_config(cfg, 'uda', target_manifest['path'] if isinstance(target_manifest, dict) else target_manifest)
    # 只用synthetic batch核对真实模型接线；不是训练实验或真实数据精度测试。
    cfg['dataset']['max_seq_len'] = 128
    cfg['model']['max_seq_len'] = 128
    torch.manual_seed(123)
    model = make_meta_arch(cfg['model_name'], cfg, device='cpu')
    model.train()
    v = dict(feats=torch.randn(2304, 128), segments=torch.tensor([[2., 6.]]), labels=torch.tensor([1]),
             metadata=dict(video_id='source', fps=2, feat_stride=2, feat_num_frames=2,
                           duration=128., crop_start=0, crop_feat_len=128))
    t = copy.deepcopy(v); t['metadata']['video_id'] = target_id
    t['segments'] = torch.empty(0, 2); t['labels'] = torch.empty(0, dtype=torch.long)
    losses = model([v, copy.deepcopy(v)], [t, copy.deepcopy(t)], mode='train',
                   curr_epoch=max(1, model.warmup_adv))[0]
    for key in ('target_gt_cls_loss', 'target_gt_reg_loss', 'target_gt_downstream_loss'):
        assert losses[key] == 0
    assert losses['rq4_accepted'] == 2 and losses['rq4_gradient_norm'] > 0
    losses['downstream_loss'].backward()
    assert model.reg_head.offset_head.conv.weight.grad.abs().sum() > 0


def test_real_training_argument_parser_accepts_rq4_config():
    import train_da
    parser = train_da.create_argparser()
    assert any(action.dest == 'rq4_config' for action in parser._actions)


def test_protocol_rejects_nonmatching_seed_and_off_control_teacher(tmp_path):
    cfg = dict(seed=6, dataset=dict(input_dim=2304), model={}, loader=dict(batch_size=2),
               opt=dict(epochs=50), train_cfg=dict(early_stop=-1), rq2_formal=dict(regime='uda'),
               rq4_boundary=spec(tmp_path))
    with pytest.raises(ValueError, match='seed123'): p.validate_config(cfg, 'uda')
    cfg['seed'] = 123
    cfg['rq4_boundary'] = dict(schema='rq4-boundary-training-v1', strategy='off', weight=0,
                               early_stop_patience=15, cache_path='wrong.json')
    with pytest.raises(ValueError, match='control'): p.validate_config(cfg, 'uda')
    cfg['rq4_boundary'].pop('cache_path')
    assert p.validate_config(cfg, 'uda') is None
