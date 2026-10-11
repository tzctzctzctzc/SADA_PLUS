"""RQ4离线边界接口。接口可用不等于实验参数已经冻结。"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'rq4-boundary-cache-v1'
TEACHERS = ('timelens2_8b', 'timeple_8b', 'sada_self')
SUPPORTED_SEEDS = (6, 123)  # 新实验默认6；123仅保留历史验收兼容。
REUSABLE_BASELINE_POLICY = dict(schema='sada-unified-source-patience-v1', max_epochs=50,
    min_delta=0.0, patience_epochs=15, selection_loss='source_gt_downstream_loss',
    selection_split='source_validation', state_key='ema_state_dict', validation_every_epochs=5)
CANDIDATE_KEYS = {'candidate_id', 'video_id', 'class_id', 'proposal', 'window',
                  'video_duration', 'anchor_sec', 'fpn_level'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('missing SHA256 identity')


def span(value, duration):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError('invalid interval')
    start, end = map(float, value)
    if not 0 <= start < end <= duration:
        raise ValueError('interval outside video/window')
    return [start, end]


def validate_candidate(row):
    # 白名单：候选来自冻结预测，不接收目标标签、GT区间或自由实例描述。
    if set(row) != CANDIDATE_KEYS:
        raise ValueError('candidate fields differ; annotations/descriptions forbidden')
    if not all(isinstance(row[k], str) and row[k] for k in ('candidate_id', 'video_id')):
        raise ValueError('invalid candidate identity')
    for key in ('class_id', 'fpn_level'):
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError('invalid class/level')
    duration = row['video_duration']
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError('invalid video duration')
    proposal, window = span(row['proposal'], duration), span(row['window'], duration)
    if not window[0] <= proposal[0] < proposal[1] <= window[1]:
        raise ValueError('context must contain original candidate')
    if not proposal[0] <= row['anchor_sec'] <= proposal[1]:
        raise ValueError('original anchor outside proposal')
    return row


def select_candidates(rows, fraction, seed=6):
    """整个shift按原候选身份排序取floor(N*fraction)，与教师分数无关。"""
    if not 0 < fraction <= 1 or seed not in SUPPORTED_SEEDS:
        raise ValueError('invalid candidate budget/seed')
    identities = set()
    rows = list(rows)
    for row in rows:
        validate_candidate(row)
        key = (row['video_id'], row['candidate_id'])
        if key in identities:
            raise ValueError('duplicate candidate')
        identities.add(key)
    ordered = sorted(rows, key=lambda r: (digest([seed, r['video_id'], r['candidate_id']]),
                                         r['video_id'], r['candidate_id']))
    return deepcopy(ordered[:math.floor(len(ordered) * fraction)])


def build_budget(pool, shift, seed=6):
    """正式20%预算：pool必须是单个shift、未扩展双上下文的原候选池。"""
    import re
    if shift not in {'S'+str(i) for i in range(1, 7)} or seed != 6:
        raise ValueError('formal budget requires shift S1-S6 and seed6')
    pool = sorted(deepcopy(pool), key=lambda r: (r['video_id'], r['candidate_id']))
    if any(re.search(r'-context\d+$', r['candidate_id']) for r in pool):
        raise ValueError('budget requires original candidate IDs, not context request IDs')
    selected = select_candidates(pool, .2, seed)
    result = dict(schema='rq4-shift-budget-v1', shift=shift, seed=seed,
                  fraction=.2, selection='sha256-seed-video-original-id',
                  pool=pool, pool_sha256=digest(pool), pool_count=len(pool),
                  selected_ids=[[r['video_id'], r['candidate_id']] for r in selected],
                  selected_count=len(selected), selected_sha256=digest(selected))
    result['payload_sha256'] = digest(result)
    return result


def validate_budget(budget, candidates, shift, seed):
    # 重算全池选择；不能只提供自声明的数量或SHA就声称符合20%。
    if budget != build_budget(budget['pool'], shift, seed):
        raise ValueError('budget artifact differs from recomputed shift selection')
    if digest(candidates) != budget['selected_sha256']:
        raise ValueError('cache candidates differ from fixed shift budget')
    return budget


def query_for(row, class_names):
    validate_candidate(row)
    # 教师只得到预测类别，不得到候选起止时间提示或目标标注。
    return class_names[row['class_id']]


def interval_iou(a, b):
    overlap = max(0., min(a[1], b[1]) - max(a[0], b[0]))
    return overlap / (max(a[1], b[1]) - min(a[0], b[0]))


def decode_response(teacher, raw_text, metadata, duration):
    """TimePLE只接受官方codec秒数；绝不fallback成文本数字/隐藏特征。"""
    if teacher == 'timelens2_8b':
        text = raw_text.strip()
        if text.startswith('```json') and text.endswith('```'):
            text = text[7:-3].strip()
        values = json.loads(text)
    elif teacher == 'timeple_8b':
        values = metadata.get('codec_predictions')
        if values is None:
            raise ValueError('missing official TimePLE codec predictions')
    else:
        raise ValueError('unknown native teacher')
    if not isinstance(values, list):
        raise ValueError('expected interval list')
    return [span(item, duration) for item in values]


def match_response(row, local_spans, min_iou, ambiguity_margin):
    """固定最大重叠实例匹配；门槛是显式输入，不根据target结果选择。"""
    validate_candidate(row)
    if not 0 <= min_iou <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError('invalid matching policy')
    origin, end = row['window']
    decoded = [span(s, end - origin) for s in local_spans]
    absolute = sorted(set((s[0] + origin, s[1] + origin) for s in decoded))
    ranked = sorted(((interval_iou(s, row['proposal']), s) for s in absolute), reverse=True)
    if not ranked or ranked[0][0] <= 0 or ranked[0][0] < min_iou:
        return None, 'no_match'
    if len(ranked) > 1 and ranked[0][0] - ranked[1][0] <= ambiguity_margin:
        return None, 'ambiguous'
    return list(ranked[0][1]), 'matched'


def build_cache(candidates, responses, *, teacher, shift, identity, min_iou, ambiguity_margin, seed=6,
                budget=None):
    """responses由原生适配器生成；技术错误必须上抛，格式错误单列拒绝。"""
    if teacher not in TEACHERS:
        raise ValueError('unknown teacher')
    if budget is not None:
        validate_budget(budget, candidates, shift, seed)
    by_id = {}
    for response in responses:
        key = (response['video_id'], response['candidate_id'])
        if key in by_id:
            raise ValueError('duplicate response')
        by_id[key] = response
    rows = []
    for candidate in candidates:
        validate_candidate(candidate)
        key = (candidate['video_id'], candidate['candidate_id'])
        response = by_id.pop(key)
        if response.get('candidate_sha256') != digest(candidate):
            raise ValueError('response belongs to another candidate/window')
        if response.get('teacher') != teacher:
            raise ValueError('response teacher mismatch')
        missing_native = response.get('output_status') == 'native_interval_missing'
        if missing_native and (teacher != 'timeple_8b' or '<|TIMESPAN|>' in response.get('raw_text', '')
                               or response.get('metadata', {}).get('codec_predictions') is not None):
            raise ValueError('inconsistent native abstention evidence')
        if response.get('error') and not missing_native:
            raise RuntimeError(response['error'])
        raw = response.get('raw_text', '')
        if teacher == 'sada_self':
            target, reason, decoded = list(candidate['proposal']), 'matched', []
        else:
            if teacher == 'timeple_8b' and response.get('metadata', {}).get('codec_predictions') is None and not missing_native:
                raise RuntimeError('missing TimePLE codec is an interface error, not a teacher abstention')
            try:
                decoded = decode_response(teacher, raw, response.get('metadata', {}),
                                          candidate['window'][1] - candidate['window'][0])
                target, reason = match_response(candidate, decoded, min_iou, ambiguity_margin)
            except (ValueError, TypeError):
                target, reason, decoded = None, 'invalid_output', []
        rows.append(dict(candidate=deepcopy(candidate), target=target, reason=reason,
                         raw_text=raw, decoded_local=decoded,
                         response_metadata=deepcopy(response.get('metadata', {}))))
        if missing_native:
            rows[-1]['output_status'] = 'native_interval_missing'
    if by_id:
        raise ValueError('extra teacher responses')
    payload = dict(schema=SCHEMA, teacher=teacher, shift=shift, seed=seed, target_gt_used=False,
                   identity=deepcopy(identity), candidates_sha256=digest(candidates),
                   matching=dict(min_iou=min_iou, ambiguity_margin=ambiguity_margin), records=rows)
    if budget is not None:
        payload['budget'] = deepcopy(budget)
    payload['payload_sha256'] = digest(payload)
    validate_cache(payload)
    return payload


def validate_cache(payload):
    if (payload.get('schema') != SCHEMA or payload.get('teacher') not in TEACHERS
            or payload.get('seed') not in SUPPORTED_SEEDS or payload.get('target_gt_used') is not False
            or payload.get('shift') not in {'S'+str(i) for i in range(1, 7)}):
        raise ValueError('invalid RQ4 cache identity')
    if digest({k: v for k, v in payload.items() if k != 'payload_sha256'}) != payload.get('payload_sha256'):
        raise ValueError('cache content changed')
    for key in ('proposal_checkpoint_sha256', 'teacher_weights_sha256', 'processor_sha256',
                'input_manifest_sha256', 'class_names_sha256'):
        require_sha(payload['identity'][key])
    if digest([r['candidate'] for r in payload['records']]) != payload['candidates_sha256']:
        raise ValueError('candidate set changed')
    if 'budget' in payload:
        validate_budget(payload['budget'], [r['candidate'] for r in payload['records']],
                        payload['shift'], payload['seed'])
    seen = set()
    for record in payload['records']:
        row = validate_candidate(record['candidate'])
        key = (row['video_id'], row['candidate_id'])
        if key in seen:
            raise ValueError('duplicate cached candidate')
        seen.add(key)
        if record['reason'] not in ('matched', 'no_match', 'ambiguous', 'invalid_output'):
            raise ValueError('unknown output reason')
        if (record['target'] is not None) != (record['reason'] == 'matched'):
            raise ValueError('target/rejection disagreement')
        if record['target'] is not None:
            start, end = span(record['target'], row['video_duration'])
            if not row['window'][0] <= start < end <= row['window'][1]:
                raise ValueError('teacher interval outside queried context')
        if payload['teacher'] == 'sada_self':
            expected_target, expected_reason = row['proposal'], 'matched'
        else:
            metadata = record['response_metadata']
            missing_native = record.get('output_status') == 'native_interval_missing'
            if missing_native and (payload['teacher'] != 'timeple_8b' or '<|TIMESPAN|>' in record['raw_text']
                                   or metadata.get('codec_predictions') is not None):
                raise ValueError('inconsistent native abstention evidence')
            if payload['teacher'] == 'timeple_8b' and metadata.get('codec_predictions') is None and not missing_native:
                raise ValueError('missing official codec output')
            try:
                decoded = decode_response(payload['teacher'], record['raw_text'], metadata,
                                          row['window'][1] - row['window'][0])
                if decoded != record['decoded_local']:
                    raise RuntimeError('stored decoded intervals differ from native output')
                expected_target, expected_reason = match_response(row, decoded, **payload['matching'])
            except (ValueError, TypeError):
                expected_target, expected_reason = None, 'invalid_output'
        if record['target'] != expected_target or record['reason'] != expected_reason:
            raise ValueError('cache disagrees with native decode/matching')
    return payload


def load_cache(path, expected_sha=None):
    if expected_sha is not None and file_sha(path) != expected_sha:
        raise ValueError('cache file SHA mismatch')
    return validate_cache(json.loads(Path(path).read_text(encoding='utf-8')))


def validate_gate(gate, cache):
    """接收源域冻结的评分/校准工件；不在训练器中调阈值或读取target GT。"""
    if (gate.get('schema') != 'rq4-source-gate-v1' or gate.get('fit_split') != 'source_validation'
            or gate.get('target_gt_used') is not False or gate.get('teacher_identity') != cache['identity']
            or gate.get('cache_sha256') != cache['payload_sha256']):
        raise ValueError('gate not source-calibrated or belongs to another teacher/cache')
    for key in ('source_calibration_sha256', 'scorer_code_sha256'):
        require_sha(gate[key])
    if not isinstance(gate.get('score_name'), str) or not gate['score_name']:
        raise ValueError('reliability definition must be explicit')
    threshold = gate.get('threshold')
    if threshold is not None and (not isinstance(threshold, (int, float)) or not math.isfinite(threshold)):
        raise ValueError('invalid source threshold')
    source = set(gate.get('source_video_ids', []))
    target = {r['candidate']['video_id'] for r in cache['records']}
    if not source or source & target:
        raise ValueError('source calibration membership invalid/overlaps target')
    expected = {digest(r['candidate']) for r in cache['records']}
    if set(gate['scores']) != expected or any(not isinstance(s, (int, float)) or not math.isfinite(s)
                                             for s in gate['scores'].values()):
        raise ValueError('missing/nonfinite reliability scores')
    if 'paired_evidence' in gate:
        from tools.rq4_artifacts import audit_evidence
        audit_evidence(gate['paired_evidence'], cache, gate)
    return gate


def configure(cfg, path):
    if not path:
        return
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg['rq4_boundary'] = dict(spec, config_path=str(Path(path).resolve()), config_sha256=file_sha(path))
    baseline = None
    if spec.get('baseline_config_path'):
        if file_sha(spec['baseline_config_path']) != spec.get('baseline_config_sha256'):
            raise ValueError('baseline config SHA mismatch')
        baseline = json.loads(Path(spec['baseline_config_path']).read_text())
        if baseline.get('sada_unified') != REUSABLE_BASELINE_POLICY:
            raise ValueError('baseline lacks the matched source-selection/stopping policy')
        # 仅记录已实现的同一选模/停止策略，不覆盖任何训练超参数。
        cfg['sada_unified'] = deepcopy(REUSABLE_BASELINE_POLICY)
    validate_config(cfg, cfg.get('rq2_formal', {}).get('regime'))
    if baseline is not None:
        audit_baseline_pair(baseline, cfg)


def validate_config(cfg, regime):
    spec = cfg['rq4_boundary']
    if spec.get('variant') and spec.get('strategy') not in ('relational', 'proposal_relation'):
        raise ValueError('new variant cannot masquerade as an original strategy')
    if (spec.get('schema') != 'rq4-boundary-training-v1' or spec.get('strategy') not in ('off', 'direct', 'gated', 'self', 'relational', 'proposal_relation')
            or regime != 'uda' or cfg.get('seed') not in SUPPORTED_SEEDS
            or cfg.get('init_rand_seed', cfg.get('seed')) != cfg.get('seed')
            or cfg.get('rq3_auxiliary')
            or cfg['dataset']['input_dim'] != 2304 or cfg['model'].get('use_trident_head', False)
            or cfg['loader']['batch_size'] != 2 or cfg['opt']['epochs'] != 50):
        raise ValueError('RQ4 requires matching seed6/123 batch2 SlowFast/SADA UDA normal regression head')
    if spec.get('early_stop_patience') not in (0, 15):
        raise ValueError('unsupported patience')
    if cfg['train_cfg'].get('early_stop', 0) > 0:
        raise ValueError('legacy early-stop policy must be disabled')
    if cfg.get('sada_unified') is not None:
        train = cfg['train_cfg']
        if (cfg['sada_unified'] != REUSABLE_BASELINE_POLICY
                or spec['early_stop_patience'] != 15 or train.get('valid_freq') != 5
                or not train.get('ema_active') or train.get('evaluate_student', False)):
            raise ValueError('reused baseline requires identical source-EMA stopping/selection policy')
    if spec['strategy'] == 'off':
        if spec.get('weight') != 0 or spec.get('cache_path') or spec.get('gate_path'):
            raise ValueError('matched SADA control must not load teacher/cache')
        return None
    weight = spec.get('weight')
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('set explicit positive boundary loss weight before testing')
    if spec.get('normalizer') != 'common-routable-candidates' or spec.get('activation') != 'sada-adaptation-phase':
        raise ValueError('unsupported normalization/activation')
    cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
    if spec.get('protocol_frozen') is True and 'budget' not in cache:
        raise ValueError('formal RQ4 requires a recomputable fixed shift budget')
    if cache['seed'] != cfg['seed']:
        raise ValueError('RQ4 cache/training seed mismatch')
    if cache['shift'] != cfg['rq2_formal']['shift']:
        raise ValueError('RQ4 cache shift mismatch')
    from tools.rq3_protocol import members_for
    members = set(members_for(cfg, 'tgt'))
    if any(r['candidate']['video_id'] not in members or r['candidate']['class_id'] >= 10
           for r in cache['records']):
        raise ValueError('RQ4 cache target membership/class mismatch')
    if (spec['strategy'] in ('self', 'proposal_relation')) != (cache['teacher'] == 'sada_self'):
        raise ValueError('self-boundary control cannot use teacher boundaries')
    if spec['strategy'] in ('relational', 'proposal_relation'):
        from tools.rq4_relational import validate_spec
        validate_spec(spec, cache)
    if spec['strategy'] == 'gated':
        if file_sha(spec['gate_path']) != spec['gate_file_sha256']:
            raise ValueError('gate artifact changed')
        validate_gate(json.loads(Path(spec['gate_path']).read_text(encoding='utf-8')), cache)
    return cache


def assert_launch_ready(cfg):
    spec = cfg.get('rq4_boundary')
    if spec and spec.get('protocol_frozen') is not True:
        raise ValueError('RQ4 interface draft: freeze parameters and GPU acceptance before formal training')
    if spec and spec.get('strategy') != 'off':
        cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
        if 'budget' not in cache:
            raise ValueError('formal RQ4 requires a recomputable fixed shift budget')
        from tools.rq4_artifacts import audit_evidence
        if not spec.get('evidence_path') or file_sha(spec['evidence_path']) != spec.get('evidence_file_sha256'):
            raise ValueError('formal RQ4 requires paired request/source evidence')
        proof = json.loads(Path(spec['evidence_path']).read_text())
        if spec['weight'] != .1 or spec['early_stop_patience'] != 15:
            raise ValueError('formal RQ4 differs from frozen weight/stopping policy')
        if spec['strategy'] in ('relational', 'proposal_relation'):
            from tools.rq4_relational import audit_spec
            audit_spec(spec, cache)
            return
        gate = None
        if spec['strategy'] == 'gated':
            gate = json.loads(Path(spec['gate_path']).read_text())
            if file_sha(spec['gate_path']) != spec['gate_file_sha256']:
                raise ValueError('formal gate SHA mismatch')
        audit_evidence(proof, cache, gate)


def selection_loss(losses, domain):
    """与已完成统一SADA基线同口径，不让新增辅助项参与选模。"""
    if domain != 'src':
        raise ValueError('RQ4 model selection must use source validation')
    return losses['source_gt_downstream_loss']


def audit_baseline_pair(baseline, candidate):
    """匹配种子和停止预算；不能仅因基线名称相同就视为同协议。"""
    b, c = deepcopy(baseline), deepcopy(candidate)
    bs, cs = b.pop('rq4_boundary', None), c.pop('rq4_boundary')
    reused = bs is None
    if reused:
        # 只接纳已审计的统一基线协议，不用补造RQ4配置来掩盖旧50轮预算。
        if b.get('sada_unified') != REUSABLE_BASELINE_POLICY or b.get('rq3_auxiliary'):
            raise ValueError('existing baseline is not the matched source-patience protocol')
        bs = dict(strategy='off', early_stop_patience=15)
    if (bs['strategy'] != 'off' or cs['strategy'] not in ('direct', 'gated', 'self', 'relational', 'proposal_relation')
            or bs['early_stop_patience'] != cs['early_stop_patience']
            or b['seed'] not in SUPPORTED_SEEDS or b['seed'] != c['seed']):
        raise ValueError('baseline seed/stopping budget/strategy mismatch')
    from tools.rq2_formal import build_formal_run_signature
    if build_formal_run_signature(b) != build_formal_run_signature(c):
        raise ValueError('non-auxiliary training/evaluation configuration differs')
    return dict(seed=b['seed'], stopping_budget_matched=True,
                only_boundary_auxiliary_changed=True, reused_existing_baseline=reused)


def audit_termination(checkpoint, spec):
    if spec['early_stop_patience'] == 15:
        from tools.rq3_early_stop import audit_termination as source_audit
        if 'rq4_termination' not in checkpoint:
            raise ValueError('missing RQ4 source-only stopping evidence')
        source_audit(dict(checkpoint, rq3_termination=checkpoint['rq4_termination']))
    elif checkpoint['epoch'] != 50:
        raise ValueError('fixed-budget RQ4 must complete 50 epochs')
