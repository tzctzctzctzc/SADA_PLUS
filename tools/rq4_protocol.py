"""RQ4离线边界接口。接口可用不等于实验参数已经冻结。"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'rq4-boundary-cache-v1'
TEACHERS = ('timelens2_8b', 'timeple_8b', 'sada_self')
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


def select_candidates(rows, fraction, seed=123):
    """可选预算实现：视频内SHA排序取floor(N*fraction)，与教师分数无关。"""
    if not 0 < fraction <= 1 or seed != 123:
        raise ValueError('invalid candidate budget/seed')
    groups, identities = {}, set()
    for row in rows:
        validate_candidate(row)
        key = (row['video_id'], row['candidate_id'])
        if key in identities:
            raise ValueError('duplicate candidate')
        identities.add(key)
        groups.setdefault(row['video_id'], []).append(row)
    selected = []
    for video in sorted(groups):
        ordered = sorted(groups[video], key=lambda r: digest([seed, video, r['candidate_id']]))
        selected.extend(deepcopy(ordered[:math.floor(len(ordered) * fraction)]))
    return selected


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


def build_cache(candidates, responses, *, teacher, shift, identity, min_iou, ambiguity_margin):
    """responses由原生适配器生成；技术错误必须上抛，格式错误单列拒绝。"""
    if teacher not in TEACHERS:
        raise ValueError('unknown teacher')
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
        if response.get('error'):
            raise RuntimeError(response['error'])
        raw = response.get('raw_text', '')
        if teacher == 'sada_self':
            target, reason, decoded = list(candidate['proposal']), 'matched', []
        else:
            if teacher == 'timeple_8b' and response.get('metadata', {}).get('codec_predictions') is None:
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
    if by_id:
        raise ValueError('extra teacher responses')
    payload = dict(schema=SCHEMA, teacher=teacher, shift=shift, seed=123, target_gt_used=False,
                   identity=deepcopy(identity), candidates_sha256=digest(candidates),
                   matching=dict(min_iou=min_iou, ambiguity_margin=ambiguity_margin), records=rows)
    payload['payload_sha256'] = digest(payload)
    validate_cache(payload)
    return payload


def validate_cache(payload):
    if (payload.get('schema') != SCHEMA or payload.get('teacher') not in TEACHERS
            or payload.get('seed') != 123 or payload.get('target_gt_used') is not False
            or payload.get('shift') not in {'S'+str(i) for i in range(1, 7)}):
        raise ValueError('invalid RQ4 cache identity')
    if digest({k: v for k, v in payload.items() if k != 'payload_sha256'}) != payload.get('payload_sha256'):
        raise ValueError('cache content changed')
    for key in ('proposal_checkpoint_sha256', 'teacher_weights_sha256', 'processor_sha256',
                'input_manifest_sha256', 'class_names_sha256'):
        require_sha(payload['identity'][key])
    if digest([r['candidate'] for r in payload['records']]) != payload['candidates_sha256']:
        raise ValueError('candidate set changed')
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
            if payload['teacher'] == 'timeple_8b' and metadata.get('codec_predictions') is None:
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
    return gate


def configure(cfg, path):
    if not path:
        return
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg['rq4_boundary'] = dict(spec, config_path=str(Path(path).resolve()), config_sha256=file_sha(path))
    validate_config(cfg, cfg.get('rq2_formal', {}).get('regime'))


def validate_config(cfg, regime):
    spec = cfg['rq4_boundary']
    if (spec.get('schema') != 'rq4-boundary-training-v1' or spec.get('strategy') not in ('direct', 'gated', 'self')
            or regime != 'uda' or cfg.get('seed') != 123 or cfg.get('rq3_auxiliary')
            or cfg['dataset']['input_dim'] != 2304 or cfg['model'].get('use_trident_head', False)
            or cfg['loader']['batch_size'] != 2 or cfg['opt']['epochs'] != 50):
        raise ValueError('RQ4 requires isolated seed123 batch2 SlowFast/SADA UDA normal regression head')
    weight = spec.get('weight')
    if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
        raise ValueError('set explicit positive boundary loss weight before testing')
    if spec.get('normalizer') != 'common-routable-candidates' or spec.get('activation') != 'sada-adaptation-phase':
        raise ValueError('unsupported normalization/activation')
    cache = load_cache(spec['cache_path'], spec['cache_file_sha256'])
    if cache['shift'] != cfg['rq2_formal']['shift']:
        raise ValueError('RQ4 cache shift mismatch')
    from tools.rq3_protocol import members_for
    members = set(members_for(cfg, 'tgt'))
    if any(r['candidate']['video_id'] not in members or r['candidate']['class_id'] >= 10
           for r in cache['records']):
        raise ValueError('RQ4 cache target membership/class mismatch')
    if (spec['strategy'] == 'self') != (cache['teacher'] == 'sada_self'):
        raise ValueError('self-boundary control cannot use teacher boundaries')
    if spec['strategy'] == 'gated':
        if file_sha(spec['gate_path']) != spec['gate_file_sha256']:
            raise ValueError('gate artifact changed')
        validate_gate(json.loads(Path(spec['gate_path']).read_text(encoding='utf-8')), cache)
    if spec.get('early_stop_patience') not in (0, 15):
        raise ValueError('unsupported patience')
    if cfg['train_cfg'].get('early_stop', 0) > 0:
        raise ValueError('legacy early-stop policy must be disabled')
    return cache


def assert_launch_ready(cfg):
    spec = cfg.get('rq4_boundary')
    if spec and spec.get('protocol_frozen') is not True:
        raise ValueError('RQ4 interface draft: freeze parameters and GPU acceptance before formal training')
