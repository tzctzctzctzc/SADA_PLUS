"""一次刷新后的真实VLM缓存；GT不参与候选、评分、名单或超参数选择。"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from tools.baseline_protocol import read, save, sha, POLICY as STOP_POLICY

SCHEMA = 'rq3-prototype-single-refresh-v1'
CACHE_SCHEMA = 'prototype-refreshed-vlm-cache-v1'
SHIFTS = tuple('S'+str(i) for i in range(1, 7))
DIAGNOSTIC = Path('/vepfs/vlms/candidate_refresh_offline_c82256c_1030')
BASELINE = Path('/vepfs/vlms/sada_baseline12_seed123_b2_08e7c24_1029')
CONTROL = Path('/vepfs/vlms/prototype_capacity12_5f9a157_1029')
LEGACY = Path('/vepfs/vlms/SADA_PLUS_worktrees/rq3-enhance-resume-final-1029')
MODEL = Path('/vepfs/vlms/cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
POLICY = dict(rounds=1, shifts=list(SHIFTS), teacher='qwen3_vl_8b',
    proposal_model='previous_prototype_source_selected_best_EMA', candidate_pool=200,
    candidate_selection='unchanged_sha256_seed6_per_video30_global200',
    frame_protocol='proposal-interior-eight-midpoints-v1', frames=8,
    input_mode='native_video', prompt_mode='baseline', score_temperature=1.,
    min_pixels=224*224, max_pixels=448*448, uniform_mixed_frame_grid=True,
    readout='closed-class-single-token-log-likelihood', participation_fraction=.2,
    participation='highest_max_class_probability_skip_top1_ties_no_refill',
    student_initialization='fresh_same_seed123_detector_and_branch_as_control',
    seed=123, batch_size=2, variant='control', target_gt_used=False,
    source_selected_early_stopping=STOP_POLICY,
    additional_pipeline_cost='previous_student_training_plus_1200_VLM_queries_report_separately')


def selected_coordinates(entries):
    """与原rq3_confidence.select_budget逐值一致。"""
    ranked = [x for x in entries if x['class_probabilities'].count(max(x['class_probabilities'])) == 1]
    ranked.sort(key=lambda x: (-max(x['class_probabilities']), x['video_id'], x['absolute_feature_index']))
    return [[x['video_id'], x['absolute_feature_index']] for x in ranked[:len(entries)//5]]


def sanitized_rows(pool, members):
    """只转交定位输出；不把诊断GT或学生类别猜测送给VLM。"""
    if len(pool) != 200 or len({x['candidate_id'] for x in pool}) != 200:
        raise ValueError('冻结候选必须精确200个唯一身份')
    output = []
    for x in pool:
        start, end = x['interval']
        if (x['video_id'] not in members or x['level'] not in (0, 1, 2)
                or not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end
                or x['candidate_id'] != f"{x['video_id']}-p{x['absolute_feature_index']:07d}"):
            raise ValueError('候选成员、坐标或区间非法')
        output.append(dict(candidate_id=x['candidate_id'], video_id=x['video_id'],
            absolute_feature_index=x['absolute_feature_index'], source_fpn_levels=[x['level']],
            proposal_start_sec=start, proposal_end_sec=end))
    return output


def prepare(root, video_root, recovery_root):
    from tools.refresh_frames import video_paths_by_id, prepare_video_candidates
    diag = read(DIAGNOSTIC/'final_audit.json')
    if diag['status'] != 'passed' or diag['count'] != 18:
        raise ValueError('前序18模型离线推理不完整')
    recovery_hash = sha(recovery_root/'manifest.json')
    if recovery_hash != '5a8041d6355e1e5b9d9f1e268f903c58dd604efa906d780f69268d8996483b90':
        raise ValueError('精确恢复帧来源改变')
    video_paths = video_paths_by_id(video_root)
    summaries = {}
    for s in SHIFTS:
        e = DIAGNOSTIC/'entries'/(s+'_prototype')
        if sha(e/'complete.json') != diag['completes'][str(e)]:
            raise ValueError('前序推理完成凭证改变')
        done, job = read(e/'inference_complete.json'), read(e/'job.json')
        complete = read(e/'complete.json')
        if (sha(e/'inference_complete.json') != complete['inference_sha256']
                or sha(e/'job.json') != done['job_sha256'] or job['reference'] != str(CONTROL/'formal/control'/s)
                or not done['target_gt_input_empty'] or not done['no_training']):
            raise ValueError('前序候选生成来源或GT隔离不符')
        for name, digest in done['outputs'].items():
            if sha(e/name) != digest: raise ValueError('冻结原始预测改变')
        for name, digest in job['reference_artifacts'].items():
            if sha(Path(job['reference'])/name) != digest: raise ValueError('候选生成模型改变')
        cfg = read(CONTROL/'formal/control'/s/'config.json')
        if sha(cfg['baseline']['target_manifest']) != job['target_manifest_sha256']:
            raise ValueError('目标训练成员改变')
        members = read(cfg['baseline']['target_manifest'])['database']
        rows = sanitized_rows(read(e/'selected_pool.json'), members)
        grouped = defaultdict(list)
        for row in rows: grouped[row['video_id']].append(row)
        destination = root/'candidates'/s
        destination.mkdir(parents=True, exist_ok=False)
        frames, rejected = [], 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = [pool.submit(prepare_video_candidates, v, rr, video_paths, recovery_root, destination)
                    for v, rr in grouped.items()]
            for future in jobs:
                out, short = future.result(); frames += out; rejected += short
        if rejected or len(frames) != 200:
            raise ValueError('新候选无法保持200条精确八帧输入；保留现场，不补抽')
        save(destination/'manifest.json', dict(schema=SCHEMA, shift=s, target_gt_used=False,
            frame_protocol=POLICY['frame_protocol'], target_video_ids=sorted(members),
            candidates=frames, selected_pool_sha256=sha(e/'selected_pool.json'),
            checkpoint_sha256=job['reference_artifacts']['best.pt'],
            reference=str(CONTROL/'formal/control'/s), original_inference_sha256=sha(e/'inference_complete.json'),
            recovery_manifest_sha256=recovery_hash))
        summaries[s] = dict(manifest=str(destination/'manifest.json'), sha256=sha(destination/'manifest.json'),
                            candidate_count=len(frames))
        print('REFRESH_FRAMES_READY', s, len(frames), flush=True)
    save(root/'candidate_audit.json', dict(status='passed', count=1200, shifts=summaries,
        diagnostic_final_sha256=sha(DIAGNOSTIC/'final_audit.json'), policy=POLICY))


def freeze_scores(root):
    from tools.qwen_source_interface import build_prompt, normalize_digit_logits, digit_to_label, LABELS
    proof = read(root/'teacher/inference_complete.json')
    predictions_path = root/'teacher/predictions.jsonl'
    if proof['status'] != 'passed' or proof['count'] != 1200 or sha(predictions_path) != proof['predictions_sha256']:
        raise ValueError('1200项真实评分不完整')
    if sha(root/'candidate_audit.json') != proof['candidate_audit_sha256']:
        raise ValueError('评分候选身份改变')
    if proof.get('policy') != POLICY or proof.get('target_gt_used') is not False:
        raise ValueError('教师评分协议或GT隔离不符')
    for name,key in [('model_identity.json','model_identity_sha256'),('device_map.json','device_map_sha256')]:
        if sha(root/'teacher'/name) != proof[key]: raise ValueError('真实VLM身份/放置证明改变')
    if sha(root/'teacher/first_cuda_four.json') != proof['guard_sha256']:
        raise ValueError('真实四卡首CUDA证明改变')
    audit = read(root/'candidate_audit.json')
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    by = {(x['shift'], x['candidate_id']): x for x in rows}
    if len(by) != 1200: raise ValueError('教师评分身份重复')
    prepared = {}
    for s in SHIFTS:
        manifest_path = Path(audit['shifts'][s]['manifest'])
        if sha(manifest_path) != audit['shifts'][s]['sha256']: raise ValueError('八帧输入改变')
        manifest = read(manifest_path); entries = []
        for candidate in manifest['candidates']:
            p = by[s, candidate['candidate_id']]
            prompt = build_prompt(candidate['candidate_id'], 8, 'baseline', 'native_video')
            if (p['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()
                    or p['frame_sha256'] != candidate['frame_sha256']):
                raise ValueError('评分提示/帧与冻结候选不一致')
            numerical = normalize_digit_logits(p['digit_logits'], digit_to_label(candidate['candidate_id']))
            if numerical['class_probabilities'] != p['class_probabilities'] or numerical['class_id'] != p['class_id']:
                raise ValueError('单token概率无法从原始logits复算')
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in p['class_probabilities']):
                raise ValueError('教师概率非法')
            entries.append(dict(candidate, class_probabilities=p['class_probabilities'], class_id=p['class_id']))
        chosen = {tuple(x) for x in selected_coordinates(entries)}
        grouped = defaultdict(list)
        for x in entries:
            # 只存训练必需字段，评分帧与完整日志由工件SHA追溯。
            row = {k: x[k] for k in ('candidate_id', 'video_id', 'absolute_feature_index',
                'source_fpn_levels', 'proposal_start_sec', 'proposal_end_sec', 'class_probabilities', 'class_id')}
            row.update(semantic_label=LABELS[x['class_id']], evidence_scope='proposal-whole-v1',
                participation_selected=(x['video_id'], x['absolute_feature_index']) in chosen)
            grouped[x['video_id']].append(row)
        output = root/'prepared'/(s+'.json')
        save(output, dict(schema=CACHE_SCHEMA, shift=s, target_gt_used=False, teacher='qwen3_vl_8b',
            candidate_count=200, selected_count=len(chosen), participation_fraction=.2, by_video=dict(grouped),
            source_artifacts={str(manifest_path):sha(manifest_path), str(predictions_path):sha(predictions_path),
                str(root/'teacher/inference_complete.json'):sha(root/'teacher/inference_complete.json')},
            policy=POLICY))
        prepared[s] = dict(path=str(output), sha256=sha(output), selected_count=len(chosen))
    save(root/'prepared/audit.json', dict(status='passed', count=6, entries=prepared, target_gt_used=False))
    return prepared


def validate_cache(path, verify_files=True):
    p = read(path)
    if p.get('schema') != CACHE_SCHEMA or p.get('target_gt_used') is not False or p.get('policy') != POLICY:
        raise ValueError('不是本轮无GT的刷新缓存')
    rows = [r for vv in p['by_video'].values() for r in vv]
    if len(rows) != 200 or len({r['candidate_id'] for r in rows}) != 200:
        raise ValueError('刷新候选数量或身份改变')
    selected = {(r['video_id'],r['absolute_feature_index']) for r in rows if r['participation_selected']}
    if selected != {tuple(x) for x in selected_coordinates(rows)} or len(selected) != p['selected_count']:
        raise ValueError('未按原top20%规则选择')
    if verify_files:
        for name, digest in p['source_artifacts'].items():
            if sha(name) != digest: raise ValueError('缓存证据SHA改变')
    return p


def config_for(control, prepared, identity, tmpdir):
    cfg = deepcopy(control)
    if (cfg['seed'] != 123 or cfg['loader']['batch_size'] != 2
            or cfg['prototype_capacity']['variant'] != 'control' or cfg['baseline']['target_gt_used'] is not False):
        raise ValueError('必须是当前seed123/batch2原版Prototype')
    cfg['baseline'].update(code=identity, tmpdir=str(tmpdir))
    cfg['prototype_capacity'].update(prepared_cache=str(prepared), prepared_cache_sha256=sha(prepared))
    cfg['prototype_refresh'] = dict(schema=SCHEMA, rounds=1, policy=POLICY)
    return cfg


def assert_pair(cfg, original):
    from libs.models.SADA.prototype_capacity import specification
    plain = deepcopy(cfg)
    if plain.pop('prototype_refresh') != dict(schema=SCHEMA, rounds=1, policy=POLICY):
        raise ValueError('刷新方法身份改变')
    for k in ('code', 'tmpdir'): plain['baseline'][k] = original['baseline'][k]
    for k in ('prepared_cache', 'prepared_cache_sha256'): plain['prototype_capacity'][k] = original['prototype_capacity'][k]
    if plain != original or cfg['prototype_capacity']['specification'] != specification('control'):
        raise ValueError('除候选缓存、独立身份和工件路径外改变了原版训练配置')


def audit_entry(entry):
    from tools.baseline_protocol import audit_entry as baseline_audit
    entry = Path(entry); cfg = read(entry/'config.json'); done = read(entry/'complete.json')
    old = Path(done['control_complete']).parent
    if sha(old/'complete.json') != done['control_complete_sha256']: raise ValueError('静态对照改变')
    assert_pair(cfg, read(old/'config.json'))
    payload = validate_cache(cfg['prototype_capacity']['prepared_cache'])
    if sha(cfg['prototype_capacity']['prepared_cache']) != cfg['prototype_capacity']['prepared_cache_sha256']:
        raise ValueError('训练教师缓存SHA不符')
    if read(entry/'detector_initialization.json')['sha256'] != read(old/'detector_initialization.json')['sha256']:
        raise ValueError('冷启动检测器与原版不配对')
    if read(entry/'termination.json')['initial_state_sha'] != read(old/'termination.json')['initial_state_sha']:
        raise ValueError('完整学生冷初始化与原版不配对')
    result = baseline_audit(entry, expected_schema=SCHEMA)
    metrics = [json.loads(x) for x in (entry/'metrics.jsonl').read_text().splitlines()]
    if not any(all(math.isfinite(x['auxiliary'].get(k,0)) and x['auxiliary'].get(k,0)>0 for k in
                   ('rq3_target_gradient_norm','rq3_alignment_loss','rq3_source_loss','rq3_selected')) for x in metrics):
        raise ValueError('没有真实有效候选和目标辅助梯度')
    result['selected_candidates'] = payload['selected_count']
    result['variant'] = 'single_refresh'
    return result
