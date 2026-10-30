"""源域能力验证的固定协议；推理输入白名单与离线GT严格分开。"""
from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import random

from tools.baseline_protocol import read, save, sha
from tools.audit_prototype_chain import truth_for, auc, tiou
from tools.qwen_source_interface import LABELS, ACTION_PHRASES

SCHEMA = 'qwen32-source-information-v1'
OPTIONS = tuple('ABCDEFGH')
STATES = ('complete', 'start_cut', 'end_cut', 'both_cut', 'known_boundary_uncertain', 'mixed_or_other', 'no_action', 'uncertain')
DESCRIPTIONS = (
    'One uniquely dominant listed action occurs inside the proposal; its execution starts and ends inside it.',
    'One uniquely dominant listed action occurs inside; its execution starts before the proposal and ends inside.',
    'One uniquely dominant listed action occurs inside; its execution starts inside the proposal and ends after it.',
    'One uniquely dominant listed action occurs inside; its execution extends across both proposal boundaries.',
    'One uniquely dominant listed action clearly occurs inside, but its temporal completeness cannot be established.',
    'Action is visible inside, but there is no uniquely dominant listed action (mixed instances or another action).',
    'No visible hand-object action occurs inside the proposal; objects or persistent states alone are insufficient.',
    'Insufficient temporal or visual evidence to decide, including inability to distinguish repeated occurrences.',
)
POLICY = dict(source_candidates=1095, source_validation_only=True, target_data_used=False,
    no_student_training=True, models=['so', 'uda', 'prototype'], student_checkpoint='source_selected_best_EMA',
    student_readout='source_branch_detection_logits_at_original_FPN_level_and_anchor_no_GT_routing',
    class_predictions='reuse_frozen_8b_and_32b_identical_8_frame_native_video_scores',
    validity_input='16_timestamped_context2_images_no_cached_class_or_GT',
    validity_model='Qwen3-VL-32B-Instruct_BF16_SDPA', completeness_gt_tolerance='half_max_observed_frame_gap',
    frame_budget_difference_explicit=True, option_order='candidate_id_SHA_permutation',
    ranking='per_shift_floor_20percent_score_desc_candidate_id_tie',
    source_validation_reused_not_untouched_test=True, no_target_gain_claim=True)


def mapping(cid):
    order = list(range(len(STATES)))
    random.Random(int.from_bytes(hashlib.sha256(cid.encode()).digest()[:8], 'big')).shuffle(order)
    return order


def inference_row(row):
    # 只有这些值能进入教师；拒绝把GT、学生类别或旧8B类别带进提示。
    keys = ('candidate_id', 'shift', 'video_id', 'proposal', 'frame_paths', 'frame_sha256', 'frame_times_sec')
    result = {k: row[k] for k in keys}
    times = result['frame_times_sec']
    if len(times) != 16 or len(set(result['frame_paths'])) != 16 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('必须是16个有序真实帧')
    if len(result['frame_sha256']) != 16 or result['proposal'][1] <= result['proposal'][0]:
        raise ValueError('输入帧SHA或区间非法')
    return result


def prompt(row):
    row = inference_row(row)
    actions = '; '.join(f'{a}: {b}' for a, b in zip(LABELS, ACTION_PHRASES))
    a, b = row['proposal']
    choices = '\n'.join(f'{letter}: {DESCRIPTIONS[state]}' for letter, state in zip(OPTIONS, mapping(row['candidate_id'])))
    return ('Judge the temporal quality of an imperfect detector proposal in an egocentric video.\n'
        f'LISTED ACTIONS: {actions}\nREFERENCE PROPOSAL: [{a:.6f}, {b:.6f}] seconds.\n'
        'The chronological images include context before and after the proposal. Judge actions INSIDE the proposal; '
        'use the outside context only to establish whether execution crosses either boundary. '
        'A uniquely dominant action occupies more than half the proposal. Relevant objects, preparatory movements '
        'or persistent resulting states alone do not prove execution. Do not assume the proposal is correct. '
        'Do not infer invisible transitions between sparse frames. If the occurrence or its boundaries cannot be '
        'established, choose insufficient evidence.\nOPTIONS:\n' + choices +
        '\nAnswer with exactly one option letter, without an explanation.')


def scores(cid, logits, log_total):
    if len(logits) != len(STATES) or not all(math.isfinite(x) for x in list(logits) + [log_total]):
        raise ValueError('选项logits非法')
    if max(logits) > log_total + 1e-5:
        raise ValueError('全词表归一化非法')
    weights = [math.exp(x-max(logits)) for x in logits]; total = sum(weights)
    probabilities = [w/total for w in weights]
    semantic = {STATES[state]: probabilities[i] for i, state in enumerate(mapping(cid))}
    absolute = [math.exp(x-log_total) for x in logits]
    if sum(absolute) > 1+1e-5:
        raise ValueError('选项词表质量非法')
    valid = sum(semantic[s] for s in STATES[:5])
    best = max(range(len(STATES)), key=lambda i: logits[i])
    return dict(state=STATES[mapping(cid)[best]], tied=logits.count(logits[best]) > 1,
        state_probabilities=semantic, validity_score=valid, complete_score=semantic['complete'],
        completeness_given_valid=semantic['complete']/valid if valid else 0.,
        candidate_mass=sum(absolute), option_logits=logits, full_vocab_logsumexp=log_total)


def offline_truth(proposal, annotations, frame_times):
    result = truth_for(proposal, annotations)
    result['valid_known_majority'] = result['unique_majority'] is not None
    result.update(completeness_state=None, completeness_observable=False, reference=None,
                  annotated_instance_coverage=None)
    label = result['unique_majority']
    # 同类反复发生时不能拿类别union伪装成单个完整动作。
    a, b = proposal
    spans = sorted({tuple(x['segment']) for x in annotations if int(x['label_id']) == label and
                    min(b, x['segment'][1])-max(a, x['segment'][0]) > (b-a)*.5}) if label is not None else []
    if len(spans) == 1:
        s, e = spans[0];tol = max(y-x for x, y in zip(frame_times, frame_times[1:]))/2
        left, right = s < a-tol, e > b+tol
        result.update(reference=[s, e], completeness_state=STATES[3 if left and right else 1 if left else 2 if right else 0],
            completeness_observable=frame_times[0] <= s <= e <= frame_times[-1],
            annotated_instance_coverage=max(0., min(b, e)-max(a, s))/(e-s), tolerance_sec=tol)
    return result


def top20(rows, score):
    chosen = []
    for shift in sorted({r['shift'] for r in rows}):
        group = [r for r in rows if r['shift'] == shift]
        chosen += sorted(group, key=lambda r: (-score(r), r['candidate_id']))[:len(group)//5]
    return chosen


def binary_summary(rows, score, truth):
    return dict(count=len(rows), positives=sum(bool(truth(r)) for r in rows),
        auroc=auc([score(r) for r in rows], [bool(truth(r)) for r in rows]),
        precision_at_20percent=(sum(bool(truth(r)) for r in top20(rows, score))/len(top20(rows, score))
                                if top20(rows, score) else None))


def complementarity(rows, student, teacher, label_key):
    valid = [r for r in rows if r['truth'][label_key] is not None and student in r['students']]
    counts = Counter()
    for r in valid:
        label = r['truth'][label_key]
        a = r['students'][student]['class_id'] == label; b = r[teacher]['class_id'] == label
        counts['both_correct' if a and b else 'student_only_correct' if a else 'teacher_only_correct' if b else 'both_wrong'] += 1
    n = len(valid)
    return dict(count=n, counts=dict(counts), student_accuracy=(counts['both_correct']+counts['student_only_correct'])/n if n else None,
        teacher_accuracy=(counts['both_correct']+counts['teacher_only_correct'])/n if n else None,
        net_corrections=counts['teacher_only_correct']-counts['student_only_correct'])


def analyze(root):
    rows = read(root/'evaluation_rows.json')
    students = {}
    for path in sorted((root/'students').glob('*/predictions.json')):
        entry = read(path.parent/'inference_complete.json')
        if (sha(path) != entry['predictions_sha256'] or entry['status'] != 'passed'
                or sha(path.parent/'job.json') != entry['job_sha256']):
            raise ValueError('学生预测SHA改变')
        for prediction in read(path):
            students.setdefault(prediction['candidate_id'], {})[entry['regime']] = prediction
    availability = {r['candidate_id']: r for r in rows}
    for cid in students:
        if cid not in availability: raise ValueError('出现未知学生候选')
    for r in rows:
        r['students'] = students.get(r['candidate_id'], {})
        if set(r['students']) != set(POLICY['models']):raise ValueError('每条候选必须有三个冻结学生预测')
    result = {}
    for scope in ['pooled_not_independent'] + [f'S{i}' for i in range(1, 7)]:
        subset = rows if scope.startswith('pooled') else [r for r in rows if r['shift'] == scope]
        values = {}
        for student in POLICY['models']:
            for teacher in ['8b', '32b']:
                key = student+'__'+teacher
                available = [r for r in subset if student in r['students']]
                values[key] = {label: complementarity(available, student, teacher, label)
                    for label in ['unique_majority', 'clean_reference_label']}
                values[key]['confidence_bins'] = []
                for lo, hi in [(0., .5), (.5, .7), (.7, .9), (.9, 1.000001)]:
                    bins = [r for r in available if lo <= r['students'][student]['score'] < hi]
                    values[key]['confidence_bins'].append(dict(lower=lo, upper=min(1.,hi),
                        **complementarity(bins, student, teacher, 'unique_majority')))
        result[scope] = values
    save(root/'complementarity.json', dict(schema=SCHEMA, status='passed', results=result,
        rows_sha256=sha(root/'evaluation_rows.json'), no_training=True,
        limitation='source-selected checkpoints evaluated on reused source validation; conditional detector-anchor comparison'))
    prediction_path = root/'validity/predictions.jsonl'
    if not (root/'validity/inference_complete.json').exists():return
    done = read(root/'validity/inference_complete.json')
    if sha(prediction_path) != done['predictions_sha256']:raise ValueError('32B新任务评分SHA改变')
    predictions = {p['candidate_id']:p for p in map(__import__('json').loads, prediction_path.read_text().splitlines())}
    if set(predictions) != set(availability):raise ValueError('32B评分身份不完整')
    for r in rows:r['quality'] = predictions[r['candidate_id']]
    results = {}
    for scope in ['pooled_not_independent'] + [f'S{i}' for i in range(1, 7)]:
        subset = rows if scope.startswith('pooled') else [r for r in rows if r['shift'] == scope]
        truth = lambda r:r['truth']['valid_known_majority']
        baselines = {t:binary_summary(subset, lambda r,t=t:r[t]['confidence'], truth) for t in ['8b','32b']}
        observable = [r for r in subset if r['truth']['completeness_observable']]
        results[scope] = dict(validity= binary_summary(subset, lambda r:r['quality']['validity_score'], truth),
            class_confidence_baselines=baselines, states=dict(Counter(r['quality']['state'] for r in subset)),
            completeness= binary_summary(observable, lambda r:r['quality']['completeness_given_valid'],
                                            lambda r:r['truth']['completeness_state']=='complete'),
            observable_reference_count=len(observable),
            completeness_state_accuracy=(sum(r['quality']['state']==r['truth']['completeness_state'] for r in observable)/len(observable) if observable else None),
            unknown_or_mixed=sum(not truth(r) and r['truth']['any_annotated_fraction']>0 for r in subset),
            no_annotation_overlap=sum(r['truth']['any_annotated_fraction']==0 for r in subset))
    save(root/'validity_comparison.json',dict(schema=SCHEMA,status='passed',results=results,
        predictions_sha256=sha(prediction_path), source_only=True, no_training=True,
        limitations=['GT may not annotate every visible motion; no-annotation is not proof of no action',
          '16 context frames versus cached 8 interior frames; task and input both differ',
          'Completeness metrics exclude nonunique instances and boundaries outside observed context',
          'Source validation reused; no target reliability or downstream mAP claim']))
