"""Qwen Prototype100%的CPU配置准备；不启动训练，不改变旧20%入口。"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file
from tools.rq2_formal import build_formal_run_signature
from tools.rq3_confidence import select_budget
from tools.rq3_early_stop import POLICY
from tools.rq3_protocol import REVISION, REVISED_FIXED, load_cache, proposal_evidence

VARIANT = 'rq3-qwen-prototype-full-participation-v1'
CONFIDENCE_POLICY = 'all-unambiguous-original-candidates-per-shift-v1'


def select_all(entries):
    """保留20%原来的稳定排名；类别top1平票仍不提供类别监督。"""
    coordinates = [(entry['video_id'], entry['absolute_feature_index']) for entry in entries]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError('duplicate candidate coordinates')
    ranked = [entry for entry in entries
              if entry['class_probabilities'].count(max(entry['class_probabilities'])) == 1]
    ranked.sort(key=lambda entry: (-max(entry['class_probabilities']), entry['video_id'],
                                   entry['absolute_feature_index']))
    return [[entry['video_id'], entry['absolute_feature_index']] for entry in ranked]


def _reference_check(reference, entries):
    auxiliary = reference['rq3_auxiliary']
    if (auxiliary.get('schema') != REVISION or auxiliary.get('strategy') != 'prototype'
            or auxiliary.get('teacher_model') != 'qwen3_vl_8b'
            or reference.get('seed') != 6 or reference.get('init_rand_seed') != 6
            or reference['dataset']['input_dim'] != 2304
            or reference['loader']['batch_size'] != 2 or reference['opt']['epochs'] != 50
            or reference['rq2_formal']['regime'] != 'uda'
            or reference['train_cfg']['early_stop'] > 0 or auxiliary.get('acceleration')
            or any(auxiliary.get(key) != value for key, value in REVISED_FIXED.items())):
        raise ValueError('reference must be original Qwen Prototype20% seed6/batch2/50')
    shift = reference['rq2_formal']['shift']
    if shift not in ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'):
        raise ValueError('unexpected reference shift')
    expected_policy = None if shift in ('S1', 'S2', 'S3') else POLICY
    if auxiliary.get('early_stopping') != expected_policy:
        raise ValueError('reference stopping policy differs from paired historical entry')
    if auxiliary.get('confidence_threshold') != 0.0:
        raise ValueError('reference confidence threshold differs')
    if auxiliary.get('selected_coordinates') != select_budget(entries):
        raise ValueError('reference 20% selection cannot be reproduced')
    # 不通过修改截断上限扩大有效候选池；历史六项均远低于该上限。
    counts = Counter(entry['video_id'] for entry in entries)
    if not counts or max(counts.values()) > auxiliary['prototype_max_proposals']:
        raise ValueError('original per-video proposal cap would truncate the full pool')


def planned_config(reference, entries):
    _reference_check(reference, entries)
    candidate = deepcopy(reference)
    candidate['rq3_auxiliary'].update(
        schema=VARIANT, confidence_policy=CONFIDENCE_POLICY,
        participation_fraction=1.0, selected_coordinates=select_all(entries))
    candidate['rq2_formal']['run_signature'] = build_formal_run_signature(candidate)
    return candidate


def audit_pair(reference, candidate, entries):
    """只允许参与身份/比例/派生名单改变；损失、源诊断和停止规则逐值配对。"""
    expected = planned_config(reference, entries)
    if candidate['rq3_auxiliary'] != expected['rq3_auxiliary']:
        raise ValueError('full-participation auxiliary differs from the paired protocol')
    if build_formal_run_signature(candidate) != build_formal_run_signature(expected):
        raise ValueError('training configuration differs beyond participation')
    if candidate['rq2_formal']['run_signature'] != build_formal_run_signature(candidate):
        raise ValueError('candidate run signature is stale')
    return dict(variant=VARIANT, pool_count=len(entries),
                old_selected_count=len(reference['rq3_auxiliary']['selected_coordinates']),
                new_selected_count=len(candidate['rq3_auxiliary']['selected_coordinates']),
                stopping_rule_matched=True, source_calibration_unchanged=True,
                actual_routed_coverage_guaranteed=False)


def prepare(complete_path):
    """读取已终审条目并绑定工件；这不是checkpoint/结果终审的替代品。"""
    complete_path = Path(complete_path).resolve()
    preflight_path = complete_path.parent / 'preflight.json'
    record = json.loads(complete_path.read_text(encoding='utf-8'))
    reference = json.loads(preflight_path.read_text(encoding='utf-8'))
    aux = reference['rq3_auxiliary']
    identity = dict(teacher='qwen3_vl_8b', strategy='prototype', seed=6, batch_size=2,
                    shift=reference['rq2_formal']['shift'], cache_sha256=aux['cache_sha256'])
    if any(record.get(key) != value for key, value in identity.items()):
        raise ValueError('complete identity differs from reference preflight')
    artifacts = {'complete': str(complete_path), 'preflight': str(preflight_path)}
    for name in ('cache', 'manifest', 'calibration'):
        path = aux[name + '_path']
        if sha256_file(path) != aux[name + '_sha256']:
            raise ValueError('reference artifact changed: ' + name)
        artifacts[name] = path
    payload, _ = load_cache(aux['cache_path'], identity['teacher'], identity['shift'])
    for key in ('candidate_sha256', 'frames_sha256', 'teacher_checkpoint_sha256', 'input_protocol'):
        if payload[key] != aux[key]:
            raise ValueError('reference cache provenance differs: ' + key)
    proposal_evidence(payload, aux['manifest_path'], aux['selected_coordinates'])
    candidate = planned_config(reference, payload['entries'])
    pairing = audit_pair(reference, candidate, payload['entries'])
    # 新schema故意不被旧训练/终审入口接受，防止100%混入20%矩阵。
    return dict(schema='rq3-full-participation-preparation-v1', training_ready=False,
                requires=['rq4-final-audit-24', 'full-participation-training-entry-and-audit'],
                reference_artifacts={key: dict(path=value, sha256=sha256_file(value))
                                     for key, value in artifacts.items()},
                pairing=pairing, planned_config=candidate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-complete', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_complete), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
