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
from tools.rq3_protocol import (REVISION, REVISED_FIXED, FULL_PARTICIPATION_REVISION,
                                load_cache, proposal_evidence)

VARIANT = FULL_PARTICIPATION_REVISION
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
    normalized = deepcopy(candidate)
    normalized['rq3_auxiliary'].pop('paired_reference', None)
    if normalized['rq3_auxiliary'] != expected['rq3_auxiliary']:
        raise ValueError('full-participation auxiliary differs from the paired protocol')
    if build_formal_run_signature(normalized) != build_formal_run_signature(expected):
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
    # 这只是配置准备；必须另外绑定原结果，完成正式入口及验收后才可训练。
    return dict(schema='rq3-full-participation-preparation-v1', training_ready=False,
                requires=['rq4-final-audit-24', 'full-participation-training-entry-and-audit'],
                reference_artifacts={key: dict(path=value, sha256=sha256_file(value))
                                     for key, value in artifacts.items()},
                pairing=pairing, planned_config=candidate)


def apply_full_participation(cfg, reference_complete):
    """先按旧CLI生成并配对完整20%配置，再只改变目标参与预算。"""
    if not reference_complete:
        return
    report = prepare(reference_complete)
    reference = json.loads(Path(report['reference_artifacts']['preflight']['path']).read_text())
    if (cfg.get('rq4_boundary') or
            build_formal_run_signature(cfg) != build_formal_run_signature(reference)):
        raise ValueError('current training configuration does not match the original 20% entry')
    cfg['rq3_auxiliary'] = deepcopy(report['planned_config']['rq3_auxiliary'])
    cfg['rq3_auxiliary']['paired_reference'] = report['reference_artifacts']
    cfg['rq2_formal']['run_signature'] = build_formal_run_signature(cfg)


def validate_runtime(cfg, regime):
    """新身份只能由已绑定原20%工件重建；完整旧缓存/源诊断门禁继续执行。"""
    from tools.rq3_protocol import validate_config
    if regime != 'uda' or cfg.get('rq4_boundary'):
        raise ValueError('full participation requires standalone matched RQ3 UDA')
    proof = cfg['rq3_auxiliary'].get('paired_reference')
    if not proof or set(proof) != {'complete', 'preflight', 'cache', 'manifest', 'calibration'}:
        raise ValueError('full participation requires bound reference artifacts')
    for artifact in proof.values():
        if sha256_file(artifact['path']) != artifact['sha256']:
            raise ValueError('paired reference artifact changed')
    report = prepare(proof['complete']['path'])
    if report['reference_artifacts'] != proof:
        raise ValueError('paired reference artifacts belong to different entries')
    reference = json.loads(Path(proof['preflight']['path']).read_text())
    payload, _ = load_cache(reference['rq3_auxiliary']['cache_path'])
    audit_pair(reference, cfg, payload['entries'])
    # reference必须仍是旧v5；此调用不会递归进入本函数。
    validate_config(reference, regime)


def assert_launch_ready(cfg):
    if cfg.get('rq3_auxiliary', {}).get('schema') == VARIANT:
        from tools.rq3_full_launch import assert_runtime_launch
        assert_runtime_launch(cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-complete', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference_complete), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
