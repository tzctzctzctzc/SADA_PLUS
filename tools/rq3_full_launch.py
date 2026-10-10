"""100%独立启动凭证：RQ4全24终审、原20%配对和真实学生验收。"""
import json
import math
import os
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_full_participation import VARIANT, prepare

SHIFTS = tuple('S'+str(i) for i in range(1, 7))
RQ4_KEYS = {(t, s, shift) for t in ('timelens2_8b', 'timeple_8b')
            for s in ('direct', 'gated') for shift in SHIFTS}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def bind(path):
    path = str(Path(path).resolve())
    return dict(path=path, sha256=sha256_file(path))


def checked_file(item):
    if sha256_file(item['path']) != item['sha256']:
        raise ValueError('launch evidence changed: '+item['path'])
    return read(item['path'])


def rq4_entries(final_path, expected_head, full_audit=False):
    """不接受只有count的声明；枚举精确24个身份并绑定每份complete。"""
    final_path = Path(final_path).resolve()
    final = read(final_path)
    entries = final.get('entries', [])
    keys = [(e['teacher'], e['strategy'], e['shift']) for e in entries]
    if (final.get('status') != 'passed' or final.get('count') != 24
            or len(keys) != 24 or set(keys) != RQ4_KEYS):
        raise ValueError('RQ4 exact 24-entry final audit is required')
    bound = []
    for entry, key in zip(entries, keys):
        root = Path(entry['root']).resolve()
        if root != final_path.parent/'formal'/key[0]/key[2]/key[1]:
            raise ValueError('RQ4 entry belongs to another matrix root')
        record = read(root/'complete.json')
        cfg = read(root/'preflight.json')
        cache = read(cfg['rq4_boundary']['cache_path'])
        if (tuple(record.get(k) for k in ('teacher', 'strategy', 'shift')) != key
                or record.get('schema') != 'rq4-complete-v1'
                or record.get('code_commit') != expected_head
                or record.get('seed') != 6 or record.get('batch_size') != 2
                or cfg['rq4_boundary']['strategy'] != key[1]
                or cache['teacher'] != key[0] or cache['shift'] != key[2]):
            raise ValueError('RQ4 completed identity mismatch')
        if full_audit:
            from tools.run_rq4_entry import audit_complete
            audit_complete(root)
        bound.append(bind(root/'complete.json'))
    return bound


def create_launch_proof(final_path, rq4_head, references, expected_head):
    from tools.run_rq2_matched_control import check_code
    from tools.run_rq3_matrix import completed_baseline
    from tools.run_rq3_trusted_monitor import audit_complete
    check_code(expected_head)
    if set(references) != set(SHIFTS):
        raise ValueError('exact six historical references required')
    completed = rq4_entries(final_path, rq4_head, full_audit=True)
    refs = {}
    for shift in SHIFTS:
        path = Path(references[shift]).resolve()
        old = read(path)
        baselines = {shift: completed_baseline(Path(old['baseline_root']), shift)}
        if audit_complete(path, baselines) != ('qwen3_vl_8b', 'prototype', shift):
            raise ValueError('historical reference identity mismatch')
        prepared = prepare(path)
        if prepared['planned_config']['rq2_formal']['shift'] != shift:
            raise ValueError('reference mapped to another shift')
        refs[shift] = prepared['reference_artifacts']
    return dict(schema='rq3-full-launch-v1', status='passed', code_commit=expected_head,
                rq4_head=rq4_head, predecessor=bind(final_path), rq4_completed=completed,
                references=refs)


def validate_launch_proof(path, expected_head=None):
    proof = read(path)
    if (proof.get('schema') != 'rq3-full-launch-v1' or proof.get('status') != 'passed'
            or set(proof.get('references', {})) != set(SHIFTS)
            or (expected_head and proof.get('code_commit') != expected_head)):
        raise ValueError('invalid full-participation launch proof')
    checked_file(proof['predecessor'])
    actual = rq4_entries(proof['predecessor']['path'], proof['rq4_head'])
    if actual != proof['rq4_completed']:
        raise ValueError('RQ4 completed evidence changed after independent audit')
    for item in actual:
        for artifact, digest in checked_file(item)['artifacts'].items():
            if sha256_file(artifact) != digest:
                raise ValueError('RQ4 audited artifact changed: '+artifact)
    for files in proof['references'].values():
        for artifact in files.values():
            checked_file(artifact)
    return proof


def validate_smoke(report, cfg):
    if (report.get('status') != 'passed' or report.get('formal_result') is not False
            or report.get('real_video_features') is not True
            or report.get('diagnostic_epoch') != 50 or report.get('seed') != 6
            or report.get('batch_size') != 2 or report.get('variant') != VARIANT
            or report.get('run_signature') != cfg['rq2_formal']['run_signature']
            or report.get('auxiliary') != cfg['rq3_auxiliary']):
        raise ValueError('real-data acceptance identity mismatch')
    m = report['metrics']
    if not m or any(not math.isfinite(v) for v in m.values()):
        raise ValueError('nonfinite acceptance metrics')
    for key in ('cls', 'reg', 'downstream'):
        if m['target_gt_'+key+'_loss'] != 0:
            raise ValueError('target GT leakage in acceptance')
    for key in ('candidate_count', 'fully_visible_count', 'selected', 'source_loss',
                'alignment_loss', 'gradient_norm', 'target_gradient_norm'):
        if m['rq3_'+key] <= 0:
            raise ValueError('missing real auxiliary routing/gradient: '+key)
    if not (m['rq3_selected'] <= m['rq3_eligible'] <= m['rq3_fully_visible_count']
            <= m['rq3_candidate_count']):
        raise ValueError('invalid acceptance routing counts')


def acceptance_ready(path, cfg, proof_path):
    record = read(path)
    proof = read(proof_path)
    if (record.get('status') != 'passed' or record.get('code_commit') != proof['code_commit']
            or record.get('launch_proof') != bind(proof_path)
            or set(record.get('reports', {})) != set(SHIFTS)):
        raise ValueError('six-shift real acceptance proof required')
    for shift, item in record['reports'].items():
        report = checked_file(item)
        if (report['run_signature'].get('shift') != shift
                or report['auxiliary'].get('paired_reference') != proof['references'][shift]):
            raise ValueError('acceptance report belongs to another shift/reference')
        validate_smoke(report, dict(rq2_formal=dict(run_signature=report['run_signature']),
                                   rq3_auxiliary=report['auxiliary']))
    validate_smoke(checked_file(record['reports'][cfg['rq2_formal']['shift']]), cfg)


def assert_runtime_launch(cfg):
    from tools.run_rq2_matched_control import check_code
    path = os.environ.get('RQ3_FULL_LAUNCH_PROOF')
    if not path:
        raise ValueError('Qwen100 launch/real-data acceptance pending')
    proof = validate_launch_proof(path)
    check_code(proof['code_commit'])
    if cfg['rq3_auxiliary']['paired_reference'] != proof['references'][cfg['rq2_formal']['shift']]:
        raise ValueError('launch proof belongs to another paired reference')
    # 只有替换成无checkpoint诊断train方法的独立wrapper可走验收路径。
    # 环境变量/CLI布尔值不能把正式Trainer变成已验收训练。
    from libs.trainer.da_trainer import Trainer
    if getattr(Trainer.train, '_rq3_full_diagnostic', False):
        return
    acceptance = os.environ.get('RQ3_FULL_ACCEPTANCE')
    if not acceptance:
        raise ValueError('Qwen100 real-data acceptance pending')
    acceptance_ready(acceptance, cfg, path)


def audit_launch_reference(root, cfg, expected_head):
    links = read(Path(root)/'launch_reference.json')
    checked_file(links['proof']); checked_file(links['acceptance'])
    proof = validate_launch_proof(links['proof']['path'], expected_head)
    if cfg['rq3_auxiliary']['paired_reference'] != proof['references'][cfg['rq2_formal']['shift']]:
        raise ValueError('completed launch reference mismatch')
    acceptance_ready(links['acceptance']['path'], cfg, links['proof']['path'])
    paths = [Path(root)/'launch_reference.json', Path(links['proof']['path']),
             Path(links['acceptance']['path']), Path(proof['predecessor']['path'])]
    paths += [Path(x['path']) for x in proof['rq4_completed']]
    paths += [Path(x['path']) for x in read(links['acceptance']['path'])['reports'].values()]
    return paths
