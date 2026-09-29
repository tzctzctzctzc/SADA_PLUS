"""从真实源域验证预测选择门槛；正式入口重算并绑定数据、教师和时间证据。"""
import argparse
import json
import math
from pathlib import Path

from tools.rq2_dense_protocol import sha256_file

SCHEMA = 'rq3-source-confidence-v1'
LOCAL_SCOPE = 'anchor-local-center-only-v1'
PROPOSAL_SCOPE = 'proposal-whole-v1'


def scope_for(strategy):
    if strategy not in ('verify', 'prototype'):
        raise ValueError('confidence calibration is only for revised Verify/Prototype')
    return LOCAL_SCOPE if strategy == 'verify' else PROPOSAL_SCOPE


def choose_threshold(scores, correct, min_precision, min_support):
    """选择源域经验精确率达标且覆盖最多的门槛；不是目标域可靠性保证。"""
    if not .5 < min_precision <= 1 or min_support < 1 or len(scores) != len(correct):
        raise ValueError('invalid source-only calibration policy')
    if any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError('invalid confidence')
    for threshold in sorted(set(scores)):
        kept = [index for index, score in enumerate(scores) if score >= threshold]
        precision = sum(correct[index] for index in kept) / len(kept)
        if len(kept) >= min_support and precision >= min_precision:
            return dict(threshold=threshold, selected_count=len(kept), empirical_precision=precision)
    return dict(threshold=None, selected_count=0, empirical_precision=None)


def members(cfg, domain, split):
    database = json.loads(Path(cfg['dataset']['json_file']).read_text(encoding='utf-8'))['database']
    subsets = {item.lower() for item in cfg[domain + '_' + split + '_split']}
    return sorted(key for key, value in database.items() if value['subset'].lower() in subsets)


def check_time_evidence(manifest, cfg, strategy):
    scope = scope_for(strategy)
    if manifest.get('evidence_scope', PROPOSAL_SCOPE) != scope:
        raise ValueError('time evidence scope differs; whole-proposal scores cannot verify local anchors')
    if strategy != 'verify':
        return
    dataset = cfg['dataset']
    grid = {'fps': dataset['default_fps'], 'feat_stride': dataset['feat_stride'],
            'num_frames': dataset['num_frames']}
    if manifest.get('reference_grid') != grid or not grid['fps'] or grid['fps'] <= 0:
        raise ValueError('local evidence reference grid differs')
    for row in manifest['candidates']:
        start = row['absolute_feature_index'] * grid['feat_stride'] / grid['fps']
        end = start + grid['num_frames'] / grid['fps']
        if abs(row['proposal_start_sec'] - start) > 1e-6 or abs(row['proposal_end_sec'] - end) > 1e-6:
            raise ValueError('local evidence interval differs from the exact anchor window')


def _source_correct(row, entry, database, cfg, strategy):
    annotations = database[row['video_id']]['annotations']
    label = entry['class_id']
    start, end = row['proposal_start_sec'], row['proposal_end_sec']
    if strategy == 'verify':
        center = (row['absolute_feature_index'] * cfg['dataset']['feat_stride']
                  + .5 * cfg['dataset']['num_frames']) / cfg['dataset']['default_fps']
        present = {int(a['label_id']) for a in annotations if a['segment'][0] <= center < a['segment'][1]}
        # 否决需要单一类别证据；“top1也是多个真动作之一”不足以否定另一个动作。
        return present == {label}
    # Prototype整段类别要求预测类覆盖过半，而非只在片段里短暂出现。
    spans = sorted((max(start, a['segment'][0]), min(end, a['segment'][1]))
                   for a in annotations if int(a['label_id']) == label
                   and max(start, a['segment'][0]) < min(end, a['segment'][1]))
    covered, right = 0., start
    for left, stop in spans:
        covered += max(0., stop - max(left, right))
        right = max(right, stop)
    return covered > .5 * (end - start)


def calibrate(cfg, cache_path, manifest_path, strategy, min_precision, min_support):
    from tools.rq3_protocol import load_cache, proposal_evidence, validate_interface_audit, audit_cache_predictions
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if manifest.get('calibration_domain') != 'source-validation':
        raise ValueError('calibration must use source validation, never target annotations')
    source_members = members(cfg, 'src', 'val')
    if set(source_members) & (set(members(cfg, 'tgt', 'train')) | set(members(cfg, 'tgt', 'val'))):
        raise ValueError('source calibration overlaps target data')
    if manifest['target_video_ids'] != source_members:
        raise ValueError('calibration members are not the complete source validation split')
    check_time_evidence(manifest, cfg, strategy)
    payload, _ = load_cache(cache_path, shift=cfg['rq2_formal']['shift'], members=source_members)
    # 权重本体在目标入口再次全量核验；这里重查真实源域预测与接口审计的绑定。
    interface = validate_interface_audit(payload['input_protocol'], payload['teacher_model'], verify_files=False)
    audit_cache_predictions(payload, interface)
    proposal_evidence(payload, manifest_path)
    by_coordinate = {(row['video_id'], row['absolute_feature_index']): row for row in manifest['candidates']}
    database = json.loads(Path(cfg['dataset']['json_file']).read_text(encoding='utf-8'))['database']
    scores, correct = [], []
    for entry in payload['entries']:
        probs = entry['class_probabilities']
        if probs.count(max(probs)) != 1:
            continue  # 平票不是可信top1。
        row = by_coordinate[entry['video_id'], entry['absolute_feature_index']]
        scores.append(max(probs))
        correct.append(_source_correct(row, entry, database, cfg, strategy))
    return dict(schema=SCHEMA, strategy=strategy, teacher_model=payload['teacher_model'], shift=payload['shift'],
                source_cache_path=str(Path(cache_path).resolve()), source_cache_sha256=sha256_file(cache_path),
                source_manifest_path=str(Path(manifest_path).resolve()), source_manifest_sha256=sha256_file(manifest_path),
                annotations_sha256=sha256_file(cfg['dataset']['json_file']),
                source_val_members=source_members, target_gt_used=False, evidence_scope=scope_for(strategy),
                proposal_teacher_sha256=payload['teacher_checkpoint_sha256'],
                gt_view=('single-class-at-anchor-center' if strategy == 'verify' else 'predicted-class-majority-of-interval'),
                input_protocol=payload['input_protocol'], min_precision=min_precision, min_support=min_support,
                evaluated_count=len(scores), **choose_threshold(scores, correct, min_precision, min_support))


def load_calibration(path, cfg, target_payload, target_manifest, strategy):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if report.get('schema') != SCHEMA or report.get('strategy') != strategy:
        raise ValueError('unknown confidence calibration/strategy')
    reproduced = calibrate(cfg, report['source_cache_path'], report['source_manifest_path'], strategy,
                           report['min_precision'], report['min_support'])
    if report != reproduced:
        raise ValueError('source confidence calibration changed or cannot be reproduced')
    if report['teacher_model'] != target_payload['teacher_model'] or report['shift'] != target_payload['shift']:
        raise ValueError('confidence calibration belongs to another teacher/shift')
    if report['proposal_teacher_sha256'] != target_payload['teacher_checkpoint_sha256']:
        raise ValueError('source/target proposals use different SO checkpoints')
    for key in ('checkpoint_sha256', 'processor_sha256', 'readout', 'frame_protocol', 'input_frames'):
        if report['input_protocol'][key] != target_payload['input_protocol'][key]:
            raise ValueError('source/target scoring interface differs: ' + key)
    # Qwen prompt hash含候选ID的选项置换，不能直接跨候选比较；模板代码已绑定processor hash。
    check_time_evidence(target_manifest, cfg, strategy)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True, help='已审计的该shift实际训练JSON配置')
    parser.add_argument('--source-cache', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--strategy', choices=('verify', 'prototype'), required=True)
    # 必须显式冻结策略；没有暗设0.7门槛或用目标mAP挑门槛的入口。
    parser.add_argument('--min-precision', type=float, required=True)
    parser.add_argument('--min-support', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    report = calibrate(cfg, args.source_cache, args.source_manifest, args.strategy, args.min_precision, args.min_support)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, sort_keys=True)
    print(json.dumps({key: report[key] for key in ('threshold', 'selected_count', 'empirical_precision')}))


if __name__ == '__main__':
    main()
