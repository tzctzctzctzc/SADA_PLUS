"""把SADA六向共同选出的唯一配方冻结迁移；不在新检测器上重新选参。"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import ours_final as frozen
from shared_recipe import factors
from uniform_tricks import transform


def duration_factor(boxes, power):
    weight = np.ones(len(boxes))
    if power == 0 or not len(boxes):
        return weight
    lengths = boxes[:, 1].astype(float) - boxes[:, 0]
    positive = lengths[lengths > 0]
    if len(positive):
        weight = np.clip(lengths / np.median(positive), .25, 4.) ** power
    return weight


@contextmanager
def installed_rule(rule):
    """仅覆盖当前独立推理进程内的函数；不改磁盘源码或训练模型。"""
    profiles, responses, final_scores = frozen.profiles, frozen.responses, frozen.final_scores
    temperature = float(rule['temp'])
    assert temperature in (1., 2.)
    policy = 'control' if temperature == 1 else 'temperature2'
    power, sigma = float(rule['proto_power']), float(rule['sigma'])
    blend, duration_power = float(rule['blend']), float(rule.get('duration_power', 0.))
    assert power >= 0 and sigma >= 0 and 0 <= blend <= 1
    assert np.isfinite([temperature, power, sigma, blend, duration_power, float(rule['width'])]).all()
    assert float(rule['width']) > 0

    def changed_profiles(cache, boxes, labels):
        return profiles(transform(cache, policy), boxes, labels)

    def changed_responses(cache, boxes, classes):
        return responses(transform(cache, policy), boxes, classes)

    def changed_scores(boxes, raw, labels, cos, ready, temporal, evidence):
        if not len(raw):
            return raw.copy()
        a, b, other = factors(boxes, raw, labels, cos, ready, temporal, evidence,
                              powers=(power,), sigmas=(sigma,))[power, sigma]
        scores = (a + blend * (b - a)) * other
        # 本入口每次接收一个视频的全部原生候选，参照中位数不按类或S选择。
        scores *= duration_factor(boxes, duration_power)
        assert np.isfinite(scores).all() and (scores >= 0).all()
        return scores

    frozen.profiles, frozen.responses, frozen.final_scores = changed_profiles, changed_responses, changed_scores
    try:
        yield
    finally:
        frozen.profiles, frozen.responses, frozen.final_scores = profiles, responses, final_scores


def apply_width(boxes, videos, database, width):
    if width == 1:
        return boxes
    limits = np.asarray([float(database[str(v)]['duration']) for v in videos])
    assert np.isfinite(limits).all() and (limits > 0).all()
    center = boxes.astype(float).mean(1)
    half = (boxes[:, 1].astype(float) - boxes[:, 0]) * (width / 2)
    return np.column_stack((np.clip(center - half, 0, limits), np.clip(center + half, 0, limits)))


@torch.inference_mode()
def evaluate_entry(entry, out, plan):
    from tools.baseline_protocol import read, save, sha
    from tools.full_readout import score
    from tools.native_ap_fast import install
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from independent_ap import IndependentAP
    import pandas as pd

    rule = dict(plan['common_rule'])
    cfg, audit = read(entry / 'config.json'), read(entry / 'audit.json')
    assert audit['passed'] and audit['source_selected'] and audit['target_GT_training'] is False and audit['seed'] == 123
    assert cfg['test_cfg'] == plan['test_cfg']
    checkpoint = sha(entry / 'best.pt')
    assert checkpoint == audit['checkpoint_sha256'] == plan['hashes'][audit['detector'] + '/' + audit['shift']]
    out.mkdir(parents=True, exist_ok=True)
    prewidth = out / 'prewidth'
    prewidth.mkdir(exist_ok=True)
    binding = dict(common_rule=rule, checkpoint_sha256=checkpoint, config_sha256=sha(entry / 'config.json'),
                   plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                   implementation_sha256=sha(Path(__file__)))
    binding_file = prewidth / 'binding.json'
    if binding_file.is_file():
        assert read(binding_file) == binding, '已有prewidth结果与冻结配方不一致，禁止覆盖'
    else:
        assert not (prewidth / 'result.json').exists() and not (prewidth / 'predictions.pt').exists()
        save(binding_file, binding)
    if not (prewidth / 'result.json').is_file():
        # 原入口照常验证源选模、EMA严格加载、GT隔离及原框原生AP；宽度此时不变。
        with installed_rule(rule):
            frozen.evaluate_entry(entry, prewidth, plan)
    before = read(prewidth / 'result.json')
    assert before['checkpoint_sha256'] == checkpoint and before['selected_epoch'] == audit['selected_epoch']
    np.testing.assert_allclose(before['native']['by_tiou'],
        100 * np.asarray(audit['results']['target']['map_by_tiou']), atol=1e-5, rtol=0)
    assert sha(prewidth / 'predictions.pt') == before['predictions_sha256']

    install()
    torch.set_num_threads(2)
    prediction = torch.load(prewidth / 'predictions.pt', map_location='cpu')
    videos = np.asarray(prediction['video-id'])
    boxes = np.column_stack((prediction['t-start'], prediction['t-end'])).astype(float)
    confidence = np.asarray(prediction['score'], dtype=float)
    labels = np.asarray(prediction['label'])
    assert len(boxes) == len(labels) == len(confidence) == before['predictions']
    assert np.isfinite(confidence).all() and (confidence >= 0).all()
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    database = read(annotation)['database']
    boxes = apply_width(boxes, videos, database, float(rule['width']))
    columns = {'video-id': videos, 't-start': boxes[:, 0], 't-end': boxes[:, 1], 'label': labels, 'score': confidence}
    metrics = score(cfg, columns)
    truth = load_gt_seg_from_json(annotation, split=cfg['tgt_val_split'], action_subsampling='10_majority_classes')
    truth = truth[truth['t-start'] < truth['t-end']].reset_index(drop=True)
    frame = pd.DataFrame(columns)
    independent = np.array([IndependentAP(truth[truth.label == c].reset_index(drop=True),
        frame[frame.label == c].reset_index(drop=True)).ap(np.ones(int((frame.label == c).sum()), dtype=bool))
        for c in sorted(truth.label.unique())]).mean(0) * 100
    np.testing.assert_allclose(independent, metrics['by_tiou'], atol=1e-5, rtol=0)
    if float(rule['width']) == 1:
        np.testing.assert_allclose(metrics['by_tiou'], before['ours']['by_tiou'], atol=1e-5, rtol=0)
    # 标注私有覆盖后的真实读出规则；保留原生对照和prewidth预测。
    before.update(rule='frozen_shared_recipe_prewidth', common_rule=rule, targetGT_only_offline=True)
    save(prewidth / 'result.json', before)
    torch.save(columns, out / 'predictions.pt')
    result = dict(before, ours=metrics, prewidth=before['ours'], common_rule=rule,
        predictions_sha256=sha(out / 'predictions.pt'), prewidth_predictions_sha256=before['predictions_sha256'],
        rule='frozen_shared_recipe', independent_AP_passed=True, targetGT_only_offline=True,
        training_runs=0, new_vlm_calls=0, seed=123, parameters_reselected=False,
        selection='one SADA-six-shift-development common_rule, frozen for every detector and shift',
        limitation='single seed; same target development corpus; frozen transfer does not create an independent test set')
    save(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('entry', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('plan', type=Path)
    args = parser.parse_args()
    evaluate_entry(args.entry, args.out, json.loads(args.plan.read_text(encoding='utf-8-sig')))
