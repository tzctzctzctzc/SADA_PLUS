"""冻结 Ours 的免训练配对实验：只改变密集证据图，不改候选或检查点。"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from ours_final import final_scores, profiles, responses

POLICIES = ('control', 'median3', 'coarse_logit', 'temperature2')


def logit(q):
    # 只使用浮点可表示端点，避免额外引入可调截断阈值。
    q = np.clip(np.asarray(q, dtype=np.float64), np.finfo(np.float64).tiny, np.nextafter(1., 0.))
    return np.log(q) - np.log1p(-q)


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def transform(cache, policy):
    """3格为最小非平凡奇数窗；层融合等权；温度固定2，均不按S选参。"""
    if policy == 'control':
        return cache
    result = dict(cache)
    for level, stride in enumerate(cache['strides']):
        q = cache['q' + str(level)]
        if policy == 'median3':
            padded = np.pad(q, ((0, 0), (1, 1)), mode='edge')
            changed = np.median(np.stack([padded[:, j:j + q.shape[1]] for j in range(3)]), axis=0)
        elif policy == 'temperature2':
            a, b = np.sqrt(q.astype(float)), np.sqrt(1 - q.astype(float))
            changed = a / (a + b)  # 等价sigmoid(logit(q)/2)，精确保留0和1。
        elif policy == 'coarse_logit':
            if level + 1 == len(cache['strides']):
                continue
            other = cache['q' + str(level + 1)]
            # FPN时间点以同一原点、各层stride为间距；不直接平均数组索引。
            coord = np.arange(q.shape[1]) * float(stride) / cache['strides'][level + 1]
            aligned = np.stack([np.interp(coord, np.arange(other.shape[1]), row) for row in logit(other)])
            changed = sigmoid((logit(q) + aligned) / 2)
            changed[:, coord > other.shape[1] - 1] = q[:, coord > other.shape[1] - 1]
        else:
            raise ValueError(policy)
        assert changed.shape == q.shape and np.isfinite(changed).all()
        result['q' + str(level)] = changed
    return result


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def measure(out, plan, shift, policies):
    from tools.full_readout import score
    from independent_ap import IndependentAP
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    import pandas as pd

    base = plan['base']
    stem = 'native_' + shift
    with np.load(Path(base['inputs']) / (stem + '_input.npz')) as f:
        data = dict(f)
    with np.load(Path(base['cosines']) / (stem + '_cosines.npz')) as f:
        cos, ready = f['cos'], f['ready'].astype(bool)
    assert len(cos) == len(data['labels'])
    scores = np.empty((len(policies), len(cos)))
    start = time.time()
    for n, video in enumerate(data['names']):
        ix = np.flatnonzero(data['idx'] == n)
        file = hashlib.sha256(str(video).encode()).hexdigest() + '.npz'
        with np.load(Path(base['dense']) / 'dense' / 'old' / shift / file) as f:
            cache = dict(f)
        assert str(cache['name']) == video
        boxes, labels = data['boxes'][ix], data['labels'][ix]
        for k, policy in enumerate(policies):
            changed = transform(cache, policy)
            temporal = profiles(changed, boxes, labels)
            contrast = responses(changed, boxes, len(ready))
            scores[k, ix] = final_scores(boxes, data['scores'][ix], labels, cos[ix], ready, temporal, contrast)
        write(out / 'progress.json', dict(shift=shift, videos=n + 1, total=len(data['names']), seconds=time.time() - start))
    original = np.load(Path(base['control_archive']) / 'old' / (stem + '_scores.npy'), mmap_mode='r')[2]
    np.testing.assert_allclose(scores[0], original, atol=1e-8, rtol=1e-6)
    cfg = json.loads((Path(base['inputs']).parents[2] / shift / 'config.json').read_text())
    fields = {'video-id': data['names'][data['idx']], 't-start': data['boxes'][:, 0],
              't-end': data['boxes'][:, 1], 'label': data['labels']}
    gt = load_gt_seg_from_json(cfg['baseline']['annotation_for_offline_evaluation'],
                              split=cfg['tgt_val_split'], action_subsampling='10_majority_classes')
    gt = gt[gt['t-start'] < gt['t-end']].reset_index(drop=True)
    row = {}
    for name, confidence in zip(policies, scores):
        assert np.isfinite(confidence).all() and (confidence >= 0).all()
        values = dict(fields, score=confidence)
        result = score(cfg, values)
        frame = pd.DataFrame(values)
        independent = np.asarray([IndependentAP(gt[gt.label == c].reset_index(drop=True),
            frame[frame.label == c].reset_index(drop=True)).ap(np.ones(int((frame.label == c).sum()), dtype=bool))
            for c in sorted(gt.label.unique())]).mean(0) * 100
        np.testing.assert_allclose(independent, result['by_tiou'], atol=1e-5, rtol=0)
        row[name] = dict(result, independent_AP_passed=True)
    np.testing.assert_allclose(row['control']['by_tiou'], base['sada'][shift]['by_tiou'], atol=1e-5, rtol=0)
    with (out / (shift + '_scores.npy')).open('wb') as f:
        np.save(f, scores)
    write(out / (shift + '.json'), row)
    return row


def main(out):
    from tools.native_ap_fast import install
    torch.set_num_threads(2)
    install()
    plan = json.loads((out / 'plan.json').read_text())
    assert plan['policies'] == list(POLICIES)
    assert (out / 'ours_final.py').read_bytes() and hashlib.sha256((out / 'ours_final.py').read_bytes()).hexdigest() == plan['ours_sha256']
    rows = {}
    for i in range(1, 7):
        shift = 'S' + str(i)
        file = out / (shift + '.json')
        rows[shift] = json.loads(file.read_text()) if file.is_file() else measure(out, plan, shift, POLICIES)
    summary = {}
    for policy in POLICIES:
        ap = np.array([rows[s][policy]['by_tiou'] for s in rows])
        control = np.array([rows[s]['control']['by_tiou'] for s in rows])
        delta = ap - control
        summary[policy] = dict(mean=float(ap.mean()), delta_pp=float(delta.mean()),
            per_shift={s: rows[s][policy]['mAP'] for s in rows},
            per_shift_delta={s: float(v) for s, v in zip(rows, delta.mean(1))},
            by_tiou=ap.mean(0).tolist(), by_tiou_delta=delta.mean(0).tolist(),
            passes=bool(delta.mean() > 0 and delta[:, -1].mean() >= 0))
    write(out / 'summary.json', dict(status='complete', results=summary, training_runs=0,
        selection='all three prespecified uniform rules reported; no per-shift tuning',
        limitation='single seed; same target development videos used in historical method selection'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.out)
