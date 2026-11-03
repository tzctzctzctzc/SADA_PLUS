"""独立NumPy匹配重算固定预算；不加载GPU，不选择操作点。"""
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from libs.evaluation.evaluator_anet import compute_average_precision_detection, load_gt_seg_from_json

RATES = (.005, .01, .02, .05)
THRESHOLDS = np.linspace(.1, .5, 5)
SOURCE = Path('/vepfs/vlms/foreground_calibration_oracle6_a4df72e_1103')
PILOT = Path('/vepfs/vlms/proposal_foreground_source_pilot6_eb30065_1103')
OLD = Path('/vepfs/vlms/hard_background_pilots24_8e80d74_1103')


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True))
    tmp.replace(p)


def overlaps(boxes, truth):
    if not len(truth):
        return np.zeros((len(boxes), 0))
    intersection = np.maximum(0, np.minimum(boxes[:, None, 1], truth[None, :, 1]) - np.maximum(boxes[:, None, 0], truth[None, :, 0]))
    return intersection / np.maximum((boxes[:, 1] - boxes[:, 0])[:, None] + (truth[:, 1] - truth[:, 0])[None, :] - intersection, 1e-8)


class IndependentAP:
    def __init__(self, gt, pred):
        self.scores = pred.score.to_numpy(dtype=float)
        self.n = len(gt)
        self.options = [[] for _ in range(len(pred))]
        g = gt.reset_index(drop=True)
        for video, ids in pred.groupby('video-id', sort=False).indices.items():
            gids = np.flatnonzero(g['video-id'].to_numpy() == video)
            matrix = overlaps(pred.iloc[ids][['t-start', 't-end']].to_numpy(dtype=float), g.iloc[gids][['t-start', 't-end']].to_numpy(dtype=float))
            for pos, row in zip(ids, matrix):
                order = np.argsort(row)[::-1]
                self.options[pos] = [(int(gids[j]), float(row[j])) for j in order if row[j] >= .1]

    def ap(self, keep):
        ids = np.flatnonzero(keep)
        ordered = ids[np.argsort(self.scores[ids])[::-1]]
        if not self.n or not len(ids):
            return np.zeros(5)
        result = []
        for threshold in THRESHOLDS:
            assigned = set()
            hit = np.zeros(len(ids))
            for rank, pos in enumerate(ordered):
                for identity, value in self.options[pos]:
                    if value < threshold:
                        break
                    if identity not in assigned:
                        assigned.add(identity)
                        hit[rank] = 1
                        break
            precision = np.cumsum(hit) / np.arange(1, len(hit) + 1)
            envelope = np.maximum.accumulate(precision[::-1])[::-1]
            result.append(float(envelope[hit.astype(bool)].sum() / self.n))
        return np.asarray(result)


def official(gt, pred):
    if not len(gt):
        return np.zeros(5)
    result = compute_average_precision_detection(gt.reset_index(drop=True), pred.reset_index(drop=True), THRESHOLDS)
    return result[0] if isinstance(result, tuple) else result


def make_frame(rows):
    return pd.DataFrame([(r['video'], float(b[0]), float(b[1]), int(c), float(s), float(p))
        for r in rows for b, c, s, p in zip(r['segments'], r['labels'], r['scores'], r['pbg'])],
        columns=['video-id', 't-start', 't-end', 'label', 'score', 'pbg'])


def background(gt, pred):
    result = np.ones(len(pred), dtype=bool)
    for video, ids in pred.groupby('video-id', sort=False).indices.items():
        truth = gt[gt['video-id'] == video][['t-start', 't-end']].to_numpy(dtype=float)
        matrix = overlaps(pred.iloc[ids][['t-start', 't-end']].to_numpy(dtype=float), truth)
        if len(truth):
            result[ids] = matrix.max(axis=1) < .01
    return result


def selected_indices(frame, rate, confidence=False):
    selected = []
    for label in range(10):
        pool = [i for i, (c, s) in enumerate(zip(frame.label, frame.score)) if c == label and s >= .3]
        if confidence:
            pool.sort(key=lambda i: (float(frame.score.iloc[i]), i))
        else:
            pool.sort(key=lambda i: (-float(frame.pbg.iloc[i]), i))
        selected.extend(pool[:int(len(pool) * rate)])
    return np.asarray(sorted(selected), dtype=int)


def audit_shift(root, shift):
    entry = root / shift
    done = read(entry / 'complete.json')
    prior = read(root / 'independent_GPU_and_fixed20_audit.json')['results'][shift]
    assert sha(entry / 'gpu_complete.json') == done['gpu_complete_sha256'] == prior['gpu_complete_sha256']
    for p, digest in done['artifacts'].items():
        assert sha(p) == digest
    cost = read(entry / 'costs.json')
    idx = read(SOURCE / shift / 'cache_index.json')
    truth = read(SOURCE / shift / 'source_truth.json')
    assert sha(SOURCE / shift / 'cache_index.json') == cost['source_index_sha256']
    assert sha(SOURCE / shift / 'source_truth.json') == cost['source_truth_sha256']
    records = {r['video']: r for r in idx['records']}
    assert len(records) == len(cost['rows']) == len(truth)
    assert set(records) == set(truth) == set(read(SOURCE / shift / 'selection.json')['source_fit'])
    validation = [r for r in read(PILOT / shift / 'cache_index.json')['records'] if r['role'] == 'validation']
    assert cost['validation'] == validation
    assert not set(records) & {r['video'] for r in validation}
    # 已有全源缓存SHA审计绑定，不反复hash大FPN。逐行独立核全部标签和hard身份。
    spot = {}
    for row in cost['rows']:
        record = row['record']
        assert record == records[record['video']]
        data = torch.load(record['path'], map_location='cpu')
        hard = np.flatnonzero(data['scores'].numpy() >= .3)
        assert hard.tolist() == row['hard_indices']
        gt = truth[record['video']]
        boxes = data['boxes'].numpy()[hard]
        matrix = overlaps(boxes, np.asarray(gt['segments'], dtype=np.float32).reshape(-1, 2))
        bg = matrix.max(1) < .01 if len(gt['labels']) else np.ones(len(hard), dtype=bool)
        assert np.array_equal(bg, row['bg'])
        q = np.where(data['labels'].numpy()[hard, None] == np.asarray(gt['labels'])[None, :], matrix, 0).max(1) if len(gt['labels']) else np.zeros(len(hard))
        assert np.allclose(q, row['quality'], atol=2e-6, rtol=0)
        for kind, candidates in [('beneficial_BG', np.flatnonzero(bg & (np.asarray(row['delta']) > 0))), ('harmful_nonBG', np.flatnonzero(~bg & (np.asarray(row['delta']) < 0)))]:
            if kind in spot or not len(candidates):
                continue
            j = int(candidates[0]); original_id = int(hard[j]); label = int(data['labels'][original_id])
            all_boxes = data['boxes'].numpy()
            p = pd.DataFrame({'video-id': record['video'], 't-start': all_boxes[:, 0], 't-end': all_boxes[:, 1], 'label': data['labels'].numpy(), 'score': data['scores'].numpy()})
            g = pd.DataFrame({'video-id': record['video'], 't-start': [b[0] for b in gt['segments']], 't-end': [b[1] for b in gt['segments']], 'label': gt['labels']})
            p = p[p.label == label]; g = g[g.label == label]
            delta = float(official(g, p.drop(original_id)).mean() - official(g, p).mean())
            assert abs(delta - row['delta'][j]) < 1e-7
            spot[kind] = dict(video=record['video'], index=original_id, delta=delta)
    assert len(spot) == 2
    predicted = torch.load(entry / 'source_val_predictions.pt', map_location='cpu')
    old = torch.load(OLD / shift / 'source_val_predictions.pt', map_location='cpu')['conditional']
    old = {r['video']: r for r in old}
    assert set(old) == {r['video'] for r in validation}
    for mode, rows in predicted.items():
        assert {r['video'] for r in rows} == set(old) and len(rows) == len(old)
        for row in rows:
            for key in ('segments', 'scores', 'labels'):
                assert torch.equal(row[key], old[row['video']][key])
            assert torch.isfinite(row['pbg']).all() and (row['pbg'][row['scores'] < .3] == 0).all()
    cfg = read(SOURCE / shift / 'config.json')
    gt = load_gt_seg_from_json(cfg['dataset']['json_file'], cfg['src_val_split'], action_subsampling='10_majority_classes')
    gt = gt[gt['t-start'] != gt['t-end']].reset_index(drop=True)
    frame = make_frame(predicted['control'])
    assert frame.drop(columns='pbg').equals(make_frame(predicted['ap_cost']).drop(columns='pbg'))
    engines = {}; positions = {}; baseline = []
    old_result = read(OLD / shift / 'complete.json')['results']['conditional']['source_val']['classes']
    for c in range(10):
        pos = np.flatnonzero(frame.label.to_numpy() == c); positions[c] = pos
        engines[c] = IndependentAP(gt[gt.label == c], frame.iloc[pos].reset_index(drop=True))
        base = engines[c].ap(np.ones(len(pos), dtype=bool))
        assert np.allclose(base, old_result[str(c)]['original']['ap'], atol=1e-7, rtol=0)
        baseline.append(base.tolist())
    bg = background(gt, frame); results = {}
    for mode, rows in predicted.items():
        f = make_frame(rows); actual = done['results'][mode]
        assert abs(np.mean(baseline) - actual['baseline']) < 1e-7
        details = []
        for rate, reported in zip(RATES, actual['all_fixed_budgets']):
            assert rate == reported['rate']
            selected = selected_indices(f, rate); confidence = selected_indices(f, rate, True)
            assert len(selected) == len(confidence) and (f.iloc[selected].score >= .3).all()
            arrays = []
            for chosen in (selected, confidence):
                keep = np.ones(len(f), dtype=bool); keep[chosen] = False
                arrays.append([engines[c].ap(keep[positions[c]]).tolist() for c in range(10)])
            assert abs(np.mean(arrays[0]) - reported['map']) < 1e-7
            assert abs(np.mean(arrays[1]) - reported['confidence_map']) < 1e-7
            assert int(bg[selected].sum()) == reported['BG_removed']
            assert int((~bg[selected]).sum()) == reported['nonBG_removed']
            assert int(bg[confidence].sum()) == reported['confidence_BG_removed']
            assert int((~bg[confidence]).sum()) == reported['confidence_nonBG_removed']
            details.append(dict(rate=rate, AP=arrays[0], confidence_AP=arrays[1], selected_indices=selected.tolist(), confidence_indices=confidence.tolist(), **{k:reported[k] for k in ['delta_pp','confidence_delta_pp','BG_removed','nonBG_removed']}))
        results[mode] = details
    report = dict(passed=True, complete_sha256=sha(entry/'complete.json'), original_arrays_exact=True, original_per_class_AP=baseline, full_source_labels_checked=True, official_cost_spots=spot, results=results)
    save(root/'independent_results'/f'{shift}.json', report)
    print('AUDITED', shift, flush=True)
    return report


def run(root):
    results = {f'S{i}': audit_shift(root, f'S{i}') for i in range(1, 7)}
    final = read(root/'final.json')
    assert set(final['results']) == set(results)
    for s in results:
        assert final['results'][s] == read(root/s/'complete.json')
        assert final['complete_sha256'][s] == results[s]['complete_sha256']
    means = {m:[float(np.mean([results[s]['results'][m][i]['delta_pp'] for s in results])) for i in range(4)] for m in ('control','ap_cost')}
    means['confidence'] = [float(np.mean([results[s]['results']['control'][i]['confidence_delta_pp'] for s in results])) for i in range(4)]
    save(root/'independent_final_audit.json', dict(passed=True, source_only=True, fixed_rates=RATES, mean_delta_pp=means, per_shift_audit_sha256={s:sha(root/'independent_results'/f'{s}.json') for s in results}, final_sha256=sha(root/'final.json')))
    print(json.dumps(means), flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]))
