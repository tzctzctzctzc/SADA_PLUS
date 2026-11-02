"""冻结预测上的多规则AP；匹配/排序逐值对齐原EPIC评估，不重复计算IoU。"""
import numpy as np
from numba import njit

TIOUS = np.linspace(.1, .5, 5)
RULES = tuple((float(t), float(s)) for t in np.r_[np.arange(0, 1.001, .025), 1.01]
              for s in (.1, .3, .5, .7, 1.01))


@njit(cache=False)
def _matched_ap(order, pointers, matches, overlaps, n_gt, thresholds):
    ap = np.zeros(len(thresholds)); totals = np.zeros(len(thresholds))
    if not len(order) or not n_gt:
        return ap, totals
    for t in range(len(thresholds)):
        used = np.zeros(n_gt, np.bool_); tp = np.zeros(len(order))
        for rank in range(len(order)):
            p = order[rank]
            for j in range(pointers[p], pointers[p+1]):
                if overlaps[j] < thresholds[t]:
                    break
                g = matches[j]
                if not used[g]:
                    used[g] = True; tp[rank] = 1.; break
        # 原interpolated_prec_rec：score排序后一对一匹配和precision envelope。
        cumulative = 0.; precision = np.zeros(len(order))
        for i in range(len(order)):
            cumulative += tp[i]; precision[i] = cumulative / (i+1)
        totals[t] = cumulative
        envelope = 0.
        for i in range(len(order)-1, -1, -1):
            envelope = max(envelope, precision[i])
            if tp[i] > 0:
                ap[t] += envelope / n_gt
    return ap, totals


class ClassAP:
    def __init__(self, ground_truth, predictions):
        self.prediction = predictions.reset_index(drop=True)
        self.n_gt = len(ground_truth)
        self.scores = self.prediction['score'].to_numpy(dtype=float)
        gt = ground_truth.reset_index(drop=True)
        groups = {v: x.index.to_numpy() for v, x in gt.groupby('video-id')}
        boxes = gt[['t-start', 't-end']].to_numpy(dtype=float)
        pointers, matches, overlaps = [0], [], []
        for row in self.prediction.itertuples(index=False, name=None):
            # 调用端列顺序固定video-id,t-start,t-end,label,score,pbg。
            vid, start, end = row[:3]
            ids = groups.get(vid, np.empty(0, dtype=int)); b = boxes[ids]
            intersection = np.maximum(0, np.minimum(end, b[:,1])-np.maximum(start, b[:,0]))
            iou = intersection / ((end-start)+(b[:,1]-b[:,0])-intersection)
            order = iou.argsort()[::-1]; order = order[iou[order] >= TIOUS[0]]
            matches.extend(ids[order]); overlaps.extend(iou[order]); pointers.append(len(matches))
        self.pointers = np.asarray(pointers, dtype=np.int64)
        self.matches = np.asarray(matches, dtype=np.int64)
        self.overlaps = np.asarray(overlaps, dtype=float)

    def evaluate(self, keep):
        indices = np.flatnonzero(keep)
        # 必须对删后集合重新用原numpy默认argsort；同分时不能沿用全集合排序。
        order = indices[self.scores[indices].argsort()[::-1]]
        ap, tp = _matched_ap(order, self.pointers, self.matches, self.overlaps,
                             self.n_gt, TIOUS)
        return dict(ap=ap.tolist(), map=float(ap.mean()), tp=tp.astype(int).tolist(),
                    kept=int(len(indices)), gt=int(self.n_gt))


def choose_rule(results, baseline, retention=.995):
    """每个tIoU的重新匹配TP数不得损失超过0.5%；同AP优先少删。"""
    feasible = [i for i, r in enumerate(results)
                if all(n >= retention * b for n,b in zip(r['tp'], baseline['tp']))]
    assert feasible, '规则必须包含完全不删的回退'
    return max(feasible, key=lambda i:(results[i]['map'], results[i]['kept'], RULES[i][0], -RULES[i][1]))


def keep_rule(frame, rule):
    threshold, protection = rule
    return ~((frame['pbg'].to_numpy() > threshold) & (frame['score'].to_numpy() < protection))


def prediction_frame(rows):
    import pandas as pd
    columns = {k:[] for k in ('video-id','t-start','t-end','label','score','pbg')}
    for r in rows:
        columns['video-id'].extend([r['video']]*len(r['scores']))
        for name, values in (('t-start',r['segments'][:,0]), ('t-end',r['segments'][:,1]),
                             ('label',r['labels']), ('score',r['scores']), ('pbg',r['pbg'])):
            columns[name].extend(values.tolist())
    return pd.DataFrame(columns)


def calibrate(ground_truth, frame):
    """只对target_train GT调用。按类优化独立，统一阈值也完整保留。"""
    classes = sorted(int(c) for c in ground_truth['label'].unique()); details = {}
    for c in classes:
        f = frame[frame['label']==c].reset_index(drop=True)
        engine = ClassAP(ground_truth[ground_truth['label']==c], f)
        baseline = engine.evaluate(np.ones(len(f), dtype=bool))
        results = [engine.evaluate(keep_rule(f, rule)) for rule in RULES]
        index = choose_rule(results, baseline)
        # 无TP类只回退，不能凭没有召回约束任意选择。
        if not any(baseline['tp']):
            index = RULES.index((1.01, .1))
        details[str(c)] = dict(baseline=baseline, grid=results, selected_index=index,
                               selected_rule=RULES[index])
    feasible = [i for i in range(len(RULES)) if all(
        all(n >= .995*b for n,b in zip(details[str(c)]['grid'][i]['tp'], details[str(c)]['baseline']['tp']))
        for c in classes)]
    best = max(feasible, key=lambda i:(sum(details[str(c)]['grid'][i]['map'] for c in classes),
                                      sum(details[str(c)]['grid'][i]['kept'] for c in classes), RULES[i][0], -RULES[i][1]))
    return dict(classes=classes, global_index=best, global_rule=RULES[best], per_class=details,
                grid=RULES, min_TP_retention_each_tiou=.995,
                objective='target_train_mean_AP_0.1_to_0.5; ties_keep_more_predictions',
                role='target_train_supervised_calibration_Oracle_not_UDA')


def apply_rules(frame, calibration, per_class=False):
    keep = np.ones(len(frame), dtype=bool)
    for c in calibration['classes']:
        positions = np.flatnonzero(frame['label'].to_numpy()==c)
        rule = calibration['per_class'][str(c)]['selected_rule'] if per_class else calibration['global_rule']
        keep[positions] = keep_rule(frame.iloc[positions], rule)
    return keep


def background_mask(ground_truth, frame):
    """只供CPU离线Oracle诊断，anyclass IoU<.01沿旧Oracle定义。"""
    result = np.ones(len(frame), dtype=bool)
    for vid, positions in frame.groupby('video-id').indices.items():
        gt = ground_truth[ground_truth['video-id']==vid][['t-start','t-end']].to_numpy(dtype=float)
        b = frame.iloc[positions][['t-start','t-end']].to_numpy(dtype=float)
        if not len(gt):
            continue
        inter = np.maximum(0, np.minimum(b[:,None,1],gt[None,:,1])-np.maximum(b[:,None,0],gt[None,:,0]))
        union = (b[:,1]-b[:,0])[:,None]+(gt[:,1]-gt[:,0])[None]-inter
        result[positions] = (inter/union).max(1)<.01
    return result
