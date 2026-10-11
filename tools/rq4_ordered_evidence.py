"""E2固定±0.5秒的有序变化读出；复用E1时间网格，不读取目标结果。"""
import numpy as np
from tools import rq4_local_evidence as e1

CONDITIONS = ['geometry', 'static_sf', 'static_fusion', 'ordered_sf', 'vlm',
              'fusion', 'sf_double', 'shifted', 'order_shuffled']
COMPARISONS = ['ordered_sf', 'sf_double', 'static_fusion', 'shifted', 'order_shuffled']
POLICY = dict(e1.POLICY, experiment='E2-ordered-cached-visual-v1', window_seconds=.5,
              conditions=CONDITIONS, family_comparisons=10,
              order_control='label-independent pointwise before/after swap; cached tokens only')


def context(x, times):
    """只取真实时间戳内的邻点；片段边缘无邻点时复制中心，不补造观测。"""
    x = np.asarray(x, dtype=np.float32); times = np.asarray(times)
    if len(x) != len(times) or not np.isfinite(x).all() or not np.isfinite(times).all():
        raise ValueError('invalid feature/time array')
    if np.any(np.diff(times) < 0):
        raise ValueError('unordered timestamps')
    before, after = [], []
    for i, t in enumerate(times):
        left = (times >= t-.5) & (times < t)
        right = (times > t) & (times <= t+.5)
        before.append(x[left].mean(0) if left.any() else x[i])
        after.append(x[right].mean(0) if right.any() else x[i])
    return np.asarray(before), np.asarray(after)


def ordered(x, times, width, seed, swap=None):
    # 同一个投影用于中心/前/后，差分在相同坐标系；总维度固定。
    center = e1.project(x, width//4, seed)
    before, after = context(center, times)
    if swap is not None:
        before, after = np.where(swap[:, None], after, before), np.where(swap[:, None], before, after)
    return np.column_stack((center, before, after, after-before)).astype('float32')


def design(rows):
    batches = {k: [] for k in CONDITIONS}; ys, videos, identities = [], [], []
    for row in rows:
        sf, vlm, times = row['slowfast'], row['vlm'], row['times']
        p = row['candidate']['proposal']; length = p[1]-p[0]; z = (times-p[0])/length
        common = np.column_stack((z, np.full(len(z), np.log(length)),
            np.exp(-((times-p[0])/.25)**2), np.exp(-((times-p[1])/.25)**2),
            np.tile(np.eye(10)[row['candidate']['class_id']], (len(z), 1))))
        a, b = ordered(sf, times, 64, 63), ordered(vlm, times, 64, 64)
        rng = np.random.default_rng(int(e1.digest(['E2-order', row['id'], 6])[:16], 16))
        swap = rng.integers(0, 2, len(times)).astype(bool)
        variants = dict(geometry=np.empty((len(times), 0)),
            static_sf=e1.project(sf, 128, 61),
            static_fusion=np.column_stack((e1.project(sf, 64, 63), e1.project(vlm, 64, 64))),
            ordered_sf=ordered(sf, times, 128, 61), vlm=ordered(vlm, times, 128, 62),
            fusion=np.column_stack((a, b)),
            sf_double=np.column_stack((a, ordered(sf, times, 64, 65))),
            shifted=np.column_stack((a, np.roll(b, max(1, len(b)//2), axis=0))),
            order_shuffled=np.column_stack((a, ordered(vlm, times, 64, 64, swap))))
        for name, features in variants.items():
            batches[name].append(np.column_stack((features, common)).astype('float32'))
        ys.append(row['labels']); videos.extend([row['candidate']['video_id']]*len(times))
        identities.extend([(row['id'], float(t)) for t in times])
    return {k: np.concatenate(v) for k, v in batches.items()}, np.concatenate(ys), np.array(videos), identities


def summarize(y, predictions, videos, iterations=10000):
    weights = e1.video_weights(videos)
    scores = {k: dict(AP=e1.ap(y, p, weights)) for k, p in predictions.items()}
    for value in scores.values(): value['mean_AP'] = float(np.mean(value['AP']))
    unique, row_vid = np.unique(videos, return_inverse=True)
    orders = {k: e1.ap_order(y, predictions[k]) for k in ['fusion']+COMPARISONS}
    boot = {k: [] for k in COMPARISONS}; rng = np.random.default_rng(6)
    for _ in range(iterations):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = weights*counts[row_vid]
        if any(np.sum(w*y[:, j]) == 0 for j in range(2)): raise ValueError('bootstrap lacks positives')
        fusion = e1.ordered_ap(orders['fusion'], w)
        for k in COMPARISONS: boot[k].append(fusion-e1.ordered_ap(orders[k], w))
    tail = .05/POLICY['family_comparisons']/2
    comparisons = {k: dict(delta=scores['fusion']['mean_AP']-scores[k]['mean_AP'],
        interval=np.quantile(boot[k], [tail, 1-tail]).tolist()) for k in COMPARISONS}
    passed = all(v['interval'][0] > 0 for v in comparisons.values()) and all(
        comparisons[k]['delta'] >= .02 for k in ['ordered_sf', 'sf_double'])
    return dict(scores=scores, comparisons=comparisons, positive_signal=bool(passed),
        video_count=len(unique), rows=len(y), positive_counts=y.sum(0).tolist(),
        inference='Exploratory source crossfit on previously used pool; cached-feature order, not raw-frame re-encoding')
