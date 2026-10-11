"""E1完整有限源池的局部边界信息核验；不包含目标域或TAL训练。"""
import hashlib
import warnings
from functools import lru_cache
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

POLICY = dict(seed=6, gpu=3, folds=5, fps=8., max_frames=200, dimension=128,
              boundary_seconds=.25, C=1., max_iter=1000, bootstrap=10000,
              family_comparisons=6, minimum_gain=.02,
              conditions=['geometry', 'slowfast', 'vlm', 'fusion', 'shifted', 'sf_double'])


def digest(value):
    import json
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def folds(videos):
    unique = sorted(set(videos), key=lambda v: digest([6, 'e1-video-fold', v]))
    if len(unique) < 10:
        raise ValueError('E1 requires at least ten physical videos')
    return {v: i % POLICY['folds'] for i, v in enumerate(unique)}


def labels(times, annotations, class_id):
    """全部同类实例的边界；不依赖proposal匹配或教师正确性。"""
    spans = [a['segment'] for a in annotations if int(a['label_id']) == class_id]
    if not spans:
        return np.zeros((len(times), 2), dtype=np.int64)
    distance = np.abs(np.asarray(times)[:, None, None] - np.asarray(spans)[None, :, :])
    return (distance.min(1) <= POLICY['boundary_seconds']).astype(np.int64)


def temporal_pool(tokens, grid, times, merge_size=2, temporal_patch=2):
    """官方merger输出为T,H/merge,W/merge顺序；只池化空间。"""
    t, h, w = map(int, grid)
    times = np.asarray(times, dtype=np.float64)
    if len(times) != t * temporal_patch or h % merge_size or w % merge_size:
        raise ValueError('time/grid/merge mismatch')
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('non-monotone actual timestamps')
    spatial = h * w // merge_size ** 2
    if tokens.shape[0] != t * spatial or not np.isfinite(tokens).all():
        raise ValueError('visual token grid mismatch')
    return tokens.reshape(t, spatial, -1).mean(1), times.reshape(t, temporal_patch).mean(1)


def slowfast_at(feats, times, dataset):
    stride = dataset['feat_stride'] * dataset['downsample_rate']
    source = feats[::dataset['downsample_rate']]
    centers = (np.arange(len(source)) * stride + .5 * dataset['num_frames']) / dataset['default_fps']
    return np.stack([np.interp(times, centers, source[:, i]) for i in range(source.shape[1])], axis=1).astype('float32')


@lru_cache(maxsize=16)
def projection_matrix(input_width, width, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((input_width, width)).astype('float32') / np.sqrt(width)


def project(x, width, seed):
    return np.asarray(x @ projection_matrix(x.shape[1], width, seed), dtype='float32')


def design(rows):
    """等总维度；proposal几何及类别在所有条件一致。"""
    batches = {k: [] for k in POLICY['conditions']}
    all_y, videos, identities = [], [], []
    for row in rows:
        sf, vlm, times = row['slowfast'], row['vlm'], row['times']
        p = row['candidate']['proposal']; length = p[1] - p[0]
        z = (times - p[0]) / length
        common = np.column_stack((z, np.full(len(z), np.log(length)),
                                  np.exp(-((times-p[0])/.25)**2), np.exp(-((times-p[1])/.25)**2),
                                  np.tile(np.eye(10)[row['candidate']['class_id']], (len(z), 1))))
        a, b = project(sf, 128, 61), project(vlm, 128, 62)
        a64, b64 = project(sf, 64, 63), project(vlm, 64, 64)
        variants = dict(geometry=np.zeros((len(z), 0)), slowfast=a, vlm=b,
                        fusion=np.column_stack((a64, b64)),
                        shifted=np.column_stack((a64, np.roll(b64, max(1, len(b64)//2), axis=0))),
                        sf_double=np.column_stack((a64, project(sf, 64, 65))))
        for name, x in variants.items():
            batches[name].append(np.column_stack((x, common)).astype('float32'))
        all_y.append(row['labels'])
        videos.extend([row['candidate']['video_id']] * len(z))
        identities.extend([(row['id'], float(t)) for t in times])
    return {k: np.concatenate(v) for k, v in batches.items()}, np.concatenate(all_y), np.array(videos), identities


def video_weights(videos):
    unique, counts = np.unique(videos, return_counts=True)
    sizes = dict(zip(unique, counts))
    return np.array([1. / sizes[v] for v in videos])


def crossfit(x, y, videos, mapping):
    pred = np.zeros_like(y, dtype='float64'); diagnostics = []
    fold_ids = np.array([mapping[v] for v in videos])
    for fold in range(POLICY['folds']):
        train, test = fold_ids != fold, fold_ids == fold
        if set(videos[train]) & set(videos[test]):
            raise ValueError('video leakage')
        scaler = StandardScaler().fit(x[train])
        train_x, test_x = scaler.transform(x[train]), scaler.transform(x[test])
        weights = video_weights(videos[train]); weights *= len(weights)/weights.sum()
        iterations = []
        for endpoint in range(2):
            if len(np.unique(y[train, endpoint])) != 2:
                raise ValueError('missing boundary class in training fold')
            clf = LogisticRegression(C=POLICY['C'], max_iter=POLICY['max_iter'],
                                     solver='lbfgs', random_state=6, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                clf.fit(train_x, y[train, endpoint], sample_weight=weights)
            pred[test, endpoint] = clf.predict_proba(test_x)[:, 1]
            iterations.append(int(clf.n_iter_[0]))
        diagnostics.append(dict(fold=fold, train_videos=sorted(set(videos[train])),
                                test_videos=sorted(set(videos[test])), iterations=iterations))
    return pred, diagnostics


def ap(y, pred, weights):
    return [float(average_precision_score(y[:, j], pred[:, j], sample_weight=weights)) for j in range(2)]


def ap_order(y, pred):
    orders = []
    for j in range(2):
        order = np.argsort(-pred[:, j], kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(pred[order, j]) != 0), len(order)-1]
        orders.append((order, ends, y[order, j]))
    return orders


def ordered_ap(orders, weights):
    """固定预测只排序一次；精确保留分数并列与零权重的视频bootstrap。"""
    values = []
    for order, ends, positives in orders:
        w = weights[order]
        tp = np.cumsum(w*positives)[ends]; total = np.cumsum(w)[ends]
        precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
        values.append(float(np.sum(np.diff(np.r_[0., tp])*precision)/tp[-1]))
    return float(np.mean(values))


def summarize(y, predictions, videos):
    weights = video_weights(videos)
    scores = {k: dict(AP=ap(y, p, weights)) for k, p in predictions.items()}
    for value in scores.values(): value['mean_AP'] = float(np.mean(value['AP']))
    # 分组bootstrap只重采样视频，保留该视频所有候选/时间点的依赖。
    names = ['slowfast', 'sf_double', 'shifted']
    unique = sorted(set(videos)); indices = {v: i for i, v in enumerate(unique)}
    row_vid = np.array([indices[v] for v in videos]); rng = np.random.default_rng(6)
    boot = {name: [] for name in names}
    orders = {name: ap_order(y, predictions[name]) for name in ['fusion']+names}
    for _ in range(POLICY['bootstrap']):
        counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = weights * counts[row_vid]
        # 无正例抽样不定义AP，不能悄悄删掉；完整池应有足够视频支持。
        if any(np.sum(w*y[:, j]) == 0 for j in range(2)):
            raise ValueError('bootstrap has no positive boundary support')
        fusion = ordered_ap(orders['fusion'], w)
        for name in names: boot[name].append(float(fusion - ordered_ap(orders[name], w)))
    tail = .05 / POLICY['family_comparisons'] / 2
    comparisons = {}
    for name in names:
        comparisons[name] = dict(delta=scores['fusion']['mean_AP']-scores[name]['mean_AP'],
                                 interval=np.quantile(boot[name], [tail, 1-tail]).tolist())
    passed = all(v['interval'][0] > 0 for v in comparisons.values()) and all(
        comparisons[k]['delta'] >= POLICY['minimum_gain'] for k in ('slowfast', 'sf_double'))
    return dict(scores=scores, comparisons=comparisons, positive_signal=bool(passed),
                inference='Exploratory video-cross-fitted source information; not target TAL mAP or untouched confirmation',
                video_count=len(unique), rows=len(y), positive_counts=y.sum(0).tolist())
