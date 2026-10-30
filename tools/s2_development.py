"""S2目标GT辅助开发：冻结视频划分，点级诊断，开发选择后才核验留出集。"""
from copy import deepcopy
import hashlib
import math

from tools.baseline_protocol import read, save, sha

LEVELS = dict(repository=[.5, 0., .2, .3, .3, .3], paper=[.7, .3, .9, .8, .7, 0.])
THRESHOLDS = (.3, .5, .7)
GRID = {family+'_t%02d'%round(t*10): dict(levels=weights, threshold=t)
        for family, weights in LEVELS.items() for t in THRESHOLDS}
REUSED = 'repository_t05'


def changed_config(old, candidate, identity, tmp):
    cfg = deepcopy(old)
    for i, value in enumerate(GRID[candidate]['levels']):
        cfg['model']['weight_level_%d'%i] = value
    cfg['model']['pl_threshold'] = GRID[candidate]['threshold']
    cfg['baseline'].update(code=identity, tmpdir=str(tmp))
    return cfg


def assert_config(cfg, old, candidate, method):
    from libs.models.SADA.prototype_capacity import specification
    if method not in ('uda', 'prototype') or candidate not in GRID:
        raise ValueError('只允许冻结的S2六套配对配置')
    if old['baseline']['shift'] != 'S2' or old['baseline']['regime'] != 'uda':
        raise ValueError('方向或方法身份改变')
    if old['seed'] != 123 or old['loader']['batch_size'] != 2:
        raise ValueError('必须是seed123/batch2')
    if cfg != changed_config(old, candidate, cfg['baseline']['code'], cfg['baseline']['tmpdir']):
        raise ValueError('除层权重/阈值/运行身份外发生变化')
    if method == 'prototype':
        cap = cfg['prototype_capacity']
        if cap['variant'] != 'control' or cap['specification'] != specification('control'):
            raise ValueError('必须保留128维原版Prototype与0.1权重')
    elif 'prototype_capacity' in cfg:
        raise ValueError('UDA不能带辅助分支')


def freeze_split(cfg):
    annotation = cfg['baseline']['annotation_for_offline_evaluation']
    db = read(annotation)['database']
    ids = [v for v, row in db.items() if row['subset'].lower() in cfg['tgt_val_split']]
    ids.sort(key=lambda v: hashlib.sha256(('s2-oracle-dev-v1:123:'+v).encode()).hexdigest())
    dev, confirm = ids[:len(ids)//2], ids[len(ids)//2:]
    training = {v for v, row in db.items() if row['subset'].lower() in
                cfg['tgt_train_split']+cfg['src_train_split']+cfg['src_val_split']}
    if not dev or not confirm or set(ids) & training or len(ids) != len(set(ids)):
        raise ValueError('视频成员重叠/缺失')
    for subset in (dev, confirm):
        classes = {int(a['label_id']) for v in subset for a in db[v]['annotations']
                   if int(a['label_id']) in range(10)}
        if classes != set(range(10)):
            raise ValueError('固定划分未覆盖全部10类，不能按结果重抽')
    return dict(annotation=annotation, annotation_sha256=sha(annotation),
                target_validation_splits=cfg['tgt_val_split'],
                development=dev, confirmation=confirm, seed=123,
                rule='sha256(s2-oracle-dev-v1:123:video_id),first-half-development',
                target_gt_used_for_selection=True, target_detection_supervision=False,
                prior_full_target_metric_exposure=True,
                interpretation='exploratory-development-split-not-an-unseen-benchmark')


def subset_score(prediction_path, split, videos):
    import numpy as np
    from libs.evaluation import ANETdetection
    if sha(split['annotation']) != split['annotation_sha256']:
        raise ValueError('评估GT在冻结后改变')
    prediction = read(prediction_path)
    if prediction['annotation_sha256'] != split['annotation_sha256']:
        raise ValueError('导出预测与GT身份不配对')
    if not set(videos) <= set(prediction['video_ids']):
        raise ValueError('导出缺少核验视频')
    columns = prediction['columns']
    positions = [i for i, v in enumerate(columns['video-id']) if v in set(videos)]
    chosen = {k: np.asarray([v[i] for i in positions]) for k, v in columns.items()}
    evaluator = ANETdetection(split['annotation'], split['target_validation_splits'], num_workers=4,
                              action_subsampling='10_majority_classes')
    # 使用完整评估的10类映射，不按子集重新编号。
    evaluator.ground_truth = evaluator.ground_truth[evaluator.ground_truth['video-id'].isin(videos)].copy()
    if set(evaluator.ground_truth['label']) != set(range(10)):
        raise ValueError('核验子集缺类')
    ap, mean, *_ = evaluator.evaluate(chosen, verbose=False)
    return dict(map=float(mean), map_by_tiou=[float(x) for x in ap], videos=len(videos),
                predictions=len(positions), prediction_sha256=sha(prediction_path))


def select_candidate(rows, so):
    if set(rows) != set(GRID) or not math.isfinite(so):
        raise ValueError('必须完成精确六套配置后选择')
    enriched = {}
    for name, row in rows.items():
        u, p = row['uda'], row['prototype']
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in (u, p)):
            raise ValueError('开发mAP不合法')
        enriched[name] = dict(row, uda_minus_so_pp=100*(u-so), prototype_minus_uda_pp=100*(p-u))
    # 不用降低UDA来制造顺序：先满足两个严格正差，再在其中最大化UDA。
    eligible = [k for k, v in enriched.items() if v['uda'] > so and v['prototype'] > v['uda']]
    rank = lambda k: (-enriched[k]['uda'], k)
    best_uda = sorted(enriched, key=rank)[0]
    selected = sorted(eligible, key=rank)[0] if eligible else best_uda
    return dict(selected=selected, best_uda=best_uda, ordering_found=bool(eligible),
                eligibility='UDA>SO and Prototype>UDA', objective='maximize-UDA-within-eligible',
                rows=enriched, so=so, used_split='development-only')


def point_counts(probabilities, gt, threshold):
    """GT多标签点中预测到任一正确类计真；阈值使用官方的>=。"""
    confidence, predicted = probabilities.max(dim=1)
    foreground = gt.any(dim=1)
    accepted = confidence >= threshold
    correct = accepted & gt.gather(1, predicted[:, None]).squeeze(1).bool()
    return dict(valid=int(len(gt)), gt_foreground=int(foreground.sum()),
        predicted_foreground=int(accepted.sum()), correct_foreground=int(correct.sum()),
        foreground_predicted_background=int((foreground & ~accepted).sum()),
        background_predicted_foreground=int((~foreground & accepted).sum()),
        wrong_foreground_class=int((foreground & accepted & ~correct).sum()))


class PointObserver:
    """只读整视频诊断，复用真实label_points，不模拟训练crop覆盖。"""
    def __init__(self, development):
        self.videos = set(development)
        self.counts, self.seen, self.handles = {}, [], []
        self.enabled = False

    def attach(self, model):
        if not (model.cls_head is model.cls_head_target is model.cls_head_pl):
            raise ValueError('诊断只接受本次共享分类头架构')
        self.model = model
        def head_hook(module, args, output):
            if self.enabled:
                self.heads.append((output[0], args[1]))
        def point_hook(module, args, output):
            if self.enabled: self.points.append(output)
        self.handles = [model.cls_head.register_forward_hook(head_hook),
                        model.point_generator.register_forward_hook(point_hook)]

    def before(self, domain, videos):
        self.enabled = domain == 'target' and videos[0]['video_id'] in self.videos
        self.heads, self.points = [], []

    def after(self, domain, videos):
        if not self.enabled: return
        if len(videos) != 1 or len(self.heads) != 3 or not self.points:
            raise ValueError('无法识别真实target分类调用')
        # 第2次才是域对齐实际使用的target head；第3次是PL网络支路。
        logits, masks = self.heads[1]
        points = self.points[0]
        gt, _ = self.model.label_points(points, [videos[0]['segments']], [videos[0]['labels']])
        targets = gt[0].split([len(p) for p in points])
        for level, (values, mask, target) in enumerate(zip(logits, masks, targets)):
            valid = mask[0, 0].bool()
            probabilities = values[0].transpose(0, 1)[valid].sigmoid()
            target = target[valid].bool()
            for threshold in THRESHOLDS:
                key = '%d/%.1f'%(level, threshold)
                counts = point_counts(probabilities, target, threshold)
                total = self.counts.setdefault(key, {k: 0 for k in counts})
                for key2, value in counts.items(): total[key2] += value
        self.seen.append(videos[0]['video_id'])
        self.heads, self.points = [], []
        self.enabled = False

    def finish(self, path):
        for handle in self.handles: handle.remove()
        if set(self.seen) != self.videos or len(self.seen) != len(self.videos):
            raise ValueError('开发点级诊断缺少视频或重复')
        for row in self.counts.values():
            row['precision'] = row['correct_foreground']/max(1, row['predicted_foreground'])
            row['correct_class_recall'] = row['correct_foreground']/max(1, row['gt_foreground'])
        save(path, dict(scope='full-video-valid-FPN-points-not-random-training-crops',
            assignment='unmodified-SADA-label_points', target_head_call=2,
            videos=sorted(self.seen), per_level_threshold=self.counts))
