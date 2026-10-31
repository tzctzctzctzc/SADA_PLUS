"""源GT区间的上下文混合：不读取目标标注，不增加网络参数或推理路径。"""
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile

import numpy as np
import torch
from torch.nn import functional as F

from tools.baseline_protocol import assert_no_target_gt, read, save, sha

SCHEMA = 'prototype-context-mix-v1'
VARIANTS = ('source_context', 'target_context', 'target_context_pair')


def specification(variant):
    if variant not in VARIANTS:
        raise ValueError('未知上下文实验组')
    return dict(schema=SCHEMA, variant=variant, context='source' if variant == 'source_context' else 'target',
                interval_steps=4, mix_weight=.1, pair_weight=.1 if variant == 'target_context_pair' else 0.,
                pair_margin=1., context_fraction=[.25, .5],
                gt_coordinates='continuous_feature_grid_translation_only',
                supervision='source_train_task_gt_inside_inserted_patch_only',
                inference='unchanged', independent_rng=True, cache_bytes=536870912)


def feature_length(path, downsample=1):
    # 只读NPY头，避免准备阶段重复解压/读取全部冻结特征。
    with zipfile.ZipFile(path) as archive, archive.open('feats.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, _, _ = np.lib.format._read_array_header(stream, version)
    return (int(shape[0]) + downsample - 1) // downsample


def background_gaps(segments, length):
    intervals = sorted((max(0., float(a)), min(float(length), float(b)))
                       for a, b in segments if b > 0 and a < length)
    cursor, gaps = 0., []
    for a, b in intervals:
        if a > cursor:
            lo, hi = math.ceil(cursor), math.floor(a)
            if hi > lo:
                gaps.append([lo, hi])
        cursor = max(cursor, b)
    if cursor < length:
        lo = math.ceil(cursor)
        if length > lo:
            gaps.append([lo, int(length)])
    return gaps


def patch_record(segments, labels, row, length, margin):
    """仅切口取整；保留全部相交任务标注，跨切口或非背景过渡则拒绝。"""
    a, b = map(float, segments[row])
    lo, hi = math.floor(a) - margin, math.ceil(b) + margin
    if not 0 <= lo < a < b < hi <= length:
        return None
    contained = []
    for index, (s, e) in enumerate(segments):
        if e <= lo or s >= hi:
            continue
        if s < lo or e > hi:
            return None
        if index != row and (s < a or e > b):
            return None
        contained.append(index)
    return dict(start=lo, end=hi, primary=row, class_id=int(labels[row]),
                main=[a-lo, b-lo], segments=[[float(s)-lo, float(e)-lo] for i, (s, e)
                in enumerate(segments) if i in contained], labels=[int(labels[i]) for i in contained])


def prepare_table(cfg, inventory_path, output):
    source_path = Path(cfg['dataset']['json_file'])
    target_path = Path(cfg['baseline']['target_manifest'])
    if sha(source_path) != cfg['baseline']['source_manifest_sha256']:
        raise ValueError('源标注SHA改变')
    if sha(target_path) != cfg['baseline']['target_manifest_sha256']:
        raise ValueError('无标注目标SHA改变')
    source_db, target_db = read(source_path)['database'], read(target_path)['database']
    if any('annotations' in v for v in target_db.values()):
        raise ValueError('目标训练元数据必须无annotations')
    train = [v for v, x in source_db.items() if x['subset'].lower() in cfg['src_train_split']]
    validation = [v for v, x in source_db.items() if x['subset'].lower() in cfg['src_val_split']]
    if set(train) & (set(target_db) | set(validation)):
        raise ValueError('供体成员交叉')
    options = cfg['dataset']
    if options['action_subsampling'] != '10_majority_classes':
        raise ValueError('这里只支持已冻结的10个任务类')
    inventory = {x['path']: x for x in read(inventory_path)}
    videos, total, patches = [], 0, []
    for vid in train:
        item = source_db[vid]
        path = Path(options['feat_folder']) / ((options.get('file_prefix') or '') + vid + options['file_ext'])
        if str(path) not in inventory:
            raise ValueError('供体不在冻结特征清单: '+str(path))
        stride = options['feat_stride'] * options['downsample_rate']
        fps = options.get('default_fps') or item['fps']
        acts = [a for a in item.get('annotations', []) if 0 <= int(a['label_id']) < 10]
        # 完全沿用EpicDataset的float32时间单位转换；绝不round GT。
        sec = np.array([a['segment'] for a in acts], dtype=np.float32).reshape(-1, 2)
        grid = (sec * fps - .5 * options['num_frames']) / stride
        order = np.argsort(grid[:, 0], kind='stable')
        segments = grid[order].tolist()
        labels = [int(acts[i]['label_id']) for i in order]
        length = feature_length(path, options['downsample_rate'])
        video = dict(video_id=vid, path=str(path), length=length, segments=segments, labels=labels,
                     feature_sha256=inventory[str(path)]['sha256'],
                     gaps=background_gaps(segments, length))
        index = len(videos)
        videos.append(video)
        for row, (a, b) in enumerate(segments):
            total += 1
            # 保留至少一个原生SlowFast感受窗半径，动作边界不能成为外接缝。
            radius = max(1, math.ceil(.5 * options['num_frames'] / stride))
            low, high = max(radius, math.ceil((b-a)*.25)), max(radius, math.ceil((b-a)*.5))
            for margin in sorted(set((low, high))):
                record = patch_record(segments, labels, row, length, margin)
                if record:
                    patches.append(dict(video=index, **record))
    backgrounds = [dict(video=i, start=a, end=b) for i, v in enumerate(videos) for a, b in v['gaps']]
    longest_bg = max((x['end']-x['start'] for x in backgrounds), default=0)
    patches = [p for p in patches if p['end']-p['start'] <= longest_bg
               and p['end']-p['start'] <= options['max_seq_len']]
    if not patches:
        raise ValueError('没有合法源动作/背景替换对，不能静默零干预')
    histogram = {str(c): sum(p['class_id'] == c for p in patches) for c in range(10)}
    table = dict(schema=SCHEMA, shift=cfg['baseline']['shift'], source_train=train,
                 source_validation_excluded=validation, target_train=list(target_db),
                 source_manifest_sha256=sha(source_path), target_manifest_sha256=sha(target_path),
                 inventory_path=str(inventory_path), inventory_sha256=sha(inventory_path),
                 dataset=deepcopy(options), videos=videos, patches=patches, backgrounds=backgrounds,
                 summary=dict(total_task_actions=total, patch_count=len(patches),
                              unique_eligible_actions=len({(p['video'], p['primary']) for p in patches}),
                              class_patch_counts=histogram, background_count=len(backgrounds)))
    save(output, table)
    return table


class FeatureBank:
    def __init__(self, table, maximum_bytes=536870912):
        self.table, self.maximum_bytes = table, maximum_bytes
        self.cache, self.bytes = OrderedDict(), 0
        self.verified = {}

    def load(self, index):
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        row = self.table['videos'][index]
        info = Path(row['path']).stat()
        stamp = (info.st_size, info.st_mtime_ns)
        if self.verified.get(index) != stamp:
            if sha(row['path']) != row['feature_sha256']:
                raise ValueError('源供体特征SHA改变')
            self.verified[index] = stamp
        with np.load(row['path']) as data:
            value = data['feats'].astype(np.float32)
        value = value[::self.table['dataset']['downsample_rate']]
        result = torch.from_numpy(np.ascontiguousarray(value.transpose()))
        if result.shape[-1] != row['length']:
            raise ValueError('特征长度改变')
        while self.cache and self.bytes + result.numel()*4 > self.maximum_bytes:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.numel()*4
        self.cache[index] = result
        self.bytes += result.numel()*4
        return result


def draw_pair(bank, target, rng, context_kind):
    assert_no_target_gt([target])
    table = bank.table
    n = min(target['feats'].shape[-1], max(v['length'] for v in table['videos']))
    choices = [p for p in table['patches'] if p['end']-p['start'] <= n]
    if not choices:
        return None
    actions = {}
    for p in choices:
        actions.setdefault((p['video'], p['primary']), []).append(p)
    patch = rng.choice(actions[rng.choice(list(actions))])
    width = patch['end'] - patch['start']
    bg = rng.choice([g for g in table['backgrounds'] if g['end']-g['start'] >= width])
    bg_start = rng.randint(bg['start'], bg['end']-width)
    context_id = rng.choice([i for i, v in enumerate(table['videos']) if v['length'] >= n])
    context_start = rng.randint(0, table['videos'][context_id]['length']-n)
    u = rng.randint(0, n-width)
    device = target['feats'].device
    # 所有组用相同私有随机流抽同一供体/背景/位置。只切换上下文来源。
    context = (bank.load(context_id)[:, context_start:context_start+n].to(device)
               if context_kind == 'source' else target['feats'][:, :n])
    positive, negative = context.clone(), context.clone()
    positive[:, u:u+width] = bank.load(patch['video'])[:, patch['start']:patch['end']].to(device)
    negative[:, u:u+width] = bank.load(bg['video'])[:, bg_start:bg_start+width].to(device)
    segments = torch.tensor(patch['segments'], dtype=torch.float32, device=device) + u
    labels = torch.tensor(patch['labels'], dtype=torch.long, device=device)
    metadata = deepcopy(target['metadata'])
    metadata.update(video_id='synthetic_source_gt', crop_start=0, crop_feat_len=n, video_length=n)
    views = [dict(feats=positive, segments=segments, labels=labels, metadata=metadata),
             dict(feats=negative, segments=segments.new_empty(0, 2), labels=labels.new_empty(0), metadata=metadata)]
    main = [x+u for x in patch['main']]
    event = dict(source_video=table['videos'][patch['video']]['video_id'], source_patch=[patch['start'],patch['end']],
                 source_background_video=table['videos'][bg['video']]['video_id'], source_background=[bg_start,bg_start+width],
                 source_context_video=table['videos'][context_id]['video_id'], source_context_start=context_start,
                 target_video=target['metadata']['video_id'], target_crop_start=target['metadata'].get('crop_start', 0),
                 context_length=n, inserted=[u,u+width], main=main, primary_class=patch['class_id'],
                 labels=labels.tolist(), segments=segments.tolist())
    return views, event


def supervised_predictions(model, views):
    inputs, masks = model.preprocessing(views, mode='train')
    inputs = model.dropout(inputs)
    features, masks, _ = model.backbone_source(inputs, masks)
    features, masks = model.neck_source(features, masks)
    logits, _ = model.cls_head(features, masks)
    offsets, actionness = model.reg_head(features, masks)
    points = model.point_generator(features)
    return features, masks, points, [x.permute(0, 2, 1) for x in logits], \
        [x.permute(0, 2, 1) for x in offsets], [x.permute(0, 2, 1) for x in actionness]


def mixed_loss(model, views, event, normalizer, epoch, pair_weight, audit_gradients=False):
    """输入掩码保留全部上下文；监督掩码只覆盖J，复用原label_points与losses。"""
    if model.use_trident_head or model.cfg['model']['adv_method'] != ['None']:
        raise ValueError('混合试验只允许原无对抗非Trident检测头')
    features, masks, points, logits, offsets, actionness = supervised_predictions(model, views)
    lo, hi = event['inserted']
    supervised = [mask.squeeze(1) & ((p[:, 0] >= lo) & (p[:, 0] < hi))[None]
                  for mask, p in zip(masks, points)]
    gt_cls, gt_off = model.label_points(points, [v['segments'] for v in views], [v['labels'] for v in views])
    valid = torch.cat(supervised, 1)
    num_pos = int(((torch.stack(gt_cls).sum(-1) > 0) & valid).sum())
    # 原loss接口期待source/target两半；第二半全ignore，不产生伪目标GT监督。
    duplicate = lambda xs: [torch.cat((x, x.detach()), 0) for x in xs]
    old = model.loss_normalizer, model.loss_normalizer_pseudo_labels
    try:
        model.loss_normalizer, model.loss_normalizer_pseudo_labels = normalizer, normalizer
        result = model.losses([torch.cat((m, torch.zeros_like(m)), 0) for m in supervised],
            duplicate(logits), duplicate(offsets), duplicate(actionness),
            gt_cls + [torch.zeros_like(x) for x in gt_cls], gt_off + [torch.zeros_like(x) for x in gt_off],
            None, None, [], [], [], mode='train', domain='src', curr_epoch=epoch)
        updated = float(model.loss_normalizer)
    finally:
        model.loss_normalizer, model.loss_normalizer_pseudo_labels = old
    points_all = torch.cat(points)
    primary = (gt_cls[0][:, event['primary_class']] > 0) & valid[0]
    primary &= (points_all[:, 0] > event['main'][0]) & (points_all[:, 0] < event['main'][1])
    combined = torch.cat(logits, 1)
    if primary.any():
        difference = (combined[0, primary, event['primary_class']] - combined[1, primary, event['primary_class']]).mean()
        pair = F.softplus(1. - difference)
    else:
        difference = combined.sum() * 0.
        pair = difference
    loss = result['downstream_loss'] + pair_weight * pair
    gradient = 0.
    if audit_gradients and loss.requires_grad:
        gradients = torch.autograd.grad(loss, features, retain_graph=True, allow_unused=True)
        gradient = float(torch.stack([g.detach().norm() for g in gradients if g is not None]).norm())
    details = dict(events=1., cls_raw=float(result['cls_loss'].detach()), reg_raw=float(result['reg_loss'].detach()),
                   pair_raw=float(pair.detach()), difference=float(difference.detach()),
                   positive_points=float(num_pos), pair_points=float(primary.sum()),
                   supervised_points=float(valid.sum()), fpn_gradient_norm=gradient,
                   weighted_loss=float((.1*loss).detach()))
    return loss, updated, details


def install(ctx, cfg):
    options = cfg['context_mix']
    spec = specification(options['variant'])
    if options['specification'] != spec or sha(options['table']) != options['table_sha256']:
        raise ValueError('上下文实验spec或源表SHA改变')
    table = read(options['table'])
    if table['source_manifest_sha256'] != cfg['baseline']['source_manifest_sha256']:
        raise ValueError('源供体表不匹配')
    bank = FeatureBank(table, spec['cache_bytes'])
    model = ctx.model
    if not cfg['model']['weight_sharing']:
        raise ValueError('必须保留共享检测主干和头')
    rng = random.Random(cfg['seed'] + 3000017)
    model.context_mix_state = dict(step=0, rng=rng.getstate(), normalizer=float(model.loss_normalizer),
                                   events=0, event_digest='0'*64)
    original = model.forward
    ledger = None
    ctx.context_mix_epoch = None
    keys = ('events','cls_raw','reg_raw','pair_raw','difference','positive_points','pair_points',
            'supervised_points','fpn_gradient_norm','weighted_loss')

    def forward(*args, **kwargs):
        nonlocal ledger
        output = original(*args, **kwargs)
        if kwargs.get('mode') != 'train':
            return output
        values = {k: 0. for k in keys}
        state = model.context_mix_state
        state['step'] += 1
        if state['step'] % spec['interval_steps'] == 0:
            rng.setstate(state['rng'])
            target = rng.choice(kwargs['video_list_tgt'])
            pair = draw_pair(bank, target, rng, spec['context'])
            state['rng'] = rng.getstate()
            if pair:
                views, event = pair
                epoch = kwargs.get('curr_epoch', 0)
                # 额外视图不能改变原训练/drop-path/loader的随机流。
                devices = [0] if torch.device(model.device).type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(cfg['seed'] + 4000037 + state['step'])
                    if devices:
                        torch.cuda.manual_seed(cfg['seed'] + 4000037 + state['step'])
                    loss, state['normalizer'], values = mixed_loss(model, views, event, state['normalizer'],
                        epoch, spec['pair_weight'], ctx.context_mix_epoch != epoch)
                if not math.isfinite(values['weighted_loss']):
                    raise ValueError('混合损失非有限')
                ctx.context_mix_epoch = epoch
                output[0]['downstream_loss'] = output[0]['downstream_loss'] + spec['mix_weight']*loss
                state['events'] += 1
                event.update(step=state['step'], epoch=epoch)
                canonical = json.dumps(event, sort_keys=True, separators=(',', ':'))
                state['event_digest'] = hashlib.sha256((state['event_digest']+canonical).encode()).hexdigest()
                if getattr(ctx, 'context_mix_ledger', None):
                    if ledger is None:
                        ledger = Path(ctx.context_mix_ledger).open('a', buffering=1)
                    ledger.write(json.dumps(dict(event, context=spec['context'], digest=state['event_digest']))+'\n')
        for name, value in values.items():
            output[0]['mix_'+name] = torch.as_tensor(value, device=model.device).detach()
        return output
    model.forward = forward
    ctx.context_mix_bank = bank
