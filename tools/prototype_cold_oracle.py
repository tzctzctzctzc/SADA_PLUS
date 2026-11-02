"""专用目标训练GT容量实验；冷初始化，原检测器/Prototype/优化循环不改。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

MODES = ('joint', 'classification', 'localization')
SCHEMA = 'prototype-cold-target-train-supervised-oracle-v1'
SPEC = dict(schema=SCHEMA, modes=list(MODES), initialization='fresh_original_seed123_not_mature',
    seed=123, batch=2, onset_epoch=1, target_weight=1., max_epochs=50, validation_every=5,
    patience=15, selection='original_source_cls_plus_reg_best_EMA',
    assignment='original_center_sampling_regression_range_shortest_instance',
    classification='original_sigmoid_focal_all_valid_points_including_benchmark_background',
    localization='original_DIou_only_positive_points_assigned_to_fully_visible_instances',
    normalizer='separate_target_positive_EMA_for_classification_and_visible_regression',
    regression_weight='same_current_source_regression_weight',
    input='original_unlabeled_target_loader_uniform_crop_no_GT_sampling',
    inference='original_detector_no_GT_no_oracle_head_no_score_changes',
    original_Prototype_and_adversarial='unchanged',
    scope='target_train_supervised_capacity_reference_not_UDA_or_mathematical_upper_bound')


def crop_truth(record, metadata, length, device):
    for key in ('fps', 'feat_stride', 'feat_num_frames'):
        assert record[key] == metadata[key], key
    full = torch.as_tensor(record['segments'], device=device, dtype=torch.float32).reshape(-1, 2)
    full = full - float(metadata.get('crop_start', 0))
    labels = torch.as_tensor(record['labels'], device=device, dtype=torch.long)
    keep = (full[:, 1] > 0) & (full[:, 0] < length) & (labels >= 0) & (labels < 10)
    ids = torch.nonzero(keep).flatten()
    full, labels = full[keep], labels[keep]
    clipped = full.clamp(0, float(length))
    visible = (full[:, 0] >= 0) & (full[:, 1] <= length)
    return clipped, labels, visible, ids


def assigned_instances(points, segments, center, radius, use_range):
    """与原label_points_single_video相同的实例选择，另返回索引以屏蔽截断端点。"""
    if not len(segments):
        return torch.full((len(points),), -1, dtype=torch.long, device=points.device)
    left = points[:, 0, None] - segments[None, :, 0]
    right = segments[None, :, 1] - points[:, 0, None]
    if center == 'radius':
        middle = segments.mean(-1)[None]
        lo = torch.maximum(middle - points[:, 3, None] * radius, segments[None, :, 0])
        hi = torch.minimum(middle + points[:, 3, None] * radius, segments[None, :, 1])
        inside = torch.minimum(points[:, 0, None] - lo, hi - points[:, 0, None]) > 0
    else:
        inside = torch.minimum(left, right) > 0
    if use_range:
        distance = torch.maximum(left, right)
        inside &= (distance >= points[:, 1, None]) & (distance <= points[:, 2, None])
    lengths = (segments[:, 1] - segments[:, 0])[None].expand(len(points), -1).clone()
    lengths.masked_fill_(~inside, float('inf'))
    value, index = lengths.min(1)
    return torch.where(torch.isfinite(value), index, index.new_full(index.shape, -1))


def detection_losses(logits, offsets, valid, classes, targets, reg_mask, normalizers,
                     smoothing=0., gamma=2.):
    from libs.models.SADA.losses_aux import sigmoid_focal_loss, ctr_diou_loss_1d
    smoothed = classes[valid] * (1 - smoothing) + smoothing / (classes.shape[-1] + 1)
    cls = sigmoid_focal_loss(logits[valid], smoothed, alpha=.25, gamma=gamma, reduction='sum') / normalizers[0]
    reg = ctr_diou_loss_1d(offsets[reg_mask], targets[reg_mask], reduction='sum') / normalizers[1] if reg_mask.any() else offsets.sum() * 0
    return cls, reg


def enable_controls():
    from tools import baseline_engine as base
    if getattr(base, 'cold_oracle_controls_enabled', False):
        return
    previous = base.controls
    def controls(model):
        result = previous(model)
        if hasattr(model, 'cold_oracle_state'):
            result['cold_oracle_state'] = deepcopy(model.cold_oracle_state)
        return result
    base.controls = controls
    base.cold_oracle_controls_enabled = True


def install(ctx, cfg, truth):
    model = ctx.model
    assert not model.use_trident_head
    mode = cfg['cold_oracle']['mode']
    assert mode in MODES
    ctx.cold_oracle_enabled = True
    ctx.cold_oracle_ledger = None
    ctx.cold_oracle_gradient_epoch = None
    ctx.cold_oracle_last = {}
    for item in (model, ctx.model_ema.module):
        item.cold_oracle_state = dict(step=0, digest='0' * 64, events=0,
            cls_normalizer=float(item.train_init_loss_norm), reg_normalizer=float(item.train_init_loss_norm))
    captured = dict(cls=[], reg=[], points=[])
    active = [False]
    def head_hook(kind):
        def hook(module, args, output):
            if active[0]:
                captured[kind].append((args, output))
        return hook
    model.cls_head.register_forward_hook(head_hook('cls'))
    model.reg_head.register_forward_hook(head_hook('reg'))
    model.point_generator.register_forward_hook(head_hook('points'))
    original = model.forward
    ctx.cold_oracle_original_forward = original
    def forward(*args, **kwargs):
        if kwargs.get('mode') != 'train' or not ctx.cold_oracle_enabled:
            return original(*args, **kwargs)
        assert kwargs.get('domain', 'src') == 'src'
        videos = kwargs['video_list_tgt']
        for video in videos:
            assert not video['segments'].numel() and not video['labels'].numel()
        for values in captured.values():
            values.clear()
        active[0] = True
        try:
            output = original(*args, **kwargs)
        finally:
            active[0] = False
        assert len(captured['cls']) == len(captured['reg']) == 3
        assert len(captured['points']) == 1
        # 原共享头调用顺序为source、target、pl，只复用第二路，不重复任何网络forward。
        cls_args, cls_output = captured['cls'][1]
        reg_args, reg_output = captured['reg'][1]
        assert all(a is b for a, b in zip(cls_args[0], reg_args[0]))
        points = captured['points'][0][1]
        concat_points = torch.cat(points)
        logits = torch.cat([x.permute(0, 2, 1) for x in cls_output[0]], 1)
        offsets = torch.cat([x.permute(0, 2, 1) for x in reg_output[0]], 1)
        valid = torch.cat([m.squeeze(1) for m in cls_args[1]], 1)
        all_classes, all_offsets, all_masks, observations = [], [], [], []
        for index, video in enumerate(videos):
            key = video['metadata']['video_id']
            segments, labels, visible, global_ids = crop_truth(truth[key], video['metadata'], video['feats'].shape[-1], logits.device)
            with torch.no_grad():
                classes, target_offsets = model.label_points(points, [segments], [labels])
                classes, target_offsets = classes[0], target_offsets[0]
                assigned = assigned_instances(concat_points, segments, model.train_center_sample,
                    model.train_center_sample_radius, model.train_reg_distance)
                positive = (classes.sum(-1) > 0) & valid[index]
                assert torch.equal(positive, (assigned >= 0) & valid[index])
                regression = positive & (visible[assigned.clamp_min(0)] if len(visible) else torch.zeros_like(positive))
                # 核验实际回归标签确实来自所选实例，不把crop边缘当真实端点。
                if regression.any():
                    expected = torch.stack([concat_points[:, 0] - segments[assigned.clamp_min(0), 0],
                        segments[assigned.clamp_min(0), 1] - concat_points[:, 0]], -1) / concat_points[:, 3, None]
                    assert torch.allclose(expected[regression], target_offsets[regression], atol=1e-5, rtol=1e-5)
            pos_ids = torch.nonzero(positive).flatten()
            observations.append(dict(video=key, crop_start=video['metadata'].get('crop_start', 0),
                length=video['feats'].shape[-1], valid=int(valid[index].sum()), positive_points=pos_ids.cpu().tolist(),
                assigned_original_instances=global_ids[assigned[pos_ids]].cpu().tolist(),
                regression_points=torch.nonzero(regression).flatten().cpu().tolist(),
                intersecting_instances=len(segments), truncated_instances=int((~visible).sum())))
            all_classes.append(classes); all_offsets.append(target_offsets); all_masks.append(regression)
        classes, targets, reg_mask = torch.stack(all_classes), torch.stack(all_offsets), torch.stack(all_masks)
        positives = int(((classes.sum(-1) > 0) & valid).sum())
        state = model.cold_oracle_state
        momentum = model.loss_normalizer_momentum
        state['cls_normalizer'] = momentum * state['cls_normalizer'] + (1 - momentum) * max(positives, 1)
        state['reg_normalizer'] = momentum * state['reg_normalizer'] + (1 - momentum) * max(int(reg_mask.sum()), 1)
        cls, reg = detection_losses(logits, offsets, valid, classes, targets, reg_mask,
            [state['cls_normalizer'], state['reg_normalizer']], model.train_label_smoothing,
            cfg['model'].get('focal_loss_gamma', 2.))
        losses = output[0]
        reg_weight = float(model.train_loss_reg_weight) if model.train_loss_reg_weight > 0 else float(losses['cls_loss'].detach()) / max(float(losses['reg_loss'].detach()), .01)
        aux = (cls if mode != 'localization' else cls * 0) + reg_weight * (reg if mode != 'classification' else reg * 0)
        assert torch.isfinite(aux)
        losses['downstream_loss'] = losses['downstream_loss'] + aux
        stats = dict(cls=float(cls.detach()) if mode != 'localization' else 0.,
            reg=float(reg.detach()) if mode != 'classification' else 0., aux=float(aux.detach()),
            positives=positives, regression_points=int(reg_mask.sum()), valid_points=int(valid.sum()),
            background_points=int(valid.sum()) - positives, reg_weight=reg_weight,
            cls_normalizer=state['cls_normalizer'], reg_normalizer=state['reg_normalizer'])
        epoch = kwargs['curr_epoch']
        if ctx.cold_oracle_gradient_epoch != epoch and positives and reg_mask.any():
            for name, parameters in [('target_sgp_gradient', model.backbone_source.parameters()),
                    ('target_classification_gradient', model.cls_head.parameters()),
                    ('target_regression_gradient', model.reg_head.parameters())]:
                gradients = torch.autograd.grad(aux, tuple(parameters), retain_graph=True, allow_unused=True)
                stats[name] = sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** .5
            assert stats['target_sgp_gradient'] > 0
            assert (stats['target_classification_gradient'] > 0) == (mode != 'localization')
            assert (stats['target_regression_gradient'] > 0) == (mode != 'classification')
            ctx.cold_oracle_gradient_epoch = epoch
        state['step'] += 1
        state['events'] += int(positives > 0)
        event = dict(step=state['step'], epoch=epoch, mode=mode, observations=observations, stats=stats)
        state['digest'] = hashlib.sha256((state['digest'] + json.dumps(event, sort_keys=True)).encode()).hexdigest()
        if ctx.cold_oracle_ledger:
            with Path(ctx.cold_oracle_ledger).open('a') as stream:
                stream.write(json.dumps(dict(event, digest=state['digest'])) + '\n')
        ctx.cold_oracle_last = stats
        for key, value in stats.items():
            losses['cold_oracle_' + key] = aux.new_tensor(value).detach()
        for values in captured.values():
            values.clear()
        return output
    model.forward = forward
