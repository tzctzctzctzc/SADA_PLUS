"""独立加到5f9a157原Prototype；不叠加负监督/阶段/质量方案。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import torch
from tools.proposal_foreground_math import (ProposalForeground, balanced_loss, interval_labels,
                                           seconds_to_grid, filter_result)

SCHEMA = 'prototype-proposal-foreground-v1'
BASE_COMMIT = '5f9a157e17e4a605ff4a6ed8dac4bd725b44d7b4'
SPEC = dict(schema=SCHEMA, base_commit=BASE_COMMIT, seed=123, batch=2,
    initialization='original_Prototype_cold; isolated_seed_new_head',
    representation='finest_FPN_512to64_projection; 4left16inner4right; inner-minus-context; MLP64to3',
    background='maximum_any_task_class_tIoU_less_than_.01_same_as_old_postNMS_Oracle',
    partial='any_class_max_tIoU_in_[.01,.5)', complete='any_class_max_tIoU_at_least_.5',
    source_loss='video_mean_balanced_3class_CE', source_aux_weight=.1,
    training_candidates='all_original_postNMS_crop_proposals_plus_visible_GT_and_two_fixed_shift_neighbors',
    source_split='stable_hash_20percent_source_train_calibration_rest_fit',
    calibration='EMA_source_train_calibration_only; video_macro_TP_at_.3_retention_.99',
    inference='original_NMS_and_voting_once_then_delete_high_background; retained_scores_boxes_labels_unchanged',
    original_source_detection='unchanged', original_Prototype_and_adversarial='unchanged',
    target_new_loss=0., target_GT=False, endpoint_refinement=False, VLM_queries=False,
    max_epochs=50, valid_every=5, patience=15, selection='original_source_cls_plus_reg_EMA',
    efficiency='reuse_original_FPN_and_head_outputs; project_once_per_video; chunk256_RoI_no_extra_detector_forward')


def split_source(videos):
    ordered = sorted(videos, key=lambda v: hashlib.sha256(('proposal_fg:' + v).encode()).hexdigest())
    count = max(1, round(.2 * len(ordered)))
    return dict(calibration=sorted(ordered[:count]), fit=sorted(ordered[count:]))


def attach(model):
    if hasattr(model, 'proposal_foreground'):
        return
    device = next(model.parameters()).device
    channels = model.cls_head.cls_head.conv.in_channels
    with torch.random.fork_rng(devices=[device.index or 0] if device.type == 'cuda' else []):
        torch.manual_seed(311103)
        model.add_module('proposal_foreground', ProposalForeground(channels).to(device))
    model.proposal_fg_state = dict(step=0, digest='0' * 64, threshold=1.01, calibration_epoch=None)


def enable_controls():
    from tools import baseline_engine as base
    if getattr(base, 'proposal_fg_controls', False):
        return
    native = base.controls
    def controls(model):
        result = native(model)
        if hasattr(model, 'proposal_fg_state'):
            result['proposal_fg_state'] = deepcopy(model.proposal_fg_state)
        return result
    base.controls = controls
    base.proposal_fg_controls = True


@torch.no_grad()
def crop_predictions(model, call, reg_call, video, index):
    """调用原解码、原soft-NMS/voting；训练crop用局部[0,L]，非全视频clip0。"""
    from libs.utils.nms import batched_nms
    features, masks = call[0]
    points = model.point_generator(features)
    cls, offsets = call[1][0], reg_call[1][0]
    raw = model.inference_single_video(points, [m[index, 0] for m in masks],
        [x[index].T for x in cls], [x[index].T for x in offsets], [None] * len(cls), [None] * len(cls))
    if model.test_nms_method == 'none':
        boxes, scores, labels = raw['segments'].cpu(), raw['scores'].cpu(), raw['labels'].cpu()
    else:
        boxes, scores, labels = batched_nms(raw['segments'].cpu(), raw['scores'].cpu(), raw['labels'].cpu(),
            model.test_iou_threshold, model.test_min_score, model.test_max_seg_num,
            use_soft_nms=model.test_nms_method == 'soft', multiclass=model.test_multiclass_nms,
            sigma=model.test_nms_sigma, voting_thresh=model.test_voting_thresh)
    length = video['feats'].shape[-1]
    boxes = boxes.clamp(0, length)
    keep = boxes[:, 1] > boxes[:, 0]
    return boxes[keep].to(features[0].device), scores[keep], labels[keep]


def source_candidates(boxes, truth, length):
    """合成课程保留实例来源；所有真实候选保留，不以GT筛掉难负样本。"""
    visible = truth[(truth[:, 0] >= 0) & (truth[:, 1] <= length)]
    groups = [boxes.detach()]
    if len(visible):
        duration = visible[:, 1] - visible[:, 0]
        groups.extend([visible, (visible + .2 * duration[:, None]).clamp(0, length),
                       (visible - .2 * duration[:, None]).clamp(0, length)])
    all_boxes = torch.cat(groups)
    return all_boxes[all_boxes[:, 1] > all_boxes[:, 0]].detach()


def install(ctx, cfg):
    """原trainer/optimizer字节不改，三优化器组数不变，新增头注册到model和EMA。"""
    from tools.baseline_protocol import read, sha
    spec = cfg['proposal_foreground']
    if spec['specification'] != SPEC:
        raise ValueError('科学配置不符')
    if sha(spec['source_truth']) != spec['source_truth_sha256']:
        raise ValueError('源真值文件改变')
    payload = read(spec['source_truth'])
    truth = payload['records']
    roles = split_source(list(truth))
    if roles != payload['roles'] or set(truth) & set(payload['source_val_ids']):
        raise ValueError('源拟合/校准/验证成员不符')
    ctx.proposal_fg_roles = roles
    for model in (ctx.model, ctx.model_ema.module):
        attach(model)
    ctx.model_ema.module.proposal_foreground.load_state_dict(ctx.model.proposal_foreground.state_dict())
    for opt in (ctx.optimizer_gen_source, ctx.optimizer_gen_source_alignment):
        for name, p in ctx.model.proposal_foreground.named_parameters():
            opt.param_groups[1 if name.endswith('bias') else 0]['params'].append(p)
    ctx.proposal_fg_enabled = True
    ctx.proposal_fg_ledger = None
    ctx.proposal_fg_probe_gradients = False
    ctx.proposal_fg_probe_observations = None
    for model in (ctx.model, ctx.model_ema.module):
        bind_forward(model, ctx, truth, set(roles['fit']))


def bind_forward(model, ctx, truth, fit_ids):
    if model.use_trident_head:
        raise ValueError('本版仅核验原Prototype非trident结构，不能静默更换边界解码')
    state = dict(active=False, cls=[], reg=[])
    def hook(key):
        def record(module, args, output):
            if state['active']:
                state[key].append((args, output))
        return record
    model.cls_head.register_forward_hook(hook('cls'))
    model.reg_head.register_forward_hook(hook('reg'))
    native = model.forward
    model.proposal_fg_native_forward = native

    def forward(*args, **kwargs):
        mode = kwargs.get('mode')
        if not ctx.proposal_fg_enabled or mode not in ('train', 'test'):
            return native(*args, **kwargs)
        state['cls'].clear(); state['reg'].clear(); state['active'] = True
        try:
            output = native(*args, **kwargs)
            if len(state['cls']) != 3 or len(state['reg']) != 3:
                raise ValueError('原共享分类/回归调用顺序改变')
            if mode == 'test':
                domains = (kwargs['video_list_src'], kwargs['video_list_tgt'])
                for domain, videos in enumerate(domains):
                    features, masks = state['cls'][domain][0]
                    for i, video in enumerate(videos):
                        item = output[domain][i]
                        boxes = seconds_to_grid(item['segments'].to(features[0].device), video['metadata'])
                        p = model.proposal_foreground(features[0][i], masks[0][i, 0].bool(), boxes,
                                                     stride=model.fpn_strides[0]).softmax(-1)[:, 0]
                        output[domain][i] = filter_result(item, p, model.proposal_fg_state['threshold'])
                return output
            if any(v['segments'].numel() or v['labels'].numel() for v in kwargs['video_list_tgt']):
                raise ValueError('正式目标路径出现GT')
            features, masks = state['cls'][0][0]
            losses, rows = [], []
            for i, video in enumerate(kwargs['video_list_src']):
                vid = video['metadata']['video_id']
                if vid not in truth:
                    raise ValueError('非源训练成员进入新增监督')
                if vid not in fit_ids:
                    continue
                boxes, _, _ = crop_predictions(model, state['cls'][0], state['reg'][0], video, i)
                gt = features[0].new_tensor(truth[vid]['segments']).reshape(-1, 2)
                gt = gt - float(video['metadata'].get('crop_start', 0))
                length = video['feats'].shape[-1]
                gt = gt[(gt[:, 1] > 0) & (gt[:, 0] < length)]
                candidates = source_candidates(boxes, gt, length)
                labels, _ = interval_labels(candidates, gt)
                logits = model.proposal_foreground(features[0][i], masks[0][i, 0].bool(), candidates,
                                                   stride=model.fpn_strides[0])
                losses.append(balanced_loss(logits, labels))
                if ctx.proposal_fg_probe_observations is not None:
                    ctx.proposal_fg_probe_observations.append(dict(video=vid, crop_start=video['metadata'].get('crop_start', 0),
                        boxes=candidates.cpu(), truth=gt.cpu(), states=labels.cpu(), raw_proposals=len(boxes)))
                rows.append(dict(video=vid, crop_start=video['metadata'].get('crop_start', 0),
                    raw_proposals=len(boxes), candidates=len(candidates),
                    counts=[int((labels == k).sum()) for k in range(3)]))
            extra = torch.stack(losses).mean() if losses else features[0].sum() * 0
            if ctx.proposal_fg_probe_gradients and losses:
                ctx.proposal_fg_aux_gradients = {}
                for name, module in (('SGP', model.backbone_source), ('cls', model.cls_head),
                                     ('reg', model.reg_head), ('foreground', model.proposal_foreground)):
                    gradients = torch.autograd.grad(extra, tuple(module.parameters()), retain_graph=True, allow_unused=True)
                    ctx.proposal_fg_aux_gradients[name] = sum(float(g.detach().square().sum()) for g in gradients if g is not None) ** .5
            output[0]['downstream_loss'] = output[0]['downstream_loss'] + SPEC['source_aux_weight'] * extra
            output[0]['proposal_fg_loss'] = extra.detach()
            control = model.proposal_fg_state
            control['step'] += 1
            event = dict(step=control['step'], epoch=kwargs['curr_epoch'], source=rows,
                         calibration_excluded=[v['metadata']['video_id'] for v in kwargs['video_list_src'] if v['metadata']['video_id'] not in fit_ids],
                         target_new_loss=0., source_loss=float(extra.detach()))
            control['digest'] = hashlib.sha256((control['digest'] + json.dumps(event, sort_keys=True)).encode()).hexdigest()
            if ctx.proposal_fg_ledger:
                with open(ctx.proposal_fg_ledger, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(dict(event, digest=control['digest'])) + '\n')
            return output
        finally:
            state['active'] = False; state['cls'].clear(); state['reg'].clear()
    model.forward = forward
