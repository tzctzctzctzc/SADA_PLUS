"""候选阶段拒绝与端点误差一致性；仅源GT监督新头，目标GT不可见。"""
from copy import deepcopy
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F


def endpoint_error(head, x, reference, coordinates, label, side, stride, length, mask=None):
    from tools.phase_boundary import sample, interval_tokens
    duration = (reference[1] - reference[0]).clamp_min(1e-6)
    offsets = torch.linspace(-.125, .125, 7, device=x.device, dtype=x.dtype)
    positions = coordinates[:, None] + offsets[None] * duration
    available = (positions >= 0) & (positions < length)
    z = sample(x, positions.flatten(), stride).reshape(len(coordinates), 7, -1)
    z = z * available[..., None]
    completeness = available.float()
    if mask is not None:
        completeness = completeness * sample(mask.float()[None], positions.flatten(), stride).reshape_as(positions)
    h = head.visual(z) + head.boundary_position[None]
    h = head.boundary_compress(head.boundary_temporal(h.transpose(1, 2)).flatten(1))
    context = head.visual(interval_tokens(x, reference, stride)).mean(0).expand(len(coordinates), -1)
    text = head.text_proj(head.text[int(label), 0 if side == 0 else 2]).expand_as(h)
    geometry = torch.stack([(coordinates - reference[side]) / duration,
        (coordinates - reference[1 - side]) / duration, torch.log1p(duration / stride).expand_as(coordinates),
        completeness.mean(-1), ((coordinates >= 0) & (coordinates < length)).to(x.dtype),
        coordinates.new_full(coordinates.shape, side)], -1)
    return F.softplus(head.error_model(torch.cat([h, context, text, geometry], -1)).squeeze(-1)) + 1e-4


def candidate_coordinates(reference, side):
    # 原端点必须排第一；完全相同误差预测时严格选择原框。
    grid = torch.linspace(-.25, .25, 17, device=reference.device, dtype=reference.dtype)
    offsets = torch.cat([grid.new_zeros(1), grid[grid != 0]])
    return reference[side] + offsets * (reference[1] - reference[0])


def refine(head, feature, mask, reference, label, stride, length):
    selected, risks, rawrisks, moved = [], [], [], []
    for side in range(2):
        coordinates = candidate_coordinates(reference, side)
        risk = head.endpoint_error(feature, reference, coordinates, label, side, stride, length, mask)
        valid = (coordinates >= 0) & (coordinates < length)
        choice = risk.detach().masked_fill(~valid, float('inf')).argmin()
        selected.append(coordinates[choice]);risks.append(risk[choice]);rawrisks.append(risk[0]);moved.append(int(choice != 0))
    return torch.stack(selected), torch.stack(risks), torch.stack(rawrisks), moved


def _truth(video):
    truth, labels = video['segments'], video['labels']
    length = video['feats'].shape[-1]
    valid = (labels < 10) & ((truth[:, 1] - truth[:, 0]) > .05) & (truth[:, 1] > 0) & (truth[:, 0] < length)
    return truth[valid], labels[valid]


def _assignment(box, truth, labels, length):
    if not len(truth):
        return 10, None
    visible = truth.clamp(0, length - 1e-4)
    inter = (torch.minimum(box[1], visible[:, 1]) - torch.maximum(box[0], visible[:, 0])).clamp_min(0)
    iou = inter / ((box[1] - box[0]) + visible[:, 1] - visible[:, 0] - inter).clamp_min(1e-6)
    best = int(iou.argmax())
    return (int(labels[best]), best) if iou[best] >= .1 else (10, None)


def _level(box, levels):
    return int(torch.clamp(torch.log2((box[1] - box[0]).clamp_min(1) / 8).round(), 0, levels - 1))


def source_candidates_loss(model, head, f, masks, reg, videos, cls=None):
    from tools.phase_boundary import interval_tokens, select_proposals
    classification, risk_losses = [], []
    stats = dict(candidate_groups=0, candidate_background=0, candidate_wrong_class=0, candidate_error_endpoints=0)
    for vi, v in enumerate(videos):
        truth, labels = _truth(v);length = v['feats'].shape[-1];groups = []
        for gi, (gt, label) in enumerate(zip(truth, labels)):
            duration = gt[1] - gt[0]
            for offsets in [(0., 0.), (-.15, .15), (.15, -.15)]:
                box = (gt + gt.new_tensor(offsets) * duration).clamp(0, length - 1e-4)
                if box[1] - box[0] > .05:
                    groups.append((box, int(label), gi, int(label), _level(box, len(f))))
        # 全部实际GT空隙提供本benchmark的背景例，绝不把别的已标注实例当背景。
        cursor = 0.
        for gt in sorted(truth.detach().cpu().tolist(), key=lambda x: x[0]) + [[float(length), float(length)]]:
            start = max(0., min(float(length), gt[0]));end = max(0., min(float(length), gt[1]))
            if start - cursor > .05:
                box = f[0].new_tensor([cursor, min(start, length - 1e-4)])
                if box[1] - box[0] > .05:
                    groups.append((box, 10, None, 10, _level(box, len(f))))
            cursor = max(cursor, end)
        if cls is not None:
            for box, label, level, index, confidence in select_proposals(f, masks, cls, reg, model, vi, length):
                assigned, gi = _assignment(box, truth, labels, length)
                groups.append((box.detach(), assigned, gi, int(label), level))
                stats['candidate_wrong_class'] += int(assigned < 10 and int(label) != assigned)
        if not groups:
            continue
        stats['candidate_groups'] += len(groups)
        labels_all = torch.tensor([g[1] for g in groups], device=f[0].device)
        stats['candidate_background'] += int((labels_all == 10).sum())
        ce = []
        for start in range(0, len(groups), 32):
            batch = groups[start:start + 32]
            tokens = torch.stack([interval_tokens(f[l][vi], box, model.fpn_strides[l]) for box, _, _, _, l in batch])
            _, scores = head.sequence(tokens)
            ce.extend(F.nll_loss(scores, labels_all[start:start + len(batch)], reduction='none').unbind())
        ce = torch.stack(ce)
        # 两类分别平均，避免全部GT正例淹没真实未匹配候选。
        categories = [ce[labels_all < 10], ce[labels_all == 10]]
        classification.append(torch.stack([x.mean() for x in categories if len(x)]).mean())
        for box, assigned, gi, proposed, level in groups:
            if gi is None or assigned != proposed:
                continue
            reference = box.detach();gt = truth[gi];duration = (reference[1] - reference[0]).clamp_min(1e-6)
            for side in range(2):
                if not .05 < gt[side] < length - .05:
                    continue
                coordinates = candidate_coordinates(reference, side)
                coordinates = torch.cat([coordinates, gt[side:side + 1].detach()])
                coordinates = coordinates[(coordinates >= 0) & (coordinates < length)]
                target = ((coordinates - gt[side]) / duration).abs().detach()
                prediction = head.endpoint_error(f[level][vi], reference, coordinates, assigned, side,
                                                model.fpn_strides[level], length, masks[level][vi, 0])
                p, q = torch.log1p(prediction), torch.log1p(target)
                fit = F.smooth_l1_loss(p, q)
                different = target[:, None] < target[None, :] - 1e-6
                if different.any():
                    weight = (target[None, :] - target[:, None]).clamp(0, 1)
                    rank = (F.softplus(p[:, None] - p[None, :]) * weight)[different].mean()
                    fit = fit + .1 * rank
                risk_losses.append(fit);stats['candidate_error_endpoints'] += 1
    losses = []
    if classification:
        losses.append(torch.stack(classification).mean())
    if risk_losses:
        losses.append(torch.stack(risk_losses).mean())
    return (torch.stack(losses).sum() if losses else f[0].sum() * 0), stats


def candidate_target_loss(model, teacher, head, weak, strong, views, step, observations=None):
    from tools.phase_boundary import select_proposals, interval_tokens, phase_posterior, map_coordinates
    sf, sm, sc, sr = strong;tf, tm, tc, tr = weak;losses = []
    stats = dict(target_proposals=0, target_class_used=0, target_left_used=0, target_right_used=0, target_phase_used=0,
                 target_teacher_scale=0., target_student_scale=0., target_no_move_endpoints=0, target_background_rejected=0)
    for vi, view in enumerate(views):
        for box, c, level, index, confidence in select_proposals(tf, tm, tc, tr, teacher, vi, view['original_length']):
            stats['target_proposals'] += 1;stride = model.fpn_strides[level]
            if box[0] < view['lo'] or box[1] > view['lo'] + view['width'] - 1:
                continue
            mapped = map_coordinates(box, view);point = map_coordinates(box.new_tensor(index * stride), view)
            k = int((point / stride).round())
            if k < 0 or k >= sf[level].shape[-1] or not sm[level][vi, 0, k]:
                continue
            with torch.no_grad():
                tl, tclass = teacher.phase_boundary.sequence(interval_tokens(tf[level][vi], box, stride)[None])
                phase_class = int(tclass.argmax(-1))
                stats['target_background_rejected'] += int(phase_class == 10)
                if phase_class != int(c):
                    continue
                post = phase_posterior(tl[:, c])
                refined, t_error, _, moved = refine(teacher.phase_boundary, tf[level][vi], tm[level][vi, 0], box, c,
                                                   stride, view['original_length'])
                mapped_ref = map_coordinates(refined, view)
            sl, sclass = head.sequence(interval_tokens(sf[level][vi], mapped, stride)[None])
            losses.append(F.kl_div(sl[:, c].log_softmax(-1), post, reduction='batchmean') / 24)
            losses.append(F.nll_loss(sclass, c[None]));stats['target_phase_used'] += 1
            label = torch.zeros_like(sc[level][vi, :, k]);label[c] = 1
            losses.append(F.binary_cross_entropy_with_logits(sc[level][vi, :, k], label) * confidence)
            stats['target_class_used'] += 1
            pred = torch.stack([k * stride - sr[level][vi, 0, k] * stride, k * stride + sr[level][vi, 1, k] * stride])
            with torch.no_grad():
                s_error = torch.stack([head.endpoint_error(sf[level][vi], mapped, pred[side:side + 1].detach(), c, side,
                    stride, view['length'], sm[level][vi, 0])[0] for side in range(2)])
                mask = (t_error < s_error) & (mapped_ref >= 0) & (mapped_ref < view['length'])
            stats['target_teacher_scale'] += float(t_error.mean());stats['target_student_scale'] += float(s_error.mean())
            stats['target_no_move_endpoints'] += 2 - sum(moved)
            if mask.any():
                residual = (pred - mapped_ref.detach()) / (mapped[1] - mapped[0]).clamp_min(1e-6)
                losses.append(F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none')[mask].mean())
                stats['target_left_used'] += int(mask[0]);stats['target_right_used'] += int(mask[1])
            if observations is not None:
                observations.append(dict(video_index=vi, box=box.detach().cpu().tolist(), label=int(c), point=index * stride,
                    refined=refined.cpu().tolist(), student=map_coordinates(pred.detach(), view, True).cpu().tolist(),
                    mask=mask.cpu().tolist(), class_log_probability=tclass[0].cpu().tolist(), moved=moved, view=dict(view)))
    return (torch.stack(losses).mean() if losses else sf[0].sum() * 0), stats


def paired_source_error_loss(model, head, weak, strong, views, videos):
    """源GT训练与target完全同分布的共同anchor归一化误差比较。"""
    from tools.phase_boundary import select_proposals, map_coordinates
    wf, wm, wc, wr = weak;sf, sm, sc, sr = strong;losses = [];count = 0
    for vi, view in enumerate(views):
        truth, labels = _truth(videos[vi])
        for box, c, level, index, confidence in select_proposals(wf, wm, wc, wr, model, vi, view['original_length']):
            same = (labels == c) & (truth[:, 0] <= index * model.fpn_strides[level]) & (truth[:, 1] >= index * model.fpn_strides[level])
            ids = same.nonzero().flatten()
            if not len(ids) or box[0] < view['lo'] or box[1] > view['lo'] + view['width'] - 1:
                continue
            gi = ids[(truth[ids, 1] - truth[ids, 0]).argmin()];gt = truth[gi]
            stride = model.fpn_strides[level];reference = map_coordinates(box, view)
            k = int((map_coordinates(box.new_tensor(index * stride), view) / stride).round())
            if k < 0 or k >= sf[level].shape[-1] or not sm[level][vi, 0, k]:
                continue
            refined, weak_risk, _, _ = refine(head, wf[level][vi], wm[level][vi, 0], box, c, stride, view['original_length'])
            prediction = torch.stack([k * stride - sr[level][vi, 0, k] * stride, k * stride + sr[level][vi, 1, k] * stride])
            mapped_gt = map_coordinates(gt, view)
            for side in range(2):
                if not .05 < gt[side] < view['original_length'] - .05 or not 0 <= mapped_gt[side] < view['length']:
                    continue
                strong_risk = head.endpoint_error(sf[level][vi], reference, prediction[side:side + 1].detach(), c, side,
                                                  stride, view['length'], sm[level][vi, 0])[0]
                a = ((refined[side] - gt[side]).abs() / (box[1] - box[0]).clamp_min(1e-6)).detach()
                b = ((prediction[side] - mapped_gt[side]).abs() / (reference[1] - reference[0]).clamp_min(1e-6)).detach()
                p, q = torch.log1p(weak_risk[side]), torch.log1p(strong_risk)
                fit = .5 * (F.smooth_l1_loss(p, torch.log1p(a)) + F.smooth_l1_loss(q, torch.log1p(b)))
                if (a - b).abs() > 1e-6:
                    fit = fit + .1 * F.softplus((p - q) * torch.sign(b - a))
                losses.append(fit);count += 1
    return (torch.stack(losses).mean() if losses else wf[0].sum() * 0), dict(source_paired_error_endpoints=count)


def _cpu_quality(observations, videos):
    """误差与收益只在CPU使用源GT计算，不把GT传入伪标签预测函数。"""
    ce, selected, raw, gain, left, right = [], [], [], [], 0, 0
    for r in observations:
        v = videos[r['video_index']]
        truth = v['segments'].detach().cpu().numpy();labels = v['labels'].detach().cpu().numpy()
        valid = labels < 10;truth, labels = truth[valid], labels[valid]
        box = np.asarray(r['box']);best = None
        if len(truth):
            inter = np.maximum(0, np.minimum(box[1], truth[:, 1]) - np.maximum(box[0], truth[:, 0]))
            iou = inter / np.maximum(1e-6, box[1] - box[0] + truth[:, 1] - truth[:, 0] - inter)
            best = int(iou.argmax()) if iou.max() >= .1 else None
        label = int(labels[best]) if best is not None else 10
        ce.append(-r['class_log_probability'][label])
        same = [i for i, (g, c) in enumerate(zip(truth, labels)) if c == r['label'] and g[0] <= r['point'] <= g[1]]
        if not same:
            continue
        gt = truth[min(same, key=lambda i: truth[i, 1] - truth[i, 0])]
        duration = max(1e-6, box[1] - box[0])
        for side in range(2):
            if not .05 < gt[side] < v['feats'].shape[-1] - .05:
                continue
            err = abs(r['refined'][side] - gt[side]) / duration
            selected.append(err);raw.append(abs(box[side] - gt[side]) / duration)
            mapped_gt = (gt[side] - r['view']['lo'] + .5) * r['view']['ratio'] - .5
            if r['mask'][side] and 0 <= mapped_gt < r['view']['length']:
                gain.append((abs(r['student'][side] - gt[side]) - abs(r['refined'][side] - gt[side])) / duration)
                left += int(side == 0);right += int(side == 1)
    return dict(class_nll=float(np.mean(ce)) if ce else None, selected_error=float(np.mean(selected)) if selected else None,
                raw_error=float(np.mean(raw)) if raw else None, accepted_gain=float(np.mean(gain)) if gain else None,
                accepted_left=left, accepted_right=right, matched_endpoints=len(selected))


def _source_predictions(model, head, weak, strong, views):
    """源选模比较真实raw候选，包括被拒绝者，避免全拒绝获得虚假零误差。"""
    from tools.phase_boundary import select_proposals, interval_tokens, map_coordinates
    tf, tm, tc, tr = weak;sf, sm, sc, sr = strong;rows = []
    for vi, view in enumerate(views):
        for box, c, level, index, confidence in select_proposals(tf, tm, tc, tr, model, vi, view['original_length']):
            stride = model.fpn_strides[level]
            _, scores = head.sequence(interval_tokens(tf[level][vi], box, stride)[None])
            refined, teacher_error, _, moved = refine(head, tf[level][vi], tm[level][vi, 0], box, c, stride, view['original_length'])
            mapped = map_coordinates(box, view);point = map_coordinates(box.new_tensor(index * stride), view)
            k = int((point / stride).round());mask = torch.zeros(2, dtype=torch.bool, device=box.device)
            student = box.clone()
            if box[0] >= view['lo'] and box[1] <= view['lo'] + view['width'] - 1 and 0 <= k < sf[level].shape[-1] and sm[level][vi, 0, k]:
                pred = torch.stack([k * stride - sr[level][vi, 0, k] * stride, k * stride + sr[level][vi, 1, k] * stride])
                errors = torch.stack([head.endpoint_error(sf[level][vi], mapped, pred[s:s + 1], c, s, stride,
                                      view['length'], sm[level][vi, 0])[0] for s in range(2)])
                mapped_ref = map_coordinates(refined, view)
                mask = (teacher_error < errors) & (mapped_ref >= 0) & (mapped_ref < view['length']) & (scores.argmax(-1)[0] == c)
                student = map_coordinates(pred, view, True)
            rows.append(dict(video_index=vi, box=box.cpu().tolist(), label=int(c), point=index * stride, refined=refined.cpu().tolist(),
                        student=student.cpu().tolist(), mask=mask.cpu().tolist(), class_log_probability=scores[0].cpu().tolist(), view=view))
    return rows


def pretrain_candidate_head(ctx, cfg, output):
    from tools import baseline_engine as base
    from tools.baseline_protocol import save, sha
    from tools.phase_boundary import SPEC, direct, source_loss, strong_view, map_coordinates
    output = Path(output);model = ctx.model;model.eval();head = model.phase_boundary
    detector_before = base.tensor_digest({k:v for k,v in model.state_dict().items() if not k.startswith('phase_boundary.')})
    torch.cuda.reset_peak_memory_stats();cache_root = output.parent / 'source_candidate_cache';caches = {}
    source_loader = base.loader(ctx.train_loader_src.dataset, True, torch.Generator().manual_seed(123), 1, cfg['loader']['num_workers'])
    for role, loader in [('train', source_loader), ('val', ctx.val_loader_src)]:
        caches[role] = []
        for index, videos in enumerate(loader):
            videos = base.move_to_device(videos, model.device)
            with torch.no_grad():
                weak = direct(model, videos, training=role == 'train');strong_videos = [];views = []
                for vi, v in enumerate(videos):
                    x, view = strong_view(v, index + 1, vi);view['original_length'] = v['feats'].shape[-1]
                    x['segments'] = map_coordinates(v['segments'], view);x['labels'] = v['labels']
                    strong_videos.append(x);views.append(view)
                strong = direct(model, strong_videos, training=role == 'train')
            path = cache_root / role / ('%04d.pt' % index)
            base.checkpoint(path, _cpu_tree(dict(weak=weak, strong=strong, videos=videos, strong_videos=strong_videos, views=views)))
            caches[role].append(dict(path=str(path), sha256=sha(path), videos=[v['metadata']['video_id'] for v in videos]))
            print('SOURCE_CANDIDATE_CACHE', role, index, flush=True)
    save(output.parent / 'source_candidate_cache_index.json', dict(roles=caches, source_only=True, target_gt_used=False))
    optimizer = torch.optim.AdamW(head.parameters(), lr=SPEC['source_head_lr'], weight_decay=.05)
    history = [];best = float('inf');best_state = None;best_metrics = None
    for epoch in range(SPEC['source_head_epochs'] + 1):
        train_losses = [];counts = dict(groups=0, background=0, wrong_class=0)
        if epoch:
            head.train()
            for item in caches['train']:
                data = _device_tree(torch.load(item['path'], map_location='cpu'), model.device)
                optimizer.zero_grad();losses = []
                for key, videos in [('weak', 'videos'), ('strong', 'strong_videos')]:
                    f, m, c, r = data[key]
                    loss, stats = source_loss(model, head, f, m, r, data[videos], c)
                    losses.append(loss)
                    counts['groups'] += stats['candidate_groups'];counts['background'] += stats['candidate_background'];counts['wrong_class'] += stats['candidate_wrong_class']
                paired, _ = paired_source_error_loss(model, head, data['weak'], data['strong'], data['views'], data['videos'])
                loss = torch.stack(losses).mean() + paired;assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(), 1.);optimizer.step();train_losses.append(float(loss.detach()))
        head.eval();val_loss = [];qualities = []
        with torch.no_grad():
            for item in caches['val']:
                data = _device_tree(torch.load(item['path'], map_location='cpu'), model.device)
                assert len(data['videos']) == 1
                f, m, c, r = data['weak']
                val_loss.append(float(source_loss(model, head, f, m, r, data['videos'], c)[0]))
                predicted = _source_predictions(model, head, data['weak'], data['strong'], data['views'])
                # 下面只做CPU源GT误差计算；预测函数未接收GT。
                qualities.append(_cpu_quality(predicted, data['videos']))
        def mean(key):
            values = [q[key] for q in qualities if q[key] is not None]
            return sum(values) / len(values) if values else None
        quality = {k: mean(k) for k in ['class_nll', 'selected_error', 'raw_error', 'accepted_gain']}
        quality.update(accepted_left=sum(q['accepted_left'] for q in qualities), accepted_right=sum(q['accepted_right'] for q in qualities),
                       matched_endpoints=sum(q['matched_endpoints'] for q in qualities),
                       validation_videos=len(qualities),videos_with_raw_candidates=sum(q['class_nll'] is not None for q in qualities),
                       videos_with_matched_endpoints=sum(q['selected_error'] is not None for q in qualities))
        assert quality['class_nll'] is not None and quality['selected_error'] is not None
        selection = quality['class_nll'] + quality['selected_error']
        row = dict(epoch=epoch, train_loss=sum(train_losses) / len(train_losses) if train_losses else None,
                   validation_loss=sum(val_loss) / len(val_loss), source_candidate_score=selection, quality=quality, counts=counts)
        history.append(row)
        if selection < best:
            best = selection;best_epoch = epoch;best_state = deepcopy(head.state_dict());best_metrics = deepcopy(quality)
        base.checkpoint(output.with_name('source_head_latest.pt'), dict(spec=SPEC, state_dict=head.state_dict(), epoch=epoch,
             history=history, optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(0),
             rng_python=random.getstate(), rng_numpy=np.random.get_state(), cache_index_sha256=sha(output.parent / 'source_candidate_cache_index.json')))
        save(output.with_name('source_candidate_history.json'), history)
        print('SOURCE_CANDIDATE_HEAD', __import__('json').dumps(row), flush=True)
    readiness = dict(refinement_gain=best_metrics['raw_error'] - best_metrics['selected_error'], accepted_gain=best_metrics['accepted_gain'],
                     left=best_metrics['accepted_left'], right=best_metrics['accepted_right'])
    readiness['passed'] = (readiness['refinement_gain'] >= -1e-6 and readiness['accepted_gain'] is not None and readiness['accepted_gain'] > 0
                           and readiness['left'] > 0 and readiness['right'] > 0)
    assert detector_before == base.tensor_digest({k:v for k,v in model.state_dict().items() if not k.startswith('phase_boundary.')})
    base.checkpoint(output, dict(spec=SPEC, state_dict=best_state, best_epoch=best_epoch, history=history, source_only=True,
         source_videos=[r['videos'] for r in caches['train']], validation_videos=[r['videos'] for r in caches['val']],
         selection='equal_video_source_raw_candidate_class_NLL_plus_selected_endpoint_error', source_readiness=readiness))
    save(output.with_suffix('.json'), dict(spec=SPEC, best_epoch=best_epoch, history=history, head_sha256=sha(output), source_readiness=readiness,
         frozen_detector_sha256=detector_before,source_pretrain_peak_mib=torch.cuda.max_memory_reserved() / 2**20,
         target_gt_used=False, cache_index_sha256=sha(output.parent / 'source_candidate_cache_index.json')))
    head.load_state_dict(best_state, strict=True);ctx.model_ema.module.phase_boundary.load_state_dict(best_state, strict=True)
    torch.cuda.empty_cache()


def _cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(v) for v in value)
    return value


def _device_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _device_tree(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_device_tree(v, device) for v in value)
    return value
