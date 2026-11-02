"""仅源域拟合/校准；真实完整输出与等数量置信度过滤对比。"""
from contextlib import contextmanager
import random
import numpy as np
import torch
from tools.proposal_foreground_math import (seconds_to_grid, interval_labels,
    true_positive_flags, calibration_threshold)


@contextmanager
def preserve_rng(model):
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0))
    modes = [(m, m.training) for m in model.modules()]
    try:
        model.eval()
        yield
    finally:
        random.setstate(states[0]); np.random.set_state(states[1])
        torch.set_rng_state(states[2]); torch.cuda.set_rng_state(states[3], device=0)
        for m, mode in modes:
            m.training = mode


@torch.no_grad()
def collect(ctx, cfg, ids):
    from tools import baseline_engine as base
    from tools.baseline_protocol import read
    model = ctx.model_ema.module
    truth = read(cfg['proposal_foreground']['source_truth'])['records']
    if not set(ids) <= set(truth):
        raise ValueError('新增校准仅能源train，不能混入val/target')
    ds = base.dataset(cfg, cfg['dataset']['json_file'], cfg['src_train_split'], False, True)
    captured, rows = [], []
    handle = model.cls_head.register_forward_hook(lambda module, args, output: captured.append(args))
    try:
        with preserve_rng(model):
            for i, item in enumerate(ds.data_list):
                vid = item['id']
                if vid not in ids:
                    continue
                video = base.move_to_device([ds[i]], 'cuda:0')
                assert not video[0]['segments'].numel() and not video[0]['labels'].numel()
                captured.clear()
                result = model.proposal_fg_native_forward(video_list_src=video, video_list_tgt=video, mode='test')[0][0]
                features, masks = captured[0]
                boxes = seconds_to_grid(result['segments'].to('cuda:0'), video[0]['metadata'])
                raw = model.proposal_foreground(features[0][0], masks[0][0, 0].bool(), boxes,
                                               stride=model.fpn_strides[0]).softmax(-1).cpu()
                gt = torch.tensor(truth[vid]['segments'], dtype=torch.float32).reshape(-1, 2)
                gt_labels = torch.tensor(truth[vid]['labels'], dtype=torch.long)
                labels, best = interval_labels(boxes.cpu(), gt)
                true_positive = true_positive_flags(boxes.cpu(), result['labels'], result['scores'], gt, gt_labels)
                rows.append(dict(video=vid, background_probability=raw[:, 0], states=labels, best_iou=best,
                    true_positive=true_positive, scores=result['scores'].cpu(), segments=result['segments'].cpu(),
                    labels=result['labels'].cpu()))
                captured.clear()
        if {r['video'] for r in rows} != set(ids):
            raise ValueError('源校准视频覆盖不完整')
        return rows
    finally:
        handle.remove(); captured.clear()


def quality(rows, threshold):
    per_video = []
    for r in rows:
        removed = r['background_probability'] > threshold
        number = int(removed.sum())
        confidence = torch.zeros_like(removed)
        confidence[r['scores'].argsort()[:number]] = True
        bg, tp = r['states'] == 0, r['true_positive']
        per_video.append(dict(video=r['video'], candidates=len(bg), true_positive=int(tp.sum()),
            removed=number, background_removed=int((removed & bg).sum()),
            true_positive_removed=int((removed & tp).sum()),
            incomplete_removed=int((removed & (r['states'] == 1)).sum()),
            confidence_equal_count_background_removed=int((confidence & bg).sum()),
            confidence_equal_count_true_positive_removed=int((confidence & tp).sum())))
    return dict(videos=per_video, total={k:sum(r[k] for r in per_video) for k in per_video[0] if k != 'video'} if per_video else {})


def calibrate(ctx, cfg, entry, epoch):
    from tools.baseline_protocol import save, sha
    from tools.baseline_engine import checkpoint
    rows = collect(ctx, cfg, ctx.proposal_fg_roles['calibration'])
    calibrated = calibration_threshold(rows)
    calibrated.update(epoch=epoch, videos=[r['video'] for r in rows], source_train_only=True)
    for m in (ctx.model, ctx.model_ema.module):
        m.proposal_fg_state.update(threshold=calibrated['threshold'], calibration_epoch=epoch)
    path = entry/'calibration'/('e%03d.pt' % epoch)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint(path, dict(rows=rows, calibration=calibrated))
    save(path.with_suffix('.json'), dict(calibration=calibrated, quality=quality(rows, calibrated['threshold']),
                                        rows_sha256=sha(path), source_val_used=False, target_used=False))
    return calibrated
