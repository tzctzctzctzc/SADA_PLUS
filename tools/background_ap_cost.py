"""源视频删一框的真实AP代价；只改变背景头训练权重，不改变背景定义。"""
import numpy as np
import torch
from torch.nn import functional as F
from tools.foreground_oracle_ap import ClassAP
from tools.hard_background_modules import balanced_bg, quality_loss


def deletion_costs(truth, predictions, cutoff=.3):
    """每视频每类，删后重新排序/匹配；包含全部低分预测作为替补。

    返回AP fraction，不是pp；各框边际效应不可相加，也不是全域AP导数。
    调用方只允许传源train。此函数不选择阈值。
    """
    delta = np.zeros(len(predictions), dtype=np.float64)
    for label in range(10):
        positions = np.flatnonzero(predictions['label'].to_numpy() == label)
        frame = predictions.iloc[positions].reset_index(drop=True)
        engine = ClassAP(truth[truth['label'] == label], frame)
        keep = np.ones(len(frame), dtype=bool)
        original = engine.evaluate(keep)['map']
        for i in np.flatnonzero(frame['score'].to_numpy() >= cutoff):
            keep[i] = False
            delta[positions[i]] = engine.evaluate(keep)['map'] - original
            keep[i] = True
    return delta


def cost_balanced_bg(logits, bg, delta):
    """背景按删对收益、前景按误删损失组内归一；零代价组退回普通BCE。"""
    losses = []
    for kind in (0, 1):
        mask = bg == kind
        if not mask.any():
            continue
        cost = (delta[mask] if kind else -delta[mask]).clamp_min(0).detach()
        ce = F.binary_cross_entropy_with_logits(logits[mask], bg[mask], reduction='none')
        losses.append((ce * cost).sum() / cost.sum() if cost.sum() > 0 else ce.mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def step(head, item, mode):
    out = head(item['feature'], item['valid'], item['boxes'], item['labels'], item['scores'])
    bg = balanced_bg(out[:, 0], item['bg'])
    if mode == 'ap_cost':
        bg = .5 * bg + .5 * cost_balanced_bg(out[:, 0], item['bg'], item['delta'])
    elif mode != 'control':
        raise ValueError(mode)
    return bg + .25 * quality_loss(out[:, 1], item['quality'])
