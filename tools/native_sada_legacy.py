"""77ef5d7未优化条件损失；只作随机流/梯度/耗时等价参照。"""
import torch
from torch.nn import functional as F
def conditional_loss_legacy(model, source, smasks, target, tmasks, source_labels, target_logits, with_background=True):
    """原SADA类内BCE均值、域/层/背景权重；不以预测no-class冒充真实纯背景。"""
    cfg = model.cfg['model']
    total = source[0].sum() * 0
    counts = []
    accuracy = []
    for (l, (sf, sm, tf, tm, sl, tl)) in enumerate(zip(source, smasks, target, tmasks, source_labels, target_logits)):
        sx = sf.T[sm]
        tx = tf.T[tm]
        sl = sl[sm]
        prob = tl[tm].detach().sigmoid()
        (v, idx) = prob.max(-1)
        pseudo = F.one_hot(idx, num_classes=10).to(prob.dtype) * (v >= cfg['pl_threshold'])[:, None]
        src = []
        tgt = []
        level_counts = []
        for c in range(11 if with_background else 10):
            a = (sl == 0).all(-1) if c == 10 else sl[:, c] == 1
            b = (pseudo == 0).all(-1) if c == 10 else pseudo[:, c] == 1
            weight = cfg.get('weight_no_class_adv', 1.0) if c == 10 else 1.0
            level_counts.append([int(a.sum()), int(b.sum())])
            for (x, mask, terms, label) in ((sx, a, src, 0.0), (tx, b, tgt, 1.0)):
                if not mask.any():
                    continue
                value = torch.cat((x[mask], model.learnable_embeddings[c].expand(int(mask.sum()), -1)), 1)
                z = model.compute_level_dom_preds(value, True, l, 0, beta=cfg['beta']).squeeze(-1)
                terms.append(weight * F.binary_cross_entropy_with_logits(z, torch.full_like(z, label)))
                accuracy.append(float(((z.detach() >= 0) == bool(label)).float().mean()))
        s = sum(src, sx.new_zeros(())) / max(1, len(src))
        t = sum(tgt, tx.new_zeros(())) / max(1, len(tgt))
        if l in cfg['semantic_alignment_levels']:
            total = total + cfg.get('weight_level_' + str(l), 1.0) * ((cfg['weight_source_cond_adv'] * s + cfg['weight_target_cond_adv'] * t) / len(cfg['semantic_alignment_levels']))
        counts.append(level_counts)
    return (total, dict(class_counts=counts, domain_group_accuracy=sum(accuracy) / max(1, len(accuracy))))
