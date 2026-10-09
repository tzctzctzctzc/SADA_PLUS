"""已知动作边界条件下的 RQ1 分类；SADA 式前景类别条件 GRL 对齐。

不导入 TAL 训练器：没有背景/定位/FPN，也没有 RQ3 教师辅助损失。
训练函数仅接收目标特征，不接收目标类别；目标验证集只在选模结束后评估。
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Recipe:
    seed: int = 6
    epochs: int = 50
    batch_size: int = 128  # 动作片段，不是 TAL 的完整视频 batch2。
    hidden: int = 512
    classes: int = 10
    lr: float = 1e-4
    disc_lr: float = 7e-3
    min_lr: float = 5e-5
    weight_decay: float = .05
    warmup: int = 5
    adv_start: int = 5  # 第6轮起对齐；与零起始 SADA epoch 对应。
    beta: float = .5
    threshold: float = .6
    source_domain_weight: float = 1.
    target_domain_weight: float = .2
    adv_weight: float = 1.
    task_weight: float = 2.
    disc_layers: int = 2
    disc_dropout: float = .5
    disc_bias: float = .25
    ema_decay: float = .999
    val_every: int = 5
    patience: int = 15  # epoch 距离，不是验证次数；0 表示跑满。

    def validate(self):
        if min(self.epochs, self.batch_size, self.hidden, self.classes, self.val_every) < 1:
            raise ValueError('invalid dimensions or training budget')
        if not 0 <= self.threshold <= 1 or not 0 <= self.ema_decay < 1:
            raise ValueError('invalid confidence threshold or EMA')
        if self.patience and self.patience % self.val_every:
            raise ValueError('patience must be a multiple of validation interval')


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, beta):
        ctx.beta = beta
        return features.view_as(features)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.beta * gradient, None


class SemanticModel(nn.Module):
    def __init__(self, dimensions, recipe):
        super().__init__()
        self.recipe = recipe
        h = recipe.hidden
        self.adapter = nn.Sequential(nn.Linear(dimensions, h), nn.LayerNorm(h), nn.ReLU())
        self.classifier = nn.Linear(h, recipe.classes)
        self.class_embedding = nn.Embedding(recipe.classes, h)
        layers = [nn.Linear(2 * h, h), nn.LeakyReLU(), nn.Dropout(recipe.disc_dropout)]
        for _ in range(recipe.disc_layers):
            layers += [nn.Linear(h, h), nn.LeakyReLU(), nn.Dropout(recipe.disc_dropout)]
        layers += [nn.Linear(h, 1)]
        self.discriminator = nn.Sequential(*layers)
        nn.init.constant_(self.discriminator[-1].bias, recipe.disc_bias)

    def forward(self, features):
        embedding = self.adapter(features)
        return embedding, self.classifier(embedding)

    def domain_logits(self, embedding, labels):
        joint = torch.cat((embedding, self.class_embedding(labels)), dim=1)
        return self.discriminator(Reverse.apply(joint, self.recipe.beta)).flatten()


def class_mean(losses, labels, mask, classes):
    # 与 SADA 独立 source/target 的“非空类别均值”归约一致。
    counts = losses.new_zeros(classes).scatter_add(0, labels, mask.to(losses.dtype))
    totals = losses.new_zeros(classes).scatter_add(0, labels, losses * mask)
    active = counts > 0
    return ((totals / counts.clamp_min(1)) * active).sum() / active.sum().clamp_min(1), counts


def conditional_loss(model, source_embedding, source_labels, target_embedding, target_logits):
    r = model.recipe
    confidence, pseudo = target_logits.detach().softmax(1).max(1)
    accepted = confidence >= r.threshold
    # 未达阈值的已知动作片段不冒充背景；本任务没有背景类别。
    source_scores = model.domain_logits(source_embedding, source_labels)
    target_scores = model.domain_logits(target_embedding, pseudo)
    sl, sc = class_mean(F.softplus(source_scores), source_labels,
                        torch.ones_like(source_labels, dtype=torch.bool), r.classes)
    tl, tc = class_mean(F.softplus(-target_scores), pseudo, accepted, r.classes)
    loss = r.source_domain_weight * sl + r.target_domain_weight * tl
    return loss, {'source_domain_loss': sl.detach(), 'target_domain_loss': tl.detach(),
                  'target_accepted': accepted.sum().detach(), 'source_class_counts': sc.detach(),
                  'target_class_counts': tc.detach()}


class CyclingSampler:
    """独立 CPU RNG；SO/UDA 有完全相同的源域样本顺序和更新次数。"""
    def __init__(self, size, batch, seed):
        if size < 1:
            raise ValueError('empty training partition')
        self.size, self.batch = size, batch
        self.generator = torch.Generator().manual_seed(seed)
        self.remaining = torch.empty(0, dtype=torch.long)

    def next(self):
        chunks, needed = [], self.batch
        while needed:
            if not len(self.remaining):
                self.remaining = torch.randperm(self.size, generator=self.generator)
            count = min(needed, len(self.remaining))
            chunks.append(self.remaining[:count])
            self.remaining = self.remaining[count:]
            needed -= count
        return torch.cat(chunks)


def make_optimizer(model):
    r = model.recipe
    return torch.optim.AdamW([
        {'params': list(model.adapter.parameters()) + list(model.classifier.parameters()),
         'lr': r.lr, 'base_lr': r.lr},
        {'params': list(model.discriminator.parameters()) + list(model.class_embedding.parameters()),
         'lr': r.disc_lr, 'base_lr': r.disc_lr},
    ], weight_decay=r.weight_decay)


def schedule(optimizer, epoch, step, steps, recipe):
    progress = epoch + step / steps
    if recipe.warmup and progress < recipe.warmup:
        scale = max((progress + 1 / steps) / recipe.warmup, 1e-6)
    else:
        q = (progress - recipe.warmup) / max(recipe.epochs - recipe.warmup, 1)
        scale = recipe.min_lr / recipe.lr + (1 - recipe.min_lr / recipe.lr) * (1 + math.cos(math.pi * min(q, 1))) / 2
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * scale


def update_ema(ema, model):
    # 只平滑用于分类的映射/分类器；判别器不用于验证或选模。
    with torch.no_grad():
        for name in ('adapter', 'classifier'):
            for target, source in zip(getattr(ema, name).parameters(), getattr(model, name).parameters()):
                target.lerp_(source, 1 - model.recipe.ema_decay)


def optimization_step(model, optimizer, source_x, source_y, target_x, active, uda):
    r = model.recipe
    optimizer.zero_grad(set_to_none=True)
    hs, source_logits = model(source_x)
    classification = F.cross_entropy(source_logits, source_y)
    task_loss = classification * (r.task_weight if active else 1.)
    stats = {'source_ce': classification.detach(), 'alignment_loss': task_loss.detach() * 0,
             'target_accepted': task_loss.detach() * 0, 'target_embedding_gradient': task_loss.detach() * 0}
    loss = task_loss
    ht = None
    if active and uda:
        ht, logits = model(target_x)
        ht.retain_grad()
        alignment, details = conditional_loss(model, hs, source_y, ht, logits)
        stats.update(details)
        stats['alignment_loss'] = alignment.detach()
        loss = task_loss + r.adv_weight * alignment
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite training loss')
    loss.backward()
    if ht is not None:
        stats['target_embedding_gradient'] = ht.grad.detach().norm()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    stats['gradient_norm'] = norm.detach()
    return stats


@torch.no_grad()
def predict(model, features, chunk=1024):
    model.eval()
    probabilities = torch.cat([model(x)[1].softmax(1) for x in features.split(chunk)])
    if not torch.isfinite(probabilities).all():
        raise FloatingPointError('nonfinite prediction')
    return probabilities


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def fit(source_x, source_y, target_x, source_val_x, source_val_y, recipe, regime,
        output=None):
    """没有目标训练/验证类别参数；选模只看 source_val CE。"""
    recipe.validate()
    if regime not in ('so', 'uda'):
        raise ValueError(regime)
    torch.manual_seed(recipe.seed)
    model = SemanticModel(source_x.shape[1], recipe).to(source_x.device)
    ema = deepcopy(model).eval()
    # 与 RQ3 基础训练的两阶段优化一致：适应阶段使用独立 Adam 状态；
    # 配套 SO 在同一轮切换，避免优化器状态差异冒充 UDA 效果。
    warm_optimizer, alignment_optimizer = make_optimizer(model), make_optimizer(model)
    source = CyclingSampler(len(source_x), recipe.batch_size, recipe.seed + 101)
    target = CyclingSampler(len(target_x), recipe.batch_size, recipe.seed + 202)
    steps = max(math.ceil(len(source_x) / recipe.batch_size), math.ceil(len(target_x) / recipe.batch_size))
    best_loss, best_epoch, best_state = math.inf, 0, None
    history, updates = [], 0
    synchronize(source_x.device)
    start = time.perf_counter()
    for epoch in range(recipe.epochs):
        model.train()
        optimizer = warm_optimizer if epoch < recipe.adv_start else alignment_optimizer
        totals = {}
        for step in range(steps):
            schedule(optimizer, epoch, step, steps, recipe)
            si, ti = source.next().to(source_x.device), target.next().to(source_x.device)
            stats = optimization_step(model, optimizer, source_x[si], source_y[si],
                                      target_x[ti], epoch >= recipe.adv_start, regime == 'uda')
            update_ema(ema, model)
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value
            updates += 1
        row = {'epoch': epoch + 1, 'updates': updates,
               **{key: (value / steps).cpu().tolist() for key, value in totals.items()}}
        row['target_acceptance_fraction'] = row['target_accepted'] / recipe.batch_size
        row['target_gt_loss'] = 0.  # 由不接收目标标签的训练接口保证。
        validated = (epoch + 1) % recipe.val_every == 0 or epoch + 1 == recipe.epochs
        if validated:
            probabilities = predict(ema, source_val_x)
            val_loss = F.nll_loss(probabilities.clamp_min(1e-30).log(), source_val_y).item()
            row['source_val_loss'] = val_loss
            if val_loss < best_loss:
                best_loss, best_epoch = val_loss, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
                if output is not None:
                    torch.save({'model': best_state, 'epoch': best_epoch,
                                'source_val_loss': best_loss, 'recipe': asdict(recipe)}, output / 'best.pt')
        row['best_epoch'] = best_epoch
        history.append(row)
        if output is not None:
            import json
            with (output / 'history.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
        print('RQ1_EPOCH', regime, epoch + 1, 'best', best_epoch,
              'target_coverage', round(row['target_acceptance_fraction'], 4), flush=True)
        if validated and recipe.patience and epoch + 1 - best_epoch >= recipe.patience:
            reason = 'source_val_patience'
            break
    else:
        reason = 'max_epochs'
    synchronize(source_x.device)
    elapsed = time.perf_counter() - start
    termination = {'epoch': epoch + 1, 'best_epoch': best_epoch, 'reason': reason,
                   'updates': updates, 'steps_per_epoch': steps, 'seconds': elapsed}
    if output is not None:
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict(),
                    'warm_optimizer': warm_optimizer.state_dict(),
                    'alignment_optimizer': alignment_optimizer.state_dict(), 'termination': termination,
                    'recipe': asdict(recipe)}, output / 'latest.pt')
    ema.load_state_dict(best_state)
    return ema.eval(), history, termination


def metrics(probabilities, labels):
    from sklearn.metrics import accuracy_score, f1_score
    p = np.asarray(probabilities)
    return {'macro_f1': float(f1_score(labels, p.argmax(1), average='macro',
                                      labels=list(range(p.shape[1])), zero_division=0)),
            'accuracy': float(accuracy_score(labels, p.argmax(1))),
            'cross_entropy': float(-np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-30)).mean())}
