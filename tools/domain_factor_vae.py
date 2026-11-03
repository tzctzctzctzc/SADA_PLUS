"""逐时刻动作/域解耦；DIVA/DSN启发的检测适配，不是论文完整复现。

检测仅使用动作后验均值；域码只进入重建和域识别。保持原FPN时间网格。
AE/VAE共享均值编码器和全部任务目标；VAE额外学习方差、采样并施加条件KL。
target只有无标签特征、角色域ID和停止梯度的当前检测概率，不接收target GT。
"""
import torch
from torch import nn
from torch.nn import functional as F


SPEC = dict(schema='sada-prototype-domain-factor-v1', action_dim=128, domain_dim=32,
            reconstruction_weight=.1, task_weight=.1, domain_weight=.1,
            confusion_weight=.1, kl_weight=.001, kl_ramp_epochs=10,
            target_foreground_threshold=.3, logvar_min=-8., logvar_max=4.,
            logvar_init=-4., initialization_seed=20261104, sampling_seed=311104,
            reconstruction_target='detached_per_position_LayerNorm_FPN',
            original_adversary='disabled', detection='action_mean_only_no_raw_skip',
            posterior_noise='separate_checkpointed_generator',
            target_GT=False, conditional_alignment_background=False)


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


def class_distribution(labels):
    """允许重叠源标注；无任务类的时刻是BG，不等同于proposal IoU背景定义。"""
    foreground = labels.clamp(0., 1.)
    background = (foreground.sum(-1, keepdim=True) == 0).to(labels.dtype)
    result = torch.cat((foreground, background), -1)
    return result / result.sum(-1, keepdim=True).clamp_min(1.)


def balanced_mean(values, foreground):
    terms = [values[mask].mean() for mask in (foreground, ~foreground) if mask.any()]
    return torch.stack(terms).mean() if terms else values.sum() * 0.


def gaussian_kl(mean, logvar, prior_mean):
    return .5 * (logvar.exp() + (mean-prior_mean).square() - 1. - logvar).mean(-1)


class DomainFactor(nn.Module):
    def __init__(self, channels, classes, variant, device='cpu'):
        super().__init__()
        assert variant in ('ae', 'vae')
        self.variant, self.classes = variant, classes
        a, d = SPEC['action_dim'], SPEC['domain_dim']
        # 固定归一化，不能通过学习把重建目标的尺度压到零来降低重建loss。
        self.input_norm = nn.LayerNorm(channels, elementwise_affine=False)
        self.action_mean = nn.Linear(channels, a)
        self.domain_mean = nn.Linear(channels, d)
        self.task_projection = nn.Sequential(nn.Linear(a, channels), nn.LayerNorm(channels))
        self.reconstruction = nn.Sequential(nn.Linear(a+d, 256), nn.GELU(), nn.Linear(256, channels))
        self.task_classifier = nn.Linear(a, classes+1)
        self.style_classifier = nn.Linear(d, 1)
        self.action_domain_classifier = nn.Sequential(nn.Linear(a, 64), nn.GELU(), nn.Linear(64, classes))
        # 在共同模块之后构造方差/先验，不改变AE/VAE共同参数初始化。
        if variant == 'vae':
            self.action_logvar = nn.Linear(channels, a)
            self.domain_logvar = nn.Linear(channels, d)
            for layer in (self.action_logvar, self.domain_logvar):
                nn.init.zeros_(layer.weight)
                nn.init.constant_(layer.bias, SPEC['logvar_init'])
            self.action_prior = nn.Embedding(classes+1, a)
            self.domain_prior = nn.Embedding(2, d)
            nn.init.normal_(self.action_prior.weight, std=.02)
            nn.init.normal_(self.domain_prior.weight, std=.02)
        generator = torch.Generator(device=device).manual_seed(SPEC['sampling_seed'])
        self.register_buffer('sampling_state', generator.get_state())

    def transform(self, features, masks, training):
        outputs, cache = [], []
        generator = None
        if training and self.variant == 'vae':
            generator = torch.Generator(device=features[0].device)
            generator.set_state(self.sampling_state.cpu())
        for feature, mask in zip(features, masks):
            # 无随机池化/抽样，不改变时间分辨率；padding永远不能参与辅助loss。
            x = self.input_norm(feature.transpose(1, 2))
            ma, md = self.action_mean(x), self.domain_mean(x)
            output = self.task_projection(ma).transpose(1, 2) * mask.to(feature.dtype)
            outputs.append(output)
            if training:
                la = ld = None
                za, zd = ma, md
                if self.variant == 'vae':
                    la = self.action_logvar(x).clamp(SPEC['logvar_min'], SPEC['logvar_max'])
                    ld = self.domain_logvar(x).clamp(SPEC['logvar_min'], SPEC['logvar_max'])
                    za = ma + (.5*la).exp()*torch.randn(ma.shape, generator=generator, device=ma.device)
                    zd = md + (.5*ld).exp()*torch.randn(md.shape, generator=generator, device=md.device)
                cache.append(dict(x=x.detach(), ma=ma, md=md, la=la, ld=ld, za=za, zd=zd,
                                  mask=mask.squeeze(1).bool()))
        if generator is not None:
            self.sampling_state.copy_(generator.get_state().to(self.sampling_state.device))
        return outputs, cache

    def auxiliary(self, source, target, source_labels, target_logits, epoch, warmup):
        terms = {k: [] for k in ('reconstruction', 'task', 'style_domain', 'conditional_domain', 'kl')}
        counts = dict(source_fg=0, target_fg=0, aligned_classes=0)
        ma_all, md_all, va_all = [], [], []
        strength = min(1., max(0., (epoch-warmup+1)/5.))
        for src, tgt, labels, logits in zip(source, target, source_labels, target_logits):
            svalid, tvalid = src['mask'], tgt['mask']
            sm, tm = src['ma'][svalid], tgt['ma'][tvalid]
            sg = labels[svalid]
            sf = sg.sum(-1) > 0
            sq = class_distribution(sg)
            tp = logits.detach().sigmoid()[tvalid]
            confidence, tl = tp.max(-1)
            tf = confidence >= SPEC['target_foreground_threshold']
            # 此分布仅用于KL期望，是启发式权重，不是校准后概率或目标GT。
            tq = torch.cat((tp, 1.-confidence[:, None]), -1)
            tq = tq / tq.sum(-1, keepdim=True).clamp_min(1e-8)
            counts['source_fg'] += int(sf.sum())
            counts['target_fg'] += int(tf.sum())
            if svalid.any():
                ce = -(sq * self.task_classifier(sm).log_softmax(-1)).sum(-1)
                terms['task'].append(balanced_mean(ce, sf))
            for rows, valid, domain in ((src, svalid, 0), (tgt, tvalid, 1)):
                if not valid.any():
                    continue
                rec = self.reconstruction(torch.cat((rows['za'][valid], rows['zd'][valid]), -1))
                terms['reconstruction'].append(F.mse_loss(rec, rows['x'][valid]))
                prediction = self.style_classifier(rows['md'][valid]).squeeze(-1)
                terms['style_domain'].append(F.binary_cross_entropy_with_logits(prediction,
                                                       torch.full_like(prediction, float(domain))))
                ma_all.append(rows['ma'][valid].detach().square().mean())
                md_all.append(rows['md'][valid].detach().square().mean())
            # 每类、每域等权；只对源GT和目标高置信共同出现的前景类对齐。
            sd = self.action_domain_classifier(Reverse.apply(sm, strength))
            td = self.action_domain_classifier(Reverse.apply(tm, strength))
            for c in range(self.classes):
                si, ti = sg[:, c] > 0, tf & (tl == c)
                if si.any() and ti.any():
                    value = .5*(F.softplus(sd[si, c]).mean()+F.softplus(-td[ti, c]).mean())
                    terms['conditional_domain'].append(value)
                    counts['aligned_classes'] += 1
            if self.variant == 'vae':
                for rows, valid, q, domain, foreground in ((src, svalid, sq, 0, sf),
                                                          (tgt, tvalid, tq, 1, tf)):
                    if not valid.any():
                        continue
                    mean, lv = rows['ma'][valid], rows['la'][valid]
                    # 对离散类条件KL显式求期望，不能用平均prior替代混合的期望。
                    ka = torch.stack([gaussian_kl(mean, lv, prior)
                                      for prior in self.action_prior.weight], -1)
                    ka = (ka*q).sum(-1)
                    kd = gaussian_kl(rows['md'][valid], rows['ld'][valid], self.domain_prior.weight[domain])
                    terms['kl'].append(balanced_mean(ka+kd, foreground))
                    va_all.append(lv.detach().exp().mean())
        zero = sum((p.sum()*0. for p in self.parameters()))
        means = {k: torch.stack(v).mean() if v else zero for k, v in terms.items()}
        ramp = min(1., epoch / SPEC['kl_ramp_epochs'])
        loss = (SPEC['reconstruction_weight']*means['reconstruction'] + SPEC['task_weight']*means['task']
                + SPEC['domain_weight']*means['style_domain']
                + SPEC['confusion_weight']*means['conditional_domain']
                + SPEC['kl_weight']*ramp*means['kl'])
        return dict(loss=loss, **means, **counts, grl_strength=strength,
                    action_energy=torch.stack(ma_all).mean() if ma_all else zero,
                    style_energy=torch.stack(md_all).mean() if md_all else zero,
                    posterior_variance=torch.stack(va_all).mean() if va_all else zero)


def build_factor(cfg, channels, classes, device):
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(SPEC['initialization_seed'])
        return DomainFactor(channels, classes, cfg['variant'], device)
