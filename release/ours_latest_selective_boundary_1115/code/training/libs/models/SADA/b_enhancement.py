"""六条独立B描述器；训练窗口与候选查询读取同一套全序列证据图。"""
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .class_path_alignment import ClassConditionalPathAlignment, conditional_discrepancy
from .uda_refinement import rotating_starts
from .uda_routes import sample

VARIANTS = ('tcn', 'transformer', 'path8', 'wavelet', 'sinc', 'scattering')


def filter_sequence(x, kernels, dilation=1):
    """不降采样，只在物理视频边缘延拓；候选边缘不参与padding。"""
    # x: N,D,T；同一滤波器组逐通道应用。
    n, d, t = x.shape
    pad = dilation * (kernels.shape[-1] - 1) // 2
    value = F.pad(x.reshape(n*d, 1, t), (pad, pad), mode='replicate')
    return F.conv1d(value, kernels[:, None].to(x), dilation=dilation).reshape(n, d, len(kernels), t)


def starlet(x, levels=3):
    """B样条平稳小波：逐级平滑差分，无下采样，细节＋残余精确重构。"""
    kernel = x.new_tensor([[1., 4., 6., 4., 1.]]) / 16
    low, details = x, []
    for j in range(levels):
        smooth = filter_sequence(low, kernel, 2**j)[:, :, 0]
        details.append(low-smooth)
        low = smooth
    return torch.stack([low, *details], dim=2)


class SincBands(nn.Module):
    """四个连续覆盖[0,Nyquist]的频带；最小带宽阻止退化为空频带。"""
    def __init__(self, bands=4, taps=33):
        super().__init__()
        self.width_logits = nn.Parameter(torch.zeros(bands))
        self.register_buffer('positions', torch.arange(-(taps//2), taps//2+1).float())
        self.register_buffer('window', torch.hann_window(taps, periodic=False))
        self.minimum_width = .02

    def edges(self):
        widths = self.minimum_width + (.5-len(self.width_logits)*self.minimum_width)*self.width_logits.softmax(0)
        return torch.cat([widths.new_zeros(1), widths.cumsum(0)])

    def filters(self):
        edges = self.edges(); n = self.positions[None]
        low, high = edges[:-1, None], edges[1:, None]
        filters = (2*high*torch.sinc(2*high*n)-2*low*torch.sinc(2*low*n))*self.window
        return filters / filters.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def forward(self, x):
        return filter_sequence(x, self.filters())


class LocalScattering(nn.Module):
    """一阶与有序二阶Morlet模系数，保留时间轴而非全局平均。"""
    def __init__(self):
        super().__init__()
        kernels = []
        # 相同支持便于批量计算；频率单位为各FPN序列的cycles/sample。
        t = torch.arange(-32, 33).float()
        for frequency in (.25, .125, .0625):
            envelope = torch.exp(-.5*(t/(.6/frequency))**2)
            real = envelope*torch.cos(2*math.pi*frequency*t)
            imag = envelope*torch.sin(2*math.pi*frequency*t)
            real -= real.mean(); imag -= imag.mean()
            scale = (real.square().sum()+imag.square().sum()).sqrt()
            kernels.extend([real/scale, imag/scale])
        self.register_buffer('kernels', torch.stack(kernels))

    def modulus(self, x, j):
        response = filter_sequence(x, self.kernels[2*j:2*j+2])
        return (response.square().sum(2)+1e-12).sqrt()-1e-6

    def forward(self, x):
        first = [self.modulus(x, j) for j in range(3)]
        second = [self.modulus(first[i], j) for i in range(3) for j in range(i+1, 3)]
        low = starlet(x)[:, :, 0]
        return torch.stack([low, *first, *second], dim=2)


class ResidualTemporal(nn.Module):
    """残差幅度逐位置限制到输入RMS的0.1，防止仅靠编码器抹平差异。"""
    def __init__(self, kind, dim=16):
        super().__init__(); self.kind = kind
        self.norm = nn.LayerNorm(dim)
        if kind == 'tcn':
            self.encoder = nn.Sequential(nn.Conv1d(dim, 32, 3, padding=1), nn.GELU(),
                                         nn.Conv1d(32, dim, 3, padding=1))
        else:
            self.encoder = nn.TransformerEncoderLayer(dim, 4, 32, dropout=0., batch_first=True,
                                                       activation='gelu', norm_first=True)
            t = torch.arange(32).float()[:, None]
            frequency = torch.exp(torch.arange(0, dim, 2).float()*(-math.log(10000.)/dim))
            pos = torch.zeros(32, dim); pos[:, 0::2] = torch.sin(t*frequency); pos[:, 1::2] = torch.cos(t*frequency)
            self.register_buffer('position', pos)

    def forward(self, x):
        y = self.norm(x.transpose(1, 2))
        if self.kind == 'tcn':
            # 在完整有效序列上运行，不为每个候选建立卷积边缘。
            y = self.encoder(y.transpose(1, 2)).transpose(1, 2)
        else:
            n, t, d = y.shape; pad = (-t) % 32
            blocks = F.pad(y, (0, 0, 0, pad)).reshape(-1, 32, d)
            valid = (torch.arange(t+pad, device=x.device)[None] < t).expand(n, -1).reshape(-1, 32)
            y = self.encoder(blocks + .1*self.position.to(x), src_key_padding_mask=~valid)
            y = y.reshape(n, t+pad, d)[:, :t]
        rms = x.detach().square().mean(1, keepdim=True).sqrt()
        return x + .1*rms*torch.tanh(y.transpose(1, 2))


def pooled_descriptors(z, q, bins):
    n, d, t = z.shape; classes = q.shape[1]
    weighted = z[:, None]*q[:, :, None]
    pooled = weighted.sum(-1)/q.sum(-1).clamp_min(1e-6)[:, :, None]
    path = F.adaptive_avg_pool1d(weighted.reshape(n*classes, d, t), bins)
    mass = F.adaptive_avg_pool1d(q.reshape(n*classes, 1, t), bins)
    path = (path/mass.clamp_min(1e-6)).reshape(n, classes, -1)
    return pooled, path, weighted


class EnhancedAlignment(ClassConditionalPathAlignment):
    def __init__(self, channels, input_channels, classes, settings):
        super().__init__(channels, input_channels, classes, settings)
        self.variant = settings['b_variant']
        if self.variant not in VARIANTS: raise ValueError(self.variant)
        self.spec.update(b_variant=self.variant, source_anchor=True)
        self.bins = 8 if self.variant == 'path8' else 4
        self.frequency_bins = 8
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(11112026)
            self.temporal = ResidualTemporal(self.variant) if self.variant in ('tcn', 'transformer') else None
            self.bandpass = SincBands() if self.variant == 'sinc' else None
            self.scattering = LocalScattering() if self.variant == 'scattering' else None

    @property
    def descriptor_dims(self):
        frequency = 16*17 if self.variant in ('tcn', 'transformer', 'path8') else 16*self.frequency_bins*(7 if self.variant == 'scattering' else 4)
        return (16, 16*self.bins, frequency)

    def prepare_maps(self, features, valid_lengths):
        """先变换完整有效序列，再读取窗口；源/目标/参照库均走此入口。"""
        result = []
        for level, feature in enumerate(features):
            zs, bands = [], []
            for batch in range(len(feature)):
                valid = int(valid_lengths[level][batch])
                z = torch.einsum('dc,ct->dt', self.projection, feature[batch, :, :valid])[None]
                if self.temporal is not None: z = self.temporal(z)
                if self.variant == 'wavelet': band = starlet(z)
                elif self.variant == 'sinc': band = self.bandpass(z)
                elif self.variant == 'scattering': band = self.scattering(z)
                else: band = None
                zs.append(F.pad(z[0], (0, feature.shape[-1]-valid)))
                if band is not None: bands.append(F.pad(band[0], (0, feature.shape[-1]-valid)))
            result.append((torch.stack(zs), torch.stack(bands) if bands else None))
        return result

    def describe_windows(self, z, semantics, bands=None):
        q = semantics.detach().clamp(0, 1)
        pooled, path, weighted = pooled_descriptors(z, q, self.bins)
        if bands is None:
            window = torch.hann_window(z.shape[-1], periodic=False, device=z.device, dtype=z.dtype)
            frequency = torch.log1p(torch.fft.rfft(weighted*window, norm='ortho').abs().square()).flatten(2)
        else:
            # 先分解视觉特征，再施加语义权重；分类概率的突变不生成滤波系数。
            n, d, scales, t = bands.shape; classes = q.shape[1]
            energy = torch.log1p(bands.square())[:, None]*q[:, :, None, None]
            frequency = F.adaptive_avg_pool1d(energy.reshape(n*classes, d*scales, t), self.frequency_bins)
            mass = F.adaptive_avg_pool1d(q.reshape(n*classes, 1, t), self.frequency_bins)
            frequency = (frequency/mass.clamp_min(1e-6)).reshape(n, classes, -1)
        return [F.normalize(value, dim=-1) for value in (pooled, path, frequency)], q.mean(-1)

    def read_windows(self, videos, prepared, semantics, strides):
        zrows, qrows, brows, levels = [], [], [], []
        ratios = torch.linspace(0, 1, 32, device=prepared[0][0].device)
        for batch, video in enumerate(videos):
            for level, ((z, band), stride) in enumerate(zip(prepared, strides)):
                valid = min(z.shape[-1], math.ceil(video['feats'].shape[-1]/stride))
                if valid < 4: continue
                width = min(valid, 32)
                for start in rotating_starts(valid, width, 2, int(self.calls)):
                    positions = start+(width-1)*ratios
                    zrows.append(sample(z[batch], positions))
                    qrows.append(sample(semantics[level][batch].detach().T, positions))
                    if band is not None: brows.append(sample(band[batch].flatten(0, 1), positions).reshape(16, band.shape[2], 32))
                    levels.append(level)
        if not zrows: return None
        return torch.stack(zrows), torch.stack(qrows), torch.stack(brows) if brows else None, torch.tensor(levels, device=ratios.device)

    def forward(self, model, source_videos, target_videos, source_features, target_features,
                epoch, source_semantics=None, target_semantics=None):
        if any(v['segments'].numel() or v['labels'].numel() for v in target_videos):
            raise ValueError('目标TRAIN GT必须为空')
        zero = source_features[0].sum()*0
        if epoch < self.spec['start_epoch']: return zero, dict(active=zero.detach())
        strides = model.rq3_capacity.strides
        def prepare(videos, feats):
            valid = [[min(x.shape[-1], math.ceil(v['feats'].shape[-1]/s)) for v in videos] for x, s in zip(feats, strides)]
            return self.prepare_maps(feats, valid)
        # 源端整条B计算为本步参照；学习分支从目标端更新，残差/频带约束防退化。
        with torch.no_grad():
            source = self.read_windows(source_videos, prepare(source_videos, source_features), source_semantics, strides)
            if source is not None: sx, sa = self.describe_windows(*source[:3])
        target = self.read_windows(target_videos, prepare(target_videos, target_features), target_semantics, strides)
        terms, strengths = [], []
        if source is not None and target is not None:
            tx, ta = self.describe_windows(*target[:3]); sl, tl = source[3], target[3]
            for level in sl.unique().tolist():
                a, b = sl == level, tl == level
                if not b.any(): continue
                loss, strength, count = conditional_discrepancy([x[a] for x in sx], [x[b] for x in tx], sa[a], ta[b], self.spec['bandwidths'])
                if count: terms.append(loss); strengths.append(strength)
        value = torch.stack(terms).mean() if terms else zero
        ramp = min(1., (epoch-self.spec['start_epoch']+1)/self.spec['ramp_epochs'])
        if self.training: self.calls.add_(1)
        return ramp*self.spec['alignment_weight']*value, dict(active=zero.detach()+1, alignment=value.detach(),
            aligned_groups=zero.detach()+len(terms), ramp=zero.detach()+ramp,
            evidence_strength=torch.stack(strengths).mean() if strengths else zero.detach())

    def fingerprint(self):
        h = hashlib.sha256(json.dumps(dict(spec=self.spec, dims=self.descriptor_dims), sort_keys=True).encode())
        h.update(Path(__file__).read_bytes())
        for name, tensor in sorted(self.state_dict().items()):
            if name == 'calls': continue
            h.update(str((name,tuple(tensor.shape),tensor.dtype)).encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
