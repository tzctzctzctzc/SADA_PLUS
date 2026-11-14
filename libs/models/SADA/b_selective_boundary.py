"""Prototype/VLM 条件化的轻量边界响应迁移。

初始时严格退化为已有 StateBoundaryTransfer；训练只学习一个有界残差，
输入同时包含 B 的完整/扰动差异、A 的 Prototype 支持和 VLM 可靠性。
推理复用同一条件和同一残差，不新增检测头或目标域标注。
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .b_state_transfer import StateBoundaryTransfer
from .vlm_gated_boundary import class_reliability


class SelectiveBoundaryEvidence(StateBoundaryTransfer):
    def __init__(self, channels, input_channels, classes, settings):
        super().__init__(channels, input_channels, classes, settings)
        dim = sum(self.descriptor_dims)
        self.selective_residual = float(settings.get('selective_residual', .1))
        self.selective_head = nn.Sequential(
            nn.Linear(2 * dim + 3, 64), nn.GELU(), nn.Linear(64, 1))
        # 零初始化使新路线的冷初值与旧响应完全一致。
        nn.init.zeros_(self.selective_head[-1].weight)
        nn.init.zeros_(self.selective_head[-1].bias)
        self.spec.update(
            b_mechanism='vlm_prototype_conditioned_selective_boundary_evidence',
            selective_residual=self.selective_residual,
            selective_condition=('prototype_support', 'vlm_reliability', 'boundary_disagreement'),
            selective_target='source_true_boundary_gap_and_target_unlabeled_transfer',
            selective_zero_init=True,
            no_target_gt=True)

    def _prototype_gate(self, model, state, classes):
        out = state.new_ones(len(classes))
        branch = getattr(getattr(model, 'rq3_capacity', None), 'branch', None)
        if branch is None or not bool(branch.source_prototype_ready.any()):
            return out
        prototypes = F.normalize(branch.source_prototypes.detach(), dim=-1)
        start, width = self.descriptor_dims[0], self.descriptor_dims[1]
        path = state[:, start:start + width]
        if path.shape[1] != prototypes.shape[1]:
            if path.shape[1] % prototypes.shape[1]:
                raise ValueError('时序路径与Prototype维度无法对齐')
            path = path.reshape(len(path), -1, prototypes.shape[1]).mean(1)
        path = F.normalize(path, dim=-1)
        classes = classes.long().clamp(0, prototypes.shape[0] - 1)
        ready = branch.source_prototype_ready.detach()[classes]
        cosine = (path * prototypes[classes]).sum(-1)
        out[ready] = .5 + .5 * torch.sigmoid(cosine[ready])
        return out

    def _vlm_gate(self, model, video, classes, device, dtype):
        cache = getattr(getattr(model, 'rq3_capacity', None), '_moe_target_cache', None)
        values = [class_reliability(cache, video['metadata']['video_id'], int(c), self.vlm_gate_threshold)
                  for c in classes.detach().cpu().tolist()]
        return torch.as_tensor(values, device=device, dtype=dtype)

    def _selective_response(self, model, video, item):
        state = item['state']
        classes = item['classes']
        # real/swap 响应差异是同一 B 证据的边界不确定性代理。
        disagreement = item['response'].detach().abs().clamp(0, 1)
        proto = self._prototype_gate(model, state, classes)
        vlm = self._vlm_gate(model, video, classes, state.device, state.dtype)
        uncertainty = torch.exp(-.5 * disagreement)
        cond = torch.stack((proto, vlm, uncertainty), dim=-1)
        residual = self.selective_residual * torch.tanh(
            self.selective_head(torch.cat((state, cond), dim=-1)).squeeze(-1))
        item = dict(item)
        item['response'] = item['response'] + residual
        item['selective_gate'] = torch.sqrt((proto * vlm).clamp(.0625, 1.)) * uncertainty
        return item

    def _evidence(self, model, videos, features, outputs, batch, rows, bank):
        item = super()._evidence(model, videos, features, outputs, batch, rows, bank)
        if item is None:
            return None
        return self._selective_response(model, videos[batch], item)

    @torch.no_grad()
    def inference_gate(self, model, video, desc, labels, real, swapped):
        """推理时使用与训练相同的条件；返回乘在 B 偏离量上的有界门控。"""
        d = torch.as_tensor(desc, device=model.device, dtype=torch.float32)
        if not len(d):
            return real
        base = F.normalize(d, dim=-1)
        diff = torch.as_tensor(real, device=d.device, dtype=d.dtype) - torch.as_tensor(swapped, device=d.device, dtype=d.dtype)
        state = torch.cat((base, base * diff[:, None].clamp(-1, 1)), dim=-1)
        classes = torch.as_tensor(labels, device=d.device, dtype=torch.long)
        proto = self._prototype_gate(model, state, classes)
        vlm = self._vlm_gate(model, video, classes, d.device, d.dtype)
        uncertainty = torch.exp(-.5 * diff.abs().clamp(0, 1))
        cond = torch.stack((proto, vlm, uncertainty), dim=-1)
        residual = self.selective_residual * torch.tanh(
            self.selective_head(torch.cat((state, cond), dim=-1)).squeeze(-1))
        gate = torch.sqrt((proto * vlm).clamp(.0625, 1.)) * uncertainty * torch.exp(residual)
        return gate.clamp(.25, 1.).cpu().numpy()

    def fingerprint(self):
        h = hashlib.sha256(super().fingerprint().encode())
        h.update(Path(__file__).read_bytes())
        return h.hexdigest()
