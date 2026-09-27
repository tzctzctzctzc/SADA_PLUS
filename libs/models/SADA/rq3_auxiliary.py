"""RQ3三种接入方式；只加辅助损失，不改检测头、边界或原SADA路由。"""

import torch
from torch import nn
from torch.nn import functional as F

from .losses_aux import sigmoid_focal_loss


STRATEGIES = ("hard", "verify", "prototype")


def candidate_routes(videos, points, masks, logits, cache, levels):
    """缓存是绝对SlowFast坐标；仅命中本次crop内、原FPN网格上的点。"""
    routes = []
    for level, (grid, valid, output) in enumerate(zip(points, masks, logits)):
        valid = valid.reshape(output.shape[:2]).bool()
        eligible = torch.zeros_like(valid)
        labels = torch.zeros_like(valid, dtype=torch.long)
        if level in levels:
            coordinates = grid[:, 0].detach().cpu().tolist()
            if any(abs(value - round(value)) > 1e-6 for value in coordinates):
                raise ValueError("RQ3 requires integer FPN feature coordinates")
            index_for = {int(round(value)): index for index, value in enumerate(coordinates)}
            for batch, video in enumerate(videos):
                metadata = video["metadata"]
                crop = int(metadata.get("crop_start", 0))
                for entry in cache.get(str(metadata["video_id"]), ()):
                    if level not in entry["source_fpn_levels"]:
                        continue
                    index = index_for.get(entry["absolute_feature_index"] - crop)
                    if index is not None and valid[batch, index]:
                        eligible[batch, index] = True
                        labels[batch, index] = entry["class_id"]
        routes.append((eligible, labels))
    return routes


class RQ3Auxiliary(nn.Module):
    def __init__(self, strategy, levels, dimensions, num_classes, cache,
                 budget=0.05, momentum=0.9):
        super().__init__()
        if strategy not in STRATEGIES:
            raise ValueError("unknown RQ3 strategy")
        if not 0 < budget <= 0.05 or not 0 <= momentum < 1:
            raise ValueError("invalid RQ3 budget or prototype momentum")
        self.strategy, self.levels, self.cache = strategy, tuple(levels), cache
        self.budget, self.momentum, self.num_classes = budget, momentum, num_classes
        # 三策略都没有可学习参数，不消耗初始化RNG；buffer可随checkpoint恢复。
        self.register_buffer("prototypes", torch.zeros(len(dimensions), num_classes, dimensions[0]))
        self.register_buffer("ready", torch.zeros(len(dimensions), num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update_source(self, features, masks, labels):
        """只读source GT；每层每类先求批均值，再更新单位EMA原型。"""
        for level in self.levels:
            values = features[level].detach().transpose(1, 2).float()
            valid = masks[level].reshape(values.shape[:2]).bool()
            for class_id in range(self.num_classes):
                selected = valid & (labels[level][..., class_id] > 0)
                if not selected.any():
                    continue
                center = F.normalize(values[selected], dim=-1).mean(dim=0)
                if center.norm() <= 1e-8:
                    continue
                center = F.normalize(center, dim=0)
                if self.ready[level, class_id]:
                    center = self.momentum * self.prototypes[level, class_id] + (1 - self.momentum) * center
                self.prototypes[level, class_id].copy_(F.normalize(center, dim=0))
                self.ready[level, class_id] = True

    def forward(self, videos, points, target_masks, target_logits, target_features,
                source_features, source_masks, source_labels, source_loss,
                active, alpha, gamma=2.0):
        if self.strategy == "prototype":
            self.update_source(source_features, source_masks, source_labels)
        zero = source_loss * 0.0
        if not active:
            return {"loss": zero, "raw_loss": zero, "eligible": 0, "selected": 0,
                    "agreement": 0, "budget": zero.detach(), "scale": zero.detach(),
                    "gradient_norm": zero.detach()}
        routes = candidate_routes(videos, points, target_masks, target_logits, self.cache, self.levels)
        terms, gradient_inputs = [], []
        eligible_count = selected_count = agreement_count = 0
        for level, (eligible, labels) in enumerate(routes):
            eligible_count += int(eligible.sum())
            if not eligible.any():
                continue
            logits = target_logits[level]
            confidence, predicted = logits.detach().sigmoid().max(dim=-1)
            agrees = eligible & (predicted == labels)
            agreement_count += int(agrees.sum())
            selected = eligible
            if self.strategy == "verify":
                selected = agrees & (confidence >= alpha)
            elif self.strategy == "prototype":
                selected = eligible & self.ready[level, labels]
            selected_count += int(selected.sum())
            if not selected.any():
                continue
            if self.strategy == "prototype":
                values = target_features[level].transpose(1, 2)[selected].float()
                centers = self.prototypes[level, labels[selected]].detach().clone()
                terms.append((1 - F.cosine_similarity(values, centers, dim=-1)).clamp_min(0).sum())
                gradient_inputs.append(target_features[level])
            else:
                targets = F.one_hot(labels[selected], logits.shape[-1]).to(logits.dtype)
                terms.append(sigmoid_focal_loss(logits[selected], targets, alpha=0.25,
                                               gamma=gamma, reduction="sum"))
                gradient_inputs.append(logits)
        # Verify拒绝项和缺失原型项为零；分母始终是共同基础候选数。
        raw = sum(terms, zero) / max(eligible_count, 1)
        cap = self.budget * source_loss.detach().clamp_min(0)
        scale = torch.minimum(torch.ones_like(cap), cap / raw.detach().clamp_min(1e-12))
        loss = raw * scale.detach()
        # 记录接入张量的实际梯度范数；不宣称不同策略梯度预算完全相同。
        norm = zero.detach()
        if gradient_inputs and torch.is_grad_enabled() and loss.requires_grad:
            gradients = torch.autograd.grad(loss, gradient_inputs, retain_graph=True, allow_unused=True)
            norm = sum((value.detach().float().square().sum() for value in gradients if value is not None),
                       zero.detach()).sqrt()
        return {"loss": loss, "raw_loss": raw, "eligible": eligible_count,
                "selected": selected_count, "agreement": agreement_count,
                "budget": cap, "scale": scale.detach(), "gradient_norm": norm}
