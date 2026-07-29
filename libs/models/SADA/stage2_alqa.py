import copy
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .stage2_utils import (
    aligned_tiou_tnd,
    alqa_joint_score,
    select_same_budget_alqa,
)


class ALQAQualityHead(nn.Module):
    """Source-calibrated tIoU/tND outputs on regression hidden features."""

    def __init__(self, feature_dim):
        super().__init__()
        self.output = nn.Conv1d(feature_dim, 2, kernel_size=3, padding=1)

    def forward(self, features):
        return self.output(features)


def regression_hidden_and_offsets(regression_head, fpn_features, fpn_masks):
    hidden, offsets = [], []
    for level, (features, mask) in enumerate(zip(fpn_features, fpn_masks)):
        current = features
        for index in range(len(regression_head.head)):
            current, _ = regression_head.head[index](current, mask)
            current = regression_head.act(regression_head.norm[index](current))
        level_offsets, _ = regression_head.offset_head(current, mask)
        hidden.append(current)
        offsets.append(F.relu(regression_head.scale[level](level_offsets)))
    return tuple(hidden), tuple(offsets)


def ensemble_quality_maps(quality_heads, hidden_features):
    if not len(quality_heads):
        raise ValueError("ALQA requires at least one quality head")
    per_head = [
        [head(features).sigmoid() for features in hidden_features]
        for head in quality_heads
    ]
    return tuple(
        torch.stack([values[level] for values in per_head]).mean(dim=0)
        for level in range(len(hidden_features))
    )


def _decode_segments(offsets, centers, strides):
    return torch.stack((
        centers - offsets[:, 0] * strides,
        centers + offsets[:, 1] * strides,
    ), dim=-1)


def _move_video(video, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in video.items()
    }


@torch.no_grad()
def collect_source_quality_sequences(
        model, dataset, device, levels, batch_size, seed, mode):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    sequences = []
    positives = 0
    for start in range(0, len(dataset), batch_size):
        batch = [
            _move_video(dataset[index], device)
            for index in range(start, min(start + batch_size, len(dataset)))
        ]
        inputs, masks = model.preprocessing(batch, mode=mode)
        backbone_features, backbone_masks, _ = model.backbone_target(
            inputs, masks
        )
        fpn_features, fpn_masks = model.neck_target(
            backbone_features, backbone_masks
        )
        hidden, offsets = regression_hidden_and_offsets(
            model.reg_head_target, fpn_features, fpn_masks
        )
        points = model.point_generator(fpn_features)
        gt_cls, gt_offsets = model.label_points(
            points,
            [video["segments"] for video in batch],
            [video["labels"] for video in batch],
        )

        point_offset = 0
        for level, point in enumerate(points):
            next_offset = point_offset + point.shape[0]
            if level not in levels:
                point_offset = next_offset
                continue
            positive = torch.zeros(
                (len(batch), point.shape[0]),
                dtype=torch.bool, device=device,
            )
            targets = torch.zeros(
                (len(batch), 2, point.shape[0]),
                dtype=torch.float32, device=device,
            )
            centers = point[:, 0]
            strides = point[:, 3]
            predicted_segments = [
                _decode_segments(
                    offsets[level][batch_index].transpose(0, 1),
                    centers,
                    strides,
                )
                for batch_index in range(len(batch))
            ]
            for batch_index in range(len(batch)):
                current_positive = (
                    gt_cls[batch_index][
                        point_offset:next_offset
                    ].sum(dim=-1) > 0
                ) & fpn_masks[level][batch_index, 0]
                positive[batch_index] = current_positive
                if not current_positive.any():
                    continue
                target_segments = _decode_segments(
                    gt_offsets[batch_index][point_offset:next_offset],
                    centers,
                    strides,
                )
                tiou, tnd = aligned_tiou_tnd(
                    predicted_segments[batch_index][current_positive],
                    target_segments[current_positive],
                )
                targets[batch_index, 0, current_positive] = tiou
                targets[batch_index, 1, current_positive] = tnd
            if positive.any():
                sequences.append({
                    "features": hidden[level].detach().cpu(),
                    "targets": targets.cpu(),
                    "positive": positive.cpu(),
                    "level": level,
                })
                positives += int(positive.sum())
            point_offset = next_offset
    return sequences, positives


def _rank(values):
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(
        len(values), dtype=torch.float32, device=values.device
    )
    return ranks


def _correlation(left, right):
    if left.numel() < 2:
        return None
    left = left.float() - left.float().mean()
    right = right.float() - right.float().mean()
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return None
    return float((left * right).sum() / denominator)


def evaluate_quality_head(head, sequences, device):
    head.eval()
    loss_sum = 0.0
    count = 0
    predictions, targets = [], []
    with torch.no_grad():
        for sequence in sequences:
            features = sequence["features"].to(device)
            target = sequence["targets"].to(device)
            positive = sequence["positive"].to(device)
            logits = head(features).permute(0, 2, 1)[positive]
            selected_target = target.permute(0, 2, 1)[positive]
            loss_sum += float(F.binary_cross_entropy_with_logits(
                logits, selected_target, reduction="sum"
            ))
            count += int(selected_target.numel())
            predictions.append(logits.sigmoid().cpu())
            targets.append(selected_target.cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    predicted_diou = prediction[:, 0] - prediction[:, 1]
    target_diou = target[:, 0] - target[:, 1]
    return {
        "bce": loss_sum / max(count, 1),
        "positive_anchors": int(target.shape[0]),
        "diou_pearson": _correlation(predicted_diou, target_diou),
        "diou_spearman": _correlation(
            _rank(predicted_diou), _rank(target_diou)
        ),
    }


def calibrate_quality_heads(
        quality_heads, train_sequences, val_sequences, device,
        seeds, max_epochs, patience, learning_rate):
    results = {}
    for head, seed in zip(quality_heads, seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        head.output.reset_parameters()
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        best_state = copy.deepcopy(head.state_dict())
        best_epoch = 0
        best_bce = float("inf")
        remaining = patience
        history = []
        for epoch in range(1, max_epochs + 1):
            head.train()
            order = list(range(len(train_sequences)))
            random.Random(seed + epoch).shuffle(order)
            for index in order:
                sequence = train_sequences[index]
                features = sequence["features"].to(device)
                target = sequence["targets"].to(device)
                positive = sequence["positive"].to(device)
                logits = head(features).permute(0, 2, 1)[positive]
                selected_target = target.permute(0, 2, 1)[positive]
                loss = F.binary_cross_entropy_with_logits(
                    logits, selected_target
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation = evaluate_quality_head(head, val_sequences, device)
            history.append({
                "epoch": epoch,
                "validation_bce": validation["bce"],
            })
            if validation["bce"] < best_bce - 1e-6:
                best_bce = validation["bce"]
                best_epoch = epoch
                best_state = copy.deepcopy(head.state_dict())
                remaining = patience
            else:
                remaining -= 1
                if remaining == 0:
                    break
        head.load_state_dict(best_state)
        head.eval()
        results[str(seed)] = {
            "best_epoch": best_epoch,
            "validation": evaluate_quality_head(
                head, val_sequences, device
            ),
            "history": history,
        }
    return results
