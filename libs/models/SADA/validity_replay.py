"""Exact source-view cache and video-balanced replay for Validity Head training."""

import copy
import random
from collections import defaultdict

import torch
from torch.nn import functional as F


def exact_view_windows(
    feat_len,
    max_seq_len,
    crop_ratio,
    views_per_video,
    alignment,
    rng,
):
    feat_len = int(feat_len)
    max_seq_len = int(max_seq_len)
    views_per_video = int(views_per_video)
    alignment = int(alignment)
    if min(feat_len, max_seq_len, views_per_video, alignment) <= 0:
        raise ValueError("exact-view sizes must be positive")

    if feat_len > max_seq_len:
        lengths = (max_seq_len,)
    elif crop_ratio is None:
        lengths = (feat_len,)
    else:
        lower = max(round(float(crop_ratio[0]) * feat_len), 1)
        upper = min(round(float(crop_ratio[1]) * feat_len), feat_len)
        lengths = tuple(range(lower, upper + 1))
    if not lengths:
        raise ValueError("exact-view crop range is empty")

    max_phase = min(alignment - 1, feat_len - min(lengths))
    candidates_by_phase = {}
    for phase in range(max_phase + 1):
        candidates = []
        for length in lengths:
            max_start = feat_len - length
            candidates.extend(
                (start, start + length)
                for start in range(phase, max_start + 1, alignment)
            )
        candidates_by_phase[phase] = tuple(sorted(set(candidates)))

    maximum = max(len(values) for values in candidates_by_phase.values())
    target = min(views_per_video, maximum)
    phases = [
        phase
        for phase, values in candidates_by_phase.items()
        if len(values) >= target
    ]
    phase = rng.choice(phases)
    candidates = list(candidates_by_phase[phase])
    rng.shuffle(candidates)
    windows = tuple(sorted(candidates[:target]))
    if len(set(windows)) != len(windows):
        raise RuntimeError("exact-view sampler produced duplicate windows")
    if any(start % alignment != phase for start, _ in windows):
        raise RuntimeError("exact-view windows do not share one lattice phase")
    return {"phase": phase, "windows": windows}


def crop_source_view(full_video, start, end, trunc_thresh):
    start, end = int(start), int(end)
    feat_len = int(full_video["feats"].shape[-1])
    if not (0 <= start < end <= feat_len):
        raise ValueError("source replay crop is outside the full video")
    segments = full_video["segments"]
    if segments is None or segments.numel() == 0:
        return None

    window_start = segments.new_tensor(float(start))
    window_end = segments.new_tensor(float(end))
    left = torch.maximum(segments[:, 0], window_start)
    right = torch.minimum(segments[:, 1], window_end)
    intersection = (right - left).clamp_min(0.0)
    ratio = intersection / (segments[:, 1] - segments[:, 0]).abs()
    keep = ratio >= float(trunc_thresh)
    if not keep.any():
        return None

    metadata = copy.deepcopy(full_video["metadata"])
    metadata["original_feat_len"] = feat_len
    metadata["crop_feat_len"] = end - start
    metadata["crop_start"] = start
    return {
        "feats": full_video["feats"][:, start:end].clone(),
        "segments": torch.stack((left[keep], right[keep]), dim=1) - start,
        "labels": full_video["labels"][keep].clone(),
        "metadata": metadata,
    }


class FrozenAnchorReplayPool:
    def __init__(
        self,
        features,
        offsets,
        levels,
        labels,
        video_indices,
        absolute_points,
        crop_indices,
        video_ids,
        seed,
    ):
        tensors = (
            features,
            offsets,
            levels,
            labels,
            video_indices,
            absolute_points,
            crop_indices,
        )
        if not tensors or any(len(value) != len(features) for value in tensors):
            raise ValueError("replay tensors have inconsistent lengths")
        if len(features) == 0:
            raise ValueError("replay pool is empty")
        self.features = features
        self.offsets = offsets
        self.levels = levels.long()
        self.labels = labels.bool()
        self.video_indices = video_indices.long()
        self.absolute_points = absolute_points.long()
        self.crop_indices = crop_indices.long()
        self.video_ids = tuple(str(value) for value in video_ids)
        self.n_levels = int(self.levels.max().item()) + 1
        self.rng = random.Random(int(seed))

        base_records = defaultdict(list)
        for index, (video, level, point) in enumerate(zip(
            self.video_indices.tolist(),
            self.levels.tolist(),
            self.absolute_points.tolist(),
        )):
            base_records[(video, level, point)].append(index)

        self.key_records = {}
        self.consistent_pair_keys = set()
        self.cell_video_keys = defaultdict(lambda: defaultdict(list))
        inconsistent = 0
        for base_key, indices in base_records.items():
            label_values = {bool(self.labels[index]) for index in indices}
            if len(label_values) > 1:
                inconsistent += 1
            for label in sorted(label_values):
                key = (*base_key, label)
                matching = tuple(
                    index for index in indices if bool(self.labels[index]) == label
                )
                self.key_records[key] = matching
                video, level, _ = base_key
                self.cell_video_keys[(level, label)][video].append(key)
                crop_count = len({int(self.crop_indices[index]) for index in matching})
                if len(label_values) == 1 and crop_count >= 2:
                    self.consistent_pair_keys.add(key)

        expected_cells = {
            (level, label)
            for level in range(self.n_levels)
            for label in (False, True)
        }
        missing = sorted(expected_cells - set(self.cell_video_keys))
        if missing:
            raise ValueError("replay pool has empty level/class cells: {}".format(missing))

        self.video_orders = {}
        self.video_cursors = {}
        for cell in sorted(expected_cells):
            order = sorted(self.cell_video_keys[cell])
            self.rng.shuffle(order)
            self.video_orders[cell] = order
            self.video_cursors[cell] = 0
        self.inconsistent_label_keys = inconsistent
        self.unique_anchor_keys = len(base_records)

    def _next_videos(self, cell, requested):
        order = self.video_orders[cell]
        count = int(requested)
        if count > len(order):
            raise ValueError(
                "replay cell {} has {} videos, fewer than requested {}".format(
                    cell, len(order), count
                )
            )
        chosen = []
        while len(chosen) < count:
            cursor = self.video_cursors[cell]
            if cursor >= len(order):
                self.rng.shuffle(order)
                cursor = 0
            video = order[cursor]
            self.video_cursors[cell] = cursor + 1
            if video not in chosen:
                chosen.append(video)
        return chosen

    def sample_groups(self, videos_per_cell):
        groups = []
        for level in range(self.n_levels):
            for label in (False, True):
                cell = (level, label)
                for video in self._next_videos(cell, videos_per_cell):
                    key = self.rng.choice(self.cell_video_keys[cell][video])
                    records = self.key_records[key]
                    if key in self.consistent_pair_keys:
                        records_by_crop = defaultdict(list)
                        for index in records:
                            records_by_crop[int(self.crop_indices[index])].append(index)
                        crops = self.rng.sample(sorted(records_by_crop), 2)
                        chosen_records = tuple(
                            self.rng.choice(records_by_crop[crop]) for crop in crops
                        )
                    else:
                        chosen_records = (self.rng.choice(records),)
                    groups.append({
                        "level": level,
                        "label": label,
                        "video_index": video,
                        "absolute_point": key[2],
                        "record_indices": chosen_records,
                        "consistency_pair": len(chosen_records) == 2,
                    })
        return groups

    def stats(self):
        cells = {}
        for level in range(self.n_levels):
            for label in (False, True):
                cell = (level, label)
                keys = {
                    key
                    for values in self.cell_video_keys[cell].values()
                    for key in values
                }
                cells["L{}_{}".format(level, "positive" if label else "negative")] = {
                    "videos": len(self.cell_video_keys[cell]),
                    "unique_anchors": len(keys),
                    "pairable_anchors": sum(
                        key in self.consistent_pair_keys for key in keys
                    ),
                }
        return {
            "records": len(self.features),
            "unique_anchor_keys": self.unique_anchor_keys,
            "pairable_anchor_keys": len(self.consistent_pair_keys),
            "inconsistent_label_keys": self.inconsistent_label_keys,
            "cells": cells,
        }


def exact_replay_batch(model, pool, device, videos_per_cell, consistency_weight):
    """Train on equal level/class cells and exact-key cross-view pairs."""
    groups = pool.sample_groups(videos_per_cell)
    flat_indices = [
        index for group in groups for index in group["record_indices"]
    ]
    flat_levels = torch.tensor(
        [group["level"] for group in groups for _ in group["record_indices"]],
        dtype=torch.long,
    )
    features = pool.features[flat_indices].float().to(device)
    offsets = pool.offsets[flat_indices].to(device)

    row_logits = torch.empty(len(flat_indices), device=device)
    head_features = []
    head_offsets = []
    head_masks = []
    level_rows = []
    for level in range(model.n_levels):
        rows = torch.where(flat_levels == level)[0]
        if rows.numel() == 0:
            raise RuntimeError("replay batch has an empty FPN level")
        device_rows = rows.to(device)
        level_rows.append(device_rows)
        head_features.append(features[device_rows].unsqueeze(-1))
        head_offsets.append(offsets[device_rows].unsqueeze(-1))
        head_masks.append(
            torch.ones((len(rows), 1, 1), dtype=torch.bool, device=device)
        )
    logits = model.validity_head(
        tuple(head_features),
        tuple(head_offsets),
        tuple(head_masks),
        model.fpn_strides,
    )
    for rows, level_logits in zip(level_rows, logits):
        row_logits[rows] = level_logits[:, 0]

    cell_classification = {}
    cell_consistency = {}
    cursor = 0
    for group in groups:
        count = len(group["record_indices"])
        current_logits = row_logits[cursor:cursor + count]
        cursor += count
        target = current_logits.new_full(
            current_logits.shape, float(group["label"])
        )
        cell = (group["level"], group["label"])
        cell_classification.setdefault(cell, []).append(
            F.binary_cross_entropy_with_logits(current_logits, target)
        )
        if group["consistency_pair"]:
            probabilities = current_logits.sigmoid()
            cell_consistency.setdefault(cell, []).append(
                (probabilities[0] - probabilities[1]).abs()
            )

    classification = torch.stack([
        torch.stack(cell_classification[cell]).mean()
        for cell in sorted(cell_classification)
    ]).mean()
    if cell_consistency:
        consistency = torch.stack([
            torch.stack(cell_consistency[cell]).mean()
            for cell in sorted(cell_consistency)
        ]).mean()
    else:
        consistency = classification * 0.0
    loss = classification + float(consistency_weight) * consistency
    per_level = []
    for level in range(model.n_levels):
        positive = sum(
            group["level"] == level and group["label"] for group in groups
        )
        negative = sum(
            group["level"] == level and not group["label"] for group in groups
        )
        per_level.append({
            "count": positive + negative,
            "positive": positive,
            "negative": negative,
        })
    return loss, {
        "count": len(groups),
        "rows": len(flat_indices),
        "positive": sum(group["label"] for group in groups),
        "negative": sum(not group["label"] for group in groups),
        "consistency_pairs": sum(group["consistency_pair"] for group in groups),
        "classification_loss": float(classification.detach()),
        "consistency_loss": float(consistency.detach()),
        "per_level": per_level,
    }
