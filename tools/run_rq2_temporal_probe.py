"""Run the formal source-only unified temporal probe for RQ2.

All model selection is restricted to grouped source-train folds. Source-val and
target-val are read only after the probe configuration and final epoch count are
frozen. The VLM is represented by precomputed time-aligned features, so the only
trainable parameters are the class-conditional linear start/end heads.
"""

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tools.run_clip_target_adapter import LABELS, read_jsonl
except ModuleNotFoundError as error:
    if error.name != "tools":
        raise
    from run_clip_target_adapter import LABELS, read_jsonl


PROTOCOL = "rq2-source-only-temporal-probe-v1"
FEATURE_SCHEMA = "rq2-qwen-time-aligned-vision-tower-v1"
SHIFTS = ("S1", "S2", "S3", "S4", "S5", "S6")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def membership_role(row, shift):
    roles = [
        item["role"]
        for item in row["memberships"]
        if item.get("shift") == shift
    ]
    if len(roles) > 1:
        raise ValueError(
            "candidate {} has multiple roles for {}".format(
                row["candidate_id"], shift
            )
        )
    return roles[0] if roles else None


def role_indices(rows, shift, role):
    return np.asarray(
        [index for index, row in enumerate(rows) if membership_role(row, shift) == role],
        dtype=np.int64,
    )


def load_feature_index(index_paths, feature_root, manifest_hash):
    feature_root = Path(feature_root)
    available_files = set(os.listdir(feature_root))
    by_id = {}
    model_identity = None
    feature_schema = None
    for index_path in index_paths:
        for row in read_jsonl(index_path):
            if row.get("status") != "ok":
                continue
            if row.get("manifest_sha256") != manifest_hash:
                raise ValueError("feature index manifest hash mismatch")
            if row.get("schema_version") != FEATURE_SCHEMA:
                raise ValueError("unsupported feature schema")
            candidate_id = row["candidate_id"]
            feature_name = row["feature_file"]
            if Path(feature_name).name != feature_name:
                raise ValueError("feature index contains a non-flat path")
            feature_path = feature_root / feature_name
            if feature_name not in available_files:
                raise FileNotFoundError(str(feature_path))
            identity = (row.get("model"), row.get("model_revision"))
            if model_identity is None:
                model_identity = identity
                feature_schema = row.get("schema_version")
            elif identity != model_identity:
                raise ValueError("feature indexes contain multiple model identities")
            previous = by_id.get(candidate_id)
            if previous is not None and previous != feature_path:
                raise ValueError("conflicting feature rows for {}".format(candidate_id))
            by_id[candidate_id] = feature_path
    return by_id, model_identity, feature_schema


def prepare_feature_bank(
    rows,
    manifest_path,
    index_paths,
    feature_root,
    cache_dir,
    rebuild=False,
    loader_workers=16,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / "features.float16.npy"
    timestamp_path = cache_dir / "timestamps.float32.npy"
    length_path = cache_dir / "lengths.int16.npy"
    offset_path = cache_dir / "offsets.int64.npy"
    metadata_path = cache_dir / "feature_bank.json"
    manifest_hash = sha256_file(manifest_path)
    index_hashes = {str(path): sha256_file(path) for path in index_paths}
    expected_ids = [row["candidate_id"] for row in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("manifest contains duplicate candidate ids")

    if metadata_path.is_file() and not rebuild:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("protocol") == PROTOCOL
            and metadata.get("manifest_sha256") == manifest_hash
            and metadata.get("index_sha256") == index_hashes
            and metadata.get("rows") == len(rows)
            and feature_path.is_file()
            and timestamp_path.is_file()
            and length_path.is_file()
            and offset_path.is_file()
        ):
            features = np.load(feature_path, mmap_mode="r")
            timestamps = np.load(timestamp_path, mmap_mode="r")
            lengths = np.load(length_path, mmap_mode="r")
            offsets = np.load(offset_path, mmap_mode="r")
            if list(features.shape) != metadata["feature_shape"]:
                raise ValueError("feature bank shape differs from metadata")
            if list(timestamps.shape) != metadata["timestamp_shape"]:
                raise ValueError("timestamp bank shape differs from metadata")
            if list(lengths.shape) != metadata["length_shape"]:
                raise ValueError("length bank shape differs from metadata")
            if list(offsets.shape) != metadata["offset_shape"]:
                raise ValueError("offset bank shape differs from metadata")
            return features, timestamps, lengths, offsets, metadata

    feature_index, model_identity, feature_schema = load_feature_index(
        index_paths, feature_root, manifest_hash
    )
    missing = [candidate_id for candidate_id in expected_ids if candidate_id not in feature_index]
    if missing:
        raise RuntimeError(
            "feature index incomplete: missing {}/{} candidates; first={}".format(
                len(missing), len(rows), missing[0]
            )
        )

    with np.load(feature_index[expected_ids[0]]) as arrays:
        first_features = arrays["features"]
        first_timestamps = arrays["timestamps_sec"]
    if first_features.ndim != 2 or first_timestamps.ndim != 1:
        raise ValueError("invalid feature array rank")
    _, dimension = first_features.shape
    if len(first_timestamps) != first_features.shape[0]:
        raise ValueError("feature/timestamp length mismatch")

    expected_lengths = np.asarray(
        [int(row["frame_count"]) for row in rows], dtype=np.int64
    )
    if np.any(expected_lengths <= 0):
        raise ValueError("manifest contains non-positive frame counts")
    max_time_steps = int(expected_lengths.max())
    total_time_steps = int(expected_lengths.sum())
    expected_offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    expected_offsets[1:] = np.cumsum(expected_lengths)

    feature_tmp = feature_path.with_suffix(feature_path.suffix + ".tmp")
    timestamp_tmp = timestamp_path.with_suffix(timestamp_path.suffix + ".tmp")
    length_tmp = length_path.with_suffix(length_path.suffix + ".tmp")
    offset_tmp = offset_path.with_suffix(offset_path.suffix + ".tmp")
    features = np.lib.format.open_memmap(
        feature_tmp,
        mode="w+",
        dtype=np.float16,
        shape=(total_time_steps, dimension),
    )
    timestamps = np.lib.format.open_memmap(
        timestamp_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), max_time_steps),
    )
    lengths = np.lib.format.open_memmap(
        length_tmp, mode="w+", dtype=np.int16, shape=(len(rows),)
    )
    offsets = np.lib.format.open_memmap(
        offset_tmp, mode="w+", dtype=np.int64, shape=(len(rows) + 1,)
    )
    timestamps[:] = np.nan
    offsets[:] = expected_offsets
    def load_feature(index):
        candidate_id = expected_ids[index]
        with np.load(feature_index[candidate_id]) as arrays:
            return (
                index,
                candidate_id,
                arrays["features"].copy(),
                arrays["timestamps_sec"].copy(),
            )

    with ThreadPoolExecutor(max_workers=loader_workers) as executor:
        for batch_start in range(0, len(rows), 256):
            batch_stop = min(batch_start + 256, len(rows))
            loaded = executor.map(load_feature, range(batch_start, batch_stop))
            for index, candidate_id, row_features, row_timestamps in loaded:
                time_steps = int(expected_lengths[index])
                if row_features.shape != (time_steps, dimension):
                    raise ValueError(
                        "inconsistent feature shape for {}".format(candidate_id)
                    )
                if row_timestamps.shape != (time_steps,):
                    raise ValueError(
                        "inconsistent timestamp shape for {}".format(candidate_id)
                    )
                if not np.all(np.isfinite(row_features)) or not np.all(
                    np.isfinite(row_timestamps)
                ):
                    raise ValueError("non-finite feature data for {}".format(candidate_id))
                token_start = int(expected_offsets[index])
                token_end = int(expected_offsets[index + 1])
                features[token_start:token_end] = row_features.astype(
                    np.float16, copy=False
                )
                timestamps[index, :time_steps] = row_timestamps.astype(
                    np.float32, copy=False
                )
                lengths[index] = time_steps
            if batch_stop % 1024 == 0 or batch_stop == len(rows):
                print("feature_bank={}/{}".format(batch_stop, len(rows)), flush=True)
    features.flush()
    timestamps.flush()
    lengths.flush()
    offsets.flush()
    del features, timestamps, lengths, offsets
    os.replace(feature_tmp, feature_path)
    os.replace(timestamp_tmp, timestamp_path)
    os.replace(length_tmp, length_path)
    os.replace(offset_tmp, offset_path)

    metadata = {
        "protocol": PROTOCOL,
        "manifest_sha256": manifest_hash,
        "index_sha256": index_hashes,
        "rows": len(rows),
        "feature_shape": [total_time_steps, dimension],
        "timestamp_shape": [len(rows), max_time_steps],
        "length_shape": [len(rows)],
        "offset_shape": [len(rows) + 1],
        "feature_dtype": "float16",
        "timestamp_dtype": "float32",
        "feature_schema": feature_schema,
        "model": model_identity[0],
        "model_revision": model_identity[1],
    }
    write_json_atomic(metadata_path, metadata)
    return (
        np.load(feature_path, mmap_mode="r"),
        np.load(timestamp_path, mmap_mode="r"),
        np.load(length_path, mmap_mode="r"),
        np.load(offset_path, mmap_mode="r"),
        metadata,
    )


def quantized_targets(rows, timestamps, lengths, indices):
    starts = np.empty(len(indices), dtype=np.int64)
    ends = np.empty(len(indices), dtype=np.int64)
    for local_index, index in enumerate(indices):
        row = rows[int(index)]
        valid_timestamps = timestamps[int(index), : int(lengths[int(index)])]
        starts[local_index] = int(
            np.argmin(np.abs(valid_timestamps - float(row["gt_start_sec"])))
        )
        ends[local_index] = int(
            np.argmin(np.abs(valid_timestamps - float(row["gt_end_sec"])))
        )
        if ends[local_index] < starts[local_index]:
            ends[local_index] = starts[local_index]
    return starts, ends


def temporal_iou(prediction, target):
    intersection = max(
        0.0, min(prediction[1], target[1]) - max(prediction[0], target[0])
    )
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    return intersection / union if union > 0 else 0.0


def stable_index(value, size, seed):
    payload = "{}|{}".format(seed, value).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16) % size


def relative_to_absolute(row, center, duration):
    duration = min(max(float(duration), 0.0), 1.0)
    start = float(center) - duration / 2.0
    end = float(center) + duration / 2.0
    if start < 0.0:
        end -= start
        start = 0.0
    if end > 1.0:
        start -= end - 1.0
        end = 1.0
    window_start = float(row["proposal_start_sec"])
    window_length = float(row["window_length_sec"])
    return (
        window_start + max(0.0, start) * window_length,
        window_start + min(1.0, end) * window_length,
    )


def nonvisual_predictions(rows, train_indices, eval_indices, seed):
    class_centers = defaultdict(list)
    class_durations = defaultdict(list)
    for index in train_indices:
        row = rows[int(index)]
        window_start = float(row["proposal_start_sec"])
        window_length = float(row["window_length_sec"])
        label = int(row["gt_label_id"])
        class_centers[label].append(
            (
                (float(row["gt_start_sec"]) + float(row["gt_end_sec"])) / 2.0
                - window_start
            )
            / window_length
        )
        class_durations[label].append(
            (float(row["gt_end_sec"]) - float(row["gt_start_sec"]))
            / window_length
        )

    predictions = defaultdict(list)
    positions = (0.25, 0.5, 0.75)
    for index in eval_indices:
        row = rows[int(index)]
        label = int(row["gt_label_id"])
        if label not in class_durations:
            raise ValueError(
                "nonvisual prior has no source-train examples for class {}".format(
                    label
                )
            )
        duration_pool = class_durations[label]
        median_center = float(np.median(class_centers[label]))
        median_duration = float(np.median(duration_pool))
        chance_center = positions[
            stable_index(row["instance_id"], len(positions), seed)
        ]
        sampled_duration = duration_pool[
            stable_index(row["instance_id"], len(duration_pool), seed + 1)
        ]
        row_predictions = {
            "full_window": (
                float(row["proposal_start_sec"]),
                float(row["proposal_end_sec"]),
            ),
            "center_duration": relative_to_absolute(
                row, median_center, median_duration
            ),
            "balanced_position": relative_to_absolute(
                row, chance_center, median_duration
            ),
            "random_duration_matched": relative_to_absolute(
                row, chance_center, sampled_duration
            ),
        }
        for name, prediction in row_predictions.items():
            predictions[name].append(prediction)
    return dict(predictions)


def nonvisual_fold_scores(rows, train_indices, dev_indices, seed):
    predictions = nonvisual_predictions(rows, train_indices, dev_indices, seed)
    scores = {}
    for name, intervals in predictions.items():
        ious = []
        for index, prediction in zip(dev_indices, intervals):
            row = rows[int(index)]
            target = (float(row["gt_start_sec"]), float(row["gt_end_sec"]))
            ious.append(temporal_iou(prediction, target))
        scores[name] = float(np.mean(ious))
    return scores


def decode_probability_batches(start_probabilities, end_probabilities):
    count, time_steps = start_probabilities.shape
    scores = start_probabilities[:, :, None] * end_probabilities[:, None, :]
    invalid = np.tril_indices(time_steps, k=-1)
    scores[:, invalid[0], invalid[1]] = -1.0
    flat = np.argmax(scores.reshape(count, -1), axis=1)
    return flat // time_steps, flat % time_steps


def interval_metrics(
    rows, row_indices_value, timestamps, lengths, start_probs, end_probs
):
    starts, ends = decode_probability_batches(start_probs, end_probs)
    predictions = []
    for local_index, row_index in enumerate(row_indices_value):
        row_timestamps = timestamps[
            int(row_index), : int(lengths[int(row_index)])
        ]
        predictions.append(
            (
                float(row_timestamps[starts[local_index]]),
                float(row_timestamps[ends[local_index]]),
            )
        )
    return metrics_from_predictions(
        rows, row_indices_value, timestamps, lengths, predictions
    )


def metrics_from_predictions(
    rows, row_indices_value, timestamps, lengths, predictions
):
    ceilings = []
    ious = []
    start_errors = []
    end_errors = []
    for local_index, row_index in enumerate(row_indices_value):
        row = rows[int(row_index)]
        row_timestamps = timestamps[
            int(row_index), : int(lengths[int(row_index)])
        ]
        prediction = predictions[local_index]
        target = (float(row["gt_start_sec"]), float(row["gt_end_sec"]))
        oracle_start = int(np.argmin(np.abs(row_timestamps - target[0])))
        oracle_end = int(np.argmin(np.abs(row_timestamps - target[1])))
        if oracle_end < oracle_start:
            oracle_end = oracle_start
        ceiling = (
            float(row_timestamps[oracle_start]),
            float(row_timestamps[oracle_end]),
        )
        duration = max(float(row["window_length_sec"]), 1e-12)
        ceilings.append(temporal_iou(ceiling, target))
        ious.append(temporal_iou(prediction, target))
        start_errors.append(abs(prediction[0] - target[0]) / duration)
        end_errors.append(abs(prediction[1] - target[1]) / duration)

    grouped = defaultdict(list)
    for local_index, row_index in enumerate(row_indices_value):
        grouped[rows[int(row_index)]["instance_id"]].append(local_index)
    consistency = []
    consistent_correct = []
    complete_instances = 0
    for group in grouped.values():
        if len(group) != 3:
            continue
        complete_instances += 1
        pair_scores = [
            temporal_iou(predictions[group[left]], predictions[group[right]])
            for left, right in ((0, 1), (0, 2), (1, 2))
        ]
        consistency.append(float(np.mean(pair_scores)))
        consistent_correct.append(
            float(
                all(ious[index] >= 0.5 for index in group)
                and all(score >= 0.5 for score in pair_scores)
            )
        )
    return {
        "miou": float(np.mean(ious)),
        "recall_at_0.3": float(np.mean(np.asarray(ious) >= 0.3)),
        "recall_at_0.5": float(np.mean(np.asarray(ious) >= 0.5)),
        "recall_at_0.7": float(np.mean(np.asarray(ious) >= 0.7)),
        "start_normalized_mae": float(np.mean(start_errors)),
        "end_normalized_mae": float(np.mean(end_errors)),
        "quantized_oracle_ceiling": float(np.mean(ceilings)),
        "context_consistency": float(np.mean(consistency)) if consistency else None,
        "consistent_and_correct_at_0.5": (
            float(np.mean(consistent_correct)) if consistent_correct else None
        ),
        "complete_context_instances": complete_instances,
        "window_count": len(ious),
        "per_window_iou": ious,
        "predictions": predictions,
    }


def clustered_bootstrap(values, groups, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sums = np.bincount(inverse, weights=values)
    group_counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(
        0, len(unique_groups), size=(repeats, len(unique_groups))
    )
    estimates = group_sums[sampled].sum(axis=1) / group_counts[sampled].sum(axis=1)
    return [
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    ]


def add_distribution_metrics(metrics, rows, indices, repeats, seed):
    groups = [rows[int(index)]["video_id"] for index in indices]
    metrics["miou_95ci"] = clustered_bootstrap(
        metrics["per_window_iou"], groups, repeats, seed
    )
    per_class = defaultdict(list)
    for index, iou in zip(indices, metrics["per_window_iou"]):
        per_class[rows[int(index)]["gt_label"]].append(iou)
    metrics["per_class_miou"] = {
        label: float(np.mean(values)) for label, values in sorted(per_class.items())
    }
    metrics["class_macro_miou"] = float(
        np.mean(list(metrics["per_class_miou"].values()))
    )


def write_predictions(path, rows, indices, metrics, condition):
    payload = []
    for index, prediction, iou in zip(
        indices, metrics["predictions"], metrics["per_window_iou"]
    ):
        row = rows[int(index)]
        payload.append(
            {
                "candidate_id": row["candidate_id"],
                "instance_id": row["instance_id"],
                "video_id": row["video_id"],
                "gt_label": row["gt_label"],
                "condition": condition,
                "prediction_start_sec": prediction[0],
                "prediction_end_sec": prediction[1],
                "gt_start_sec": float(row["gt_start_sec"]),
                "gt_end_sec": float(row["gt_end_sec"]),
                "tiou": float(iou),
            }
        )
    write_jsonl_atomic(path, payload)


def build_donor_indices(indices, labels, video_ids, seed, strata=None):
    indices = np.asarray(indices, dtype=np.int64)
    class_groups = defaultdict(list)
    stratum_groups = defaultdict(list)
    for index in indices:
        stratum = None if strata is None else int(strata[index])
        label = int(labels[index])
        class_groups[label].append(int(index))
        stratum_groups[(label, stratum)].append(int(index))
    donor = {}
    rng = random.Random(seed)
    candidate_cache = {}
    for row_index in indices:
        label = int(labels[row_index])
        stratum = None if strata is None else int(strata[row_index])
        video_id = str(video_ids[row_index])
        cache_key = (label, stratum, video_id)
        candidates = candidate_cache.get(cache_key)
        if candidates is None:
            preferred = stratum_groups[(label, stratum)]
            candidates = [
                index for index in preferred if str(video_ids[index]) != video_id
            ]
            if not candidates:
                candidates = [
                    index
                    for index in class_groups[label]
                    if str(video_ids[index]) != video_id
                ]
            candidate_cache[cache_key] = candidates
        if not candidates:
            raise ValueError(
                "shuffle control lacks a cross-video donor for class {}".format(
                    label
                )
            )
        donor[int(row_index)] = int(rng.choice(candidates))
    return donor


def donor_diagnostics(donor, lengths):
    sources = list(donor)
    targets = [donor[index] for index in sources]
    return {
        "rows": len(sources),
        "unique_donor_ratio": len(set(targets)) / len(targets),
        "same_length_ratio": float(
            np.mean([lengths[source] == lengths[target] for source, target in donor.items()])
        ),
        "normalized_time_resample_ratio": float(
            np.mean([lengths[source] != lengths[target] for source, target in donor.items()])
        ),
    }


class ProbeRuntime:
    def __init__(self, features, timestamps, lengths, offsets, rows, args):
        import torch

        self.torch = torch
        self.features = features
        self.timestamps = timestamps
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.rows = rows
        self.starts = np.full(len(rows), -1, dtype=np.int64)
        self.ends = np.full(len(rows), -1, dtype=np.int64)
        self.args = args
        self.labels = np.asarray([row["gt_label_id"] for row in rows], dtype=np.int64)
        self.video_ids = np.asarray([row["video_id"] for row in rows])
        self.dimension = int(features.shape[1])
        self.time_steps = int(timestamps.shape[1])

    def ensure_targets(self, indices):
        missing = np.asarray(
            [int(index) for index in indices if self.starts[int(index)] < 0],
            dtype=np.int64,
        )
        if len(missing) == 0:
            return
        starts, ends = quantized_targets(
            self.rows, self.timestamps, self.lengths, missing
        )
        self.starts[missing] = starts
        self.ends[missing] = ends

    def new_model(self, seed):
        torch = self.torch
        torch.manual_seed(seed)

        class TemporalProbe(torch.nn.Module):
            def __init__(self, classes, dimension):
                super().__init__()
                self.start_weight = torch.nn.Parameter(torch.empty(classes, dimension))
                self.start_bias = torch.nn.Parameter(torch.zeros(classes))
                self.end_weight = torch.nn.Parameter(torch.empty(classes, dimension))
                self.end_bias = torch.nn.Parameter(torch.zeros(classes))
                torch.nn.init.normal_(self.start_weight, std=0.01)
                torch.nn.init.normal_(self.end_weight, std=0.01)

            def forward(self, features, labels):
                start_weights = self.start_weight[labels]
                end_weights = self.end_weight[labels]
                start = torch.einsum("btd,bd->bt", features, start_weights)
                end = torch.einsum("btd,bd->bt", features, end_weights)
                return start + self.start_bias[labels, None], end + self.end_bias[labels, None]

        return TemporalProbe(len(LABELS), self.dimension).to(self.args.device)

    def feature_batch(self, row_indices_value, donor=None):
        feature_indices = [
            donor.get(int(index), int(index)) if donor else int(index)
            for index in row_indices_value
        ]
        recipient_lengths = self.lengths[np.asarray(row_indices_value, dtype=np.int64)]
        donor_lengths = self.lengths[np.asarray(feature_indices, dtype=np.int64)]
        max_length = int(recipient_lengths.max())
        array = np.zeros(
            (len(feature_indices), max_length, self.dimension), dtype=np.float32
        )
        feature_indices = np.asarray(feature_indices, dtype=np.int64)
        length_pairs = np.stack((recipient_lengths, donor_lengths), axis=1)
        for recipient_length, donor_length in np.unique(length_pairs, axis=0):
            positions = np.flatnonzero(
                (recipient_lengths == recipient_length)
                & (donor_lengths == donor_length)
            )
            donor_steps = np.rint(
                np.linspace(0, int(donor_length) - 1, int(recipient_length))
            ).astype(np.int64)
            token_indices = (
                self.offsets[feature_indices[positions], None]
                + donor_steps[None, :]
            )
            array[positions, : int(recipient_length)] = self.features[token_indices]
        tensor = self.torch.from_numpy(array).to(self.args.device, non_blocking=True)
        tensor = tensor / tensor.norm(dim=2, keepdim=True).clamp_min(1e-12)
        return tensor, recipient_lengths

    def mask_logits(self, logits, lengths):
        torch = self.torch
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.args.device)
        invalid = torch.arange(logits.shape[1], device=self.args.device)[None, :] >= lengths[:, None]
        return logits.masked_fill(invalid, torch.finfo(logits.dtype).min)

    def balanced_batches(self, indices, seed):
        counts = Counter(int(self.labels[index]) for index in indices)
        weights = np.asarray(
            [1.0 / counts[int(self.labels[index])] for index in indices],
            dtype=np.float64,
        )
        weights /= weights.sum()
        rng = np.random.default_rng(seed)
        sampled = rng.choice(indices, len(indices), replace=True, p=weights)
        batches = []
        for length in np.unique(self.lengths[sampled]):
            bucket = sampled[self.lengths[sampled] == length]
            rng.shuffle(bucket)
            batches.extend(
                bucket[start : start + self.args.batch_size]
                for start in range(0, len(bucket), self.args.batch_size)
            )
        rng.shuffle(batches)
        return batches

    def train_epoch(self, model, optimizer, indices, epoch_seed, donor=None):
        torch = self.torch
        self.ensure_targets(indices)
        model.train()
        losses = []
        for batch_indices in self.balanced_batches(indices, epoch_seed):
            features, lengths = self.feature_batch(batch_indices, donor=donor)
            labels = torch.as_tensor(
                self.labels[batch_indices], dtype=torch.long, device=self.args.device
            )
            start_targets = torch.as_tensor(
                self.starts[batch_indices], dtype=torch.long, device=self.args.device
            )
            end_targets = torch.as_tensor(
                self.ends[batch_indices], dtype=torch.long, device=self.args.device
            )
            start_logits, end_logits = model(features, labels)
            start_logits = self.mask_logits(start_logits, lengths)
            end_logits = self.mask_logits(end_logits, lengths)
            loss = (
                torch.nn.functional.cross_entropy(start_logits, start_targets)
                + torch.nn.functional.cross_entropy(end_logits, end_targets)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.mean(losses))

    def probabilities(self, model, indices, donor=None):
        torch = self.torch
        model.eval()
        start_probabilities = np.zeros(
            (len(indices), self.time_steps), dtype=np.float32
        )
        end_probabilities = np.zeros_like(start_probabilities)
        with torch.inference_mode():
            eval_lengths = self.lengths[indices]
            for length in np.unique(eval_lengths):
                positions = np.flatnonzero(eval_lengths == length)
                for start in range(0, len(positions), self.args.eval_batch_size):
                    batch_positions = positions[start : start + self.args.eval_batch_size]
                    batch_indices = indices[batch_positions]
                    features, lengths = self.feature_batch(batch_indices, donor=donor)
                    labels = torch.as_tensor(
                        self.labels[batch_indices],
                        dtype=torch.long,
                        device=self.args.device,
                    )
                    start_logits, end_logits = model(features, labels)
                    start_logits = self.mask_logits(start_logits, lengths)
                    end_logits = self.mask_logits(end_logits, lengths)
                    width = start_logits.shape[1]
                    start_probabilities[batch_positions, :width] = (
                        torch.softmax(start_logits.float(), dim=1).cpu().numpy()
                    )
                    end_probabilities[batch_positions, :width] = (
                        torch.softmax(end_logits.float(), dim=1).cpu().numpy()
                    )
        return start_probabilities, end_probabilities

    def miou(self, model, indices, donor=None):
        start_probs, end_probs = self.probabilities(model, indices, donor=donor)
        metrics = interval_metrics(
            self.rows,
            indices,
            self.timestamps,
            self.lengths,
            start_probs,
            end_probs,
        )
        return metrics["miou"]

    def fit_with_early_stop(
        self,
        train_indices,
        dev_indices,
        learning_rate,
        weight_decay,
        seed,
        train_donor=None,
        dev_donor=None,
    ):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        model = self.new_model(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        best = {"miou": -1.0, "epoch": 0, "loss": None, "state": None}
        stale = 0
        for epoch in range(1, self.args.max_epochs + 1):
            loss = self.train_epoch(
                model,
                optimizer,
                train_indices,
                seed * 1000 + epoch,
                donor=train_donor,
            )
            dev_miou = self.miou(model, dev_indices, donor=dev_donor)
            if dev_miou > best["miou"] + 1e-8:
                best = {
                    "miou": dev_miou,
                    "epoch": epoch,
                    "loss": loss,
                    "state": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.args.patience:
                break
        model.load_state_dict(best.pop("state"))
        return model, best

    def fit_fixed_epochs(
        self, train_indices, learning_rate, weight_decay, epochs, seed, donor=None
    ):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        model = self.new_model(seed)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        losses = []
        for epoch in range(1, epochs + 1):
            losses.append(
                self.train_epoch(
                    model,
                    optimizer,
                    train_indices,
                    seed * 1000 + epoch,
                    donor=donor,
                )
            )
        return model, losses


def grouped_folds(indices, video_ids, folds):
    from sklearn.model_selection import GroupKFold

    indices = np.asarray(indices, dtype=np.int64)
    splitter = GroupKFold(n_splits=folds)
    return [
        (indices[train], indices[dev])
        for train, dev in splitter.split(indices, groups=video_ids[indices])
    ]


def evaluate_nonvisual_cv(rows, train_indices, video_ids, folds, seed):
    fold_results = []
    for fold_index, (fold_train, fold_dev) in enumerate(
        grouped_folds(train_indices, video_ids, folds)
    ):
        fold_results.append(
            nonvisual_fold_scores(
                rows, fold_train, fold_dev, seed=seed + fold_index
            )
        )
    names = sorted(fold_results[0])
    return {
        name: {
            "fold_miou": [result[name] for result in fold_results],
            "mean_miou": float(np.mean([result[name] for result in fold_results])),
        }
        for name in names
    }


def select_configuration(runtime, train_indices, args, shift):
    folds = grouped_folds(train_indices, runtime.video_ids, args.folds)
    grid = []
    for learning_rate in args.learning_rates:
        for weight_decay in args.weight_decays:
            fold_results = []
            for fold_index, (fold_train, fold_dev) in enumerate(folds):
                seed = args.cv_seed + fold_index
                _, result = runtime.fit_with_early_stop(
                    fold_train,
                    fold_dev,
                    learning_rate,
                    weight_decay,
                    seed,
                )
                fold_results.append(result)
                print(
                    "{} lr={} wd={} fold={} mIoU={:.4f} epoch={}".format(
                        shift,
                        learning_rate,
                        weight_decay,
                        fold_index + 1,
                        result["miou"],
                        result["epoch"],
                    ),
                    flush=True,
                )
            grid.append(
                {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "mean_miou": float(
                        np.mean([result["miou"] for result in fold_results])
                    ),
                    "folds": fold_results,
                }
            )
    selected = max(
        grid,
        key=lambda row: (
            row["mean_miou"],
            -row["weight_decay"],
            -row["learning_rate"],
        ),
    )
    selected_epochs = max(
        1, int(np.median([row["epoch"] for row in selected["folds"]]))
    )
    return grid, selected, selected_epochs


def evaluate_shuffle_cv(runtime, train_indices, selected, args, shift):
    folds = grouped_folds(train_indices, runtime.video_ids, args.folds)
    results = []
    for fold_index, (fold_train, fold_dev) in enumerate(folds):
        seed = args.shuffle_seed + fold_index
        train_donor = build_donor_indices(
            fold_train,
            runtime.labels,
            runtime.video_ids,
            seed,
            strata=runtime.lengths,
        )
        dev_donor = build_donor_indices(
            fold_dev,
            runtime.labels,
            runtime.video_ids,
            seed + 1000,
            strata=runtime.lengths,
        )
        _, result = runtime.fit_with_early_stop(
            fold_train,
            fold_dev,
            selected["learning_rate"],
            selected["weight_decay"],
            seed,
            train_donor=train_donor,
            dev_donor=dev_donor,
        )
        result["train_donor_diagnostics"] = donor_diagnostics(
            train_donor, runtime.lengths
        )
        result["dev_donor_diagnostics"] = donor_diagnostics(
            dev_donor, runtime.lengths
        )
        results.append(result)
        print(
            "{} shuffle fold={} mIoU={:.4f} epoch={}".format(
                shift, fold_index + 1, result["miou"], result["epoch"]
            ),
            flush=True,
        )
    return {
        "folds": results,
        "mean_miou": float(np.mean([result["miou"] for result in results])),
    }


def ensemble_evaluate(runtime, models, indices, donors=None):
    start_sum = np.zeros((len(indices), runtime.time_steps), dtype=np.float64)
    end_sum = np.zeros_like(start_sum)
    for model_index, model in enumerate(models):
        donor = donors[model_index] if donors else None
        start_probs, end_probs = runtime.probabilities(model, indices, donor=donor)
        start_sum += start_probs
        end_sum += end_probs
    return interval_metrics(
        runtime.rows,
        indices,
        runtime.timestamps,
        runtime.lengths,
        (start_sum / len(models)).astype(np.float32),
        (end_sum / len(models)).astype(np.float32),
    )


def strip_heavy_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in ("per_window_iou", "predictions")
    }


def run_shift(runtime, rows, shift, args, output_dir):
    train_indices = role_indices(rows, shift, "source_train")
    source_indices = role_indices(rows, shift, "source_val")
    target_indices = role_indices(rows, shift, "target_val")
    if min(len(train_indices), len(source_indices), len(target_indices)) == 0:
        raise ValueError("{} has an empty required role".format(shift))
    role_videos = {
        "source_train": set(runtime.video_ids[train_indices]),
        "source_val": set(runtime.video_ids[source_indices]),
        "target_val": set(runtime.video_ids[target_indices]),
    }
    for left, right in (
        ("source_train", "source_val"),
        ("source_train", "target_val"),
        ("source_val", "target_val"),
    ):
        overlap = role_videos[left].intersection(role_videos[right])
        if overlap:
            raise ValueError(
                "{} {}/{} video leakage: {}".format(
                    shift, left, right, sorted(overlap)[0]
                )
            )

    nonvisual_cv = evaluate_nonvisual_cv(
        rows,
        train_indices,
        runtime.video_ids,
        args.folds,
        args.cv_seed,
    )
    grid, selected, epochs = select_configuration(runtime, train_indices, args, shift)
    shuffle_cv = evaluate_shuffle_cv(runtime, train_indices, selected, args, shift)
    control_scores = {
        "feature_shuffle": shuffle_cv["mean_miou"],
        **{
            "nonvisual_{}".format(name): result["mean_miou"]
            for name, result in nonvisual_cv.items()
        },
    }
    strongest_control = max(control_scores, key=control_scores.get)
    shortcut_ratio = (
        control_scores[strongest_control] / selected["mean_miou"]
        if selected["mean_miou"] > 0
        else float("inf")
    )
    if shortcut_ratio >= args.shortcut_threshold:
        result = {
            "shift": shift,
            "status": "shortcut_alarm",
            "counts": {
                "source_train": len(train_indices),
                "source_val": len(source_indices),
                "target_val": len(target_indices),
            },
            "selection": {
                "metric": "source_train_grouped_cv_miou",
                "folds": args.folds,
                "grid": grid,
                "selected_learning_rate": selected["learning_rate"],
                "selected_weight_decay": selected["weight_decay"],
                "selected_mean_cv_miou": selected["mean_miou"],
                "final_epochs": epochs,
            },
            "shortcut_gate": {
                "threshold": args.shortcut_threshold,
                "nonvisual_cv": nonvisual_cv,
                "shuffle_cv": shuffle_cv,
                "strongest_control": strongest_control,
                "strongest_control_miou": control_scores[strongest_control],
                "strongest_control_to_real_ratio": shortcut_ratio,
                "source_val_boundary_labels_read_before_gate": False,
                "target_val_boundary_labels_read_before_gate": False,
            },
        }
        write_json_atomic(output_dir / "{}.json".format(shift), result)
        return result

    best_nonvisual = max(
        nonvisual_cv, key=lambda name: nonvisual_cv[name]["mean_miou"]
    )
    source_prior_predictions = nonvisual_predictions(
        rows, train_indices, source_indices, args.cv_seed
    )
    target_prior_predictions = nonvisual_predictions(
        rows, train_indices, target_indices, args.cv_seed
    )
    prior_metrics = {}
    prior_window_ious = {}
    for prior_offset, name in enumerate(sorted(source_prior_predictions)):
        source_prior = metrics_from_predictions(
            rows,
            source_indices,
            runtime.timestamps,
            runtime.lengths,
            source_prior_predictions[name],
        )
        target_prior = metrics_from_predictions(
            rows,
            target_indices,
            runtime.timestamps,
            runtime.lengths,
            target_prior_predictions[name],
        )
        add_distribution_metrics(
            source_prior,
            rows,
            source_indices,
            args.bootstrap_repeats,
            args.bootstrap_seed + 10 + prior_offset * 2,
        )
        add_distribution_metrics(
            target_prior,
            rows,
            target_indices,
            args.bootstrap_repeats,
            args.bootstrap_seed + 11 + prior_offset * 2,
        )
        write_predictions(
            output_dir / "{}_source_prior_{}.jsonl".format(shift, name),
            rows,
            source_indices,
            source_prior,
            "source_prior_{}".format(name),
        )
        write_predictions(
            output_dir / "{}_target_prior_{}.jsonl".format(shift, name),
            rows,
            target_indices,
            target_prior,
            "target_prior_{}".format(name),
        )
        prior_metrics[name] = {
            "source": strip_heavy_metrics(source_prior),
            "target": strip_heavy_metrics(target_prior),
        }
        prior_window_ious[name] = {
            "source": source_prior["per_window_iou"],
            "target": target_prior["per_window_iou"],
        }
    models = []
    shuffle_models = []
    source_donors = []
    target_donors = []
    shuffle_final_diagnostics = []
    for seed in args.seeds:
        model, losses = runtime.fit_fixed_epochs(
            train_indices,
            selected["learning_rate"],
            selected["weight_decay"],
            epochs,
            seed,
        )
        models.append(model)
        model_path = output_dir / "{}_seed{}.pt".format(shift, seed)
        runtime.torch.save(
            {
                "protocol": PROTOCOL,
                "shift": shift,
                "seed": seed,
                "learning_rate": selected["learning_rate"],
                "weight_decay": selected["weight_decay"],
                "epochs": epochs,
                "trainable_parameters": sorted(model.state_dict()),
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "losses": losses,
            },
            model_path,
        )

        train_donor = build_donor_indices(
            train_indices,
            runtime.labels,
            runtime.video_ids,
            args.shuffle_seed + seed,
            strata=runtime.lengths,
        )
        shuffle_model, _ = runtime.fit_fixed_epochs(
            train_indices,
            selected["learning_rate"],
            selected["weight_decay"],
            epochs,
            seed,
            donor=train_donor,
        )
        shuffle_models.append(shuffle_model)
        source_donor = build_donor_indices(
            source_indices,
            runtime.labels,
            runtime.video_ids,
            args.shuffle_seed + 1000 + seed,
            strata=runtime.lengths,
        )
        target_donor = build_donor_indices(
            target_indices,
            runtime.labels,
            runtime.video_ids,
            args.shuffle_seed + 2000 + seed,
            strata=runtime.lengths,
        )
        source_donors.append(source_donor)
        target_donors.append(target_donor)
        shuffle_final_diagnostics.append(
            {
                "seed": seed,
                "train": donor_diagnostics(train_donor, runtime.lengths),
                "source": donor_diagnostics(source_donor, runtime.lengths),
                "target": donor_diagnostics(target_donor, runtime.lengths),
            }
        )

    source_metrics = ensemble_evaluate(runtime, models, source_indices)
    target_metrics = ensemble_evaluate(runtime, models, target_indices)
    source_shuffle = ensemble_evaluate(
        runtime, shuffle_models, source_indices, donors=source_donors
    )
    target_shuffle = ensemble_evaluate(
        runtime, shuffle_models, target_indices, donors=target_donors
    )
    for offset, (metrics, indices) in enumerate(
        (
            (source_metrics, source_indices),
            (target_metrics, target_indices),
            (source_shuffle, source_indices),
            (target_shuffle, target_indices),
        )
    ):
        add_distribution_metrics(
            metrics,
            rows,
            indices,
            args.bootstrap_repeats,
            args.bootstrap_seed + offset,
        )
    prediction_sets = (
        ("source_real", source_indices, source_metrics),
        ("target_real", target_indices, target_metrics),
        ("source_shuffle", source_indices, source_shuffle),
        ("target_shuffle", target_indices, target_shuffle),
    )
    for condition, indices, metrics in prediction_sets:
        write_predictions(
            output_dir / "{}_{}.jsonl".format(shift, condition),
            rows,
            indices,
            metrics,
            condition,
        )
    source_groups = [rows[int(index)]["video_id"] for index in source_indices]
    target_groups = [rows[int(index)]["video_id"] for index in target_indices]
    source_gap = np.asarray(source_metrics["per_window_iou"]) - np.asarray(
        source_shuffle["per_window_iou"]
    )
    target_gap = np.asarray(target_metrics["per_window_iou"]) - np.asarray(
        target_shuffle["per_window_iou"]
    )
    source_prior_gap = np.asarray(source_metrics["per_window_iou"]) - np.asarray(
        prior_window_ious[best_nonvisual]["source"]
    )
    target_prior_gap = np.asarray(target_metrics["per_window_iou"]) - np.asarray(
        prior_window_ious[best_nonvisual]["target"]
    )
    source_miou = source_metrics["miou"]
    target_miou = target_metrics["miou"]
    result = {
        "shift": shift,
        "status": "complete",
        "counts": {
            "source_train": len(train_indices),
            "source_val": len(source_indices),
            "target_val": len(target_indices),
        },
        "selection": {
            "metric": "source_train_grouped_cv_miou",
            "folds": args.folds,
            "grid": grid,
            "selected_learning_rate": selected["learning_rate"],
            "selected_weight_decay": selected["weight_decay"],
            "selected_mean_cv_miou": selected["mean_miou"],
            "final_epochs": epochs,
        },
        "shortcut_gate": {
            "threshold": args.shortcut_threshold,
            "nonvisual_cv": nonvisual_cv,
            "shuffle_cv": shuffle_cv,
            "strongest_control": strongest_control,
            "strongest_control_miou": control_scores[strongest_control],
            "strongest_control_to_real_ratio": shortcut_ratio,
            "source_val_boundary_labels_read_before_gate": False,
            "target_val_boundary_labels_read_before_gate": False,
            "final_evaluation_labels_read_after_freeze": True,
        },
        "nonvisual_priors": {
            "selected_on_source_train_cv": best_nonvisual,
            "results": prior_metrics,
        },
        "source": strip_heavy_metrics(source_metrics),
        "target": strip_heavy_metrics(target_metrics),
        "source_shuffle": strip_heavy_metrics(source_shuffle),
        "target_shuffle": strip_heavy_metrics(target_shuffle),
        "shuffle_final_donor_diagnostics": shuffle_final_diagnostics,
        "cross_domain": {
            "delta_miou": target_miou - source_miou,
            "retention": target_miou / source_miou if source_miou > 0 else None,
            "source_shuffle_gap": source_miou - source_shuffle["miou"],
            "target_shuffle_gap": target_miou - target_shuffle["miou"],
            "source_prior_adjusted_gain": source_miou
            - prior_metrics[best_nonvisual]["source"]["miou"],
            "target_prior_adjusted_gain": target_miou
            - prior_metrics[best_nonvisual]["target"]["miou"],
            "source_prior_adjusted_gain_95ci": clustered_bootstrap(
                source_prior_gap,
                source_groups,
                args.bootstrap_repeats,
                args.bootstrap_seed + 20,
            ),
            "target_prior_adjusted_gain_95ci": clustered_bootstrap(
                target_prior_gap,
                target_groups,
                args.bootstrap_repeats,
                args.bootstrap_seed + 21,
            ),
            "source_shuffle_gap_95ci": clustered_bootstrap(
                source_gap, source_groups, args.bootstrap_repeats, args.bootstrap_seed
            ),
            "target_shuffle_gap_95ci": clustered_bootstrap(
                target_gap,
                target_groups,
                args.bootstrap_repeats,
                args.bootstrap_seed + 1,
            ),
        },
    }
    write_json_atomic(output_dir / "{}.json".format(shift), result)
    return result


def parse_csv(value, cast):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("value list cannot be empty")
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-index", action="append", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shift", action="append", choices=SHIFTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--learning-rates", default="1e-4,3e-4,1e-3")
    parser.add_argument("--weight-decays", default="0,1e-4,1e-3,1e-2")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--cv-seed", type=int, default=20260919)
    parser.add_argument("--shuffle-seed", type=int, default=20260920)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260919)
    parser.add_argument("--shortcut-threshold", type=float, default=0.7)
    parser.add_argument("--feature-bank-workers", type=int, default=16)
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    return parser.parse_args()


def runtime_environment(runtime, device):
    import sklearn

    torch = runtime.torch
    device_name = None
    if str(device).startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.device(device))
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "device": device,
        "device_name": device_name,
        "probe_code_sha256": sha256_file(__file__),
    }


def aggregate_complete_results(results):
    complete = [result for result in results if result["status"] == "complete"]
    if not complete:
        return None
    fields = {
        "source_miou": lambda row: row["source"]["miou"],
        "target_miou": lambda row: row["target"]["miou"],
        "delta_miou": lambda row: row["cross_domain"]["delta_miou"],
        "retention": lambda row: row["cross_domain"]["retention"],
        "target_prior_adjusted_gain": lambda row: row["cross_domain"][
            "target_prior_adjusted_gain"
        ],
        "target_shuffle_gap": lambda row: row["cross_domain"][
            "target_shuffle_gap"
        ],
        "target_consistent_and_correct_at_0.5": lambda row: row["target"][
            "consistent_and_correct_at_0.5"
        ],
        "target_class_macro_miou": lambda row: row["target"]["class_macro_miou"],
    }
    aggregate = {
        "shifts": [row["shift"] for row in complete],
        "shift_count": len(complete),
    }
    for name, getter in fields.items():
        values = np.asarray([getter(row) for row in complete], dtype=np.float64)
        aggregate[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "worst": float(values.min()),
        }
    return aggregate


def main():
    args = parse_args()
    args.learning_rates = parse_csv(args.learning_rates, float)
    args.weight_decays = parse_csv(args.weight_decays, float)
    args.seeds = parse_csv(args.seeds, int)
    if args.folds != 3:
        raise ValueError("formal protocol requires exactly three grouped folds")
    if args.seeds != [0, 1, 2, 3, 4]:
        raise ValueError("formal protocol requires seeds 0,1,2,3,4")
    if args.learning_rates != [1e-4, 3e-4, 1e-3]:
        raise ValueError("formal protocol learning-rate grid is frozen")
    if args.weight_decays != [0.0, 1e-4, 1e-3, 1e-2]:
        raise ValueError("formal protocol weight-decay grid is frozen")
    if args.max_epochs != 50 or args.patience != 5:
        raise ValueError("formal protocol requires 50 max epochs and patience 5")
    if args.batch_size != 256 or args.eval_batch_size != 512:
        raise ValueError("formal protocol batch sizes are frozen at 256/512")
    if args.cv_seed != 20260919 or args.shuffle_seed != 20260920:
        raise ValueError("formal protocol CV and shuffle seeds are frozen")
    if args.bootstrap_repeats != 1000 or args.bootstrap_seed != 20260919:
        raise ValueError("formal protocol bootstrap configuration is frozen")
    if args.shortcut_threshold != 0.7:
        raise ValueError("formal protocol shortcut threshold is frozen at 0.7")
    if args.feature_bank_workers <= 0:
        raise ValueError("feature bank workers must be positive")

    rows = read_jsonl(args.manifest)
    features, timestamps, lengths, offsets, feature_metadata = prepare_feature_bank(
        rows,
        args.manifest,
        args.feature_index,
        args.feature_root,
        args.feature_cache_dir,
        rebuild=args.rebuild_feature_cache,
        loader_workers=args.feature_bank_workers,
    )
    runtime = ProbeRuntime(features, timestamps, lengths, offsets, rows, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = runtime_environment(runtime, args.device)
    frozen_config = {
        "protocol": PROTOCOL,
        "manifest_sha256": feature_metadata["manifest_sha256"],
        "feature_metadata": feature_metadata,
        "shifts": args.shift or list(SHIFTS),
        "learning_rates": args.learning_rates,
        "weight_decays": args.weight_decays,
        "folds": args.folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "seeds": args.seeds,
        "cv_seed": args.cv_seed,
        "shuffle_seed": args.shuffle_seed,
        "bootstrap_repeats": args.bootstrap_repeats,
        "bootstrap_seed": args.bootstrap_seed,
        "shortcut_threshold": args.shortcut_threshold,
        "feature_bank_workers": args.feature_bank_workers,
        "runtime_environment": environment,
    }
    write_json_atomic(output_dir / "frozen_protocol_config.json", frozen_config)
    shifts = args.shift or list(SHIFTS)
    results = []
    for shift in shifts:
        result = run_shift(runtime, rows, shift, args, output_dir)
        results.append(result)
        if result["status"] == "shortcut_alarm":
            print(
                "{} stopped before source-val/target-val: shortcut ratio {:.4f} >= {:.4f}".format(
                    shift,
                    result["shortcut_gate"]["strongest_control_to_real_ratio"],
                    args.shortcut_threshold,
                ),
                flush=True,
            )
            break
        print(
            "{} source_mIoU={:.4f} target_mIoU={:.4f} retention={:.4f}".format(
                shift,
                result["source"]["miou"],
                result["target"]["miou"],
                result["cross_domain"]["retention"],
            ),
            flush=True,
        )
    payload = {
        "protocol": PROTOCOL,
        "evidence_status": (
            "formal_complete"
            if all(row["status"] == "complete" for row in results)
            else "stopped_shortcut_alarm"
        ),
        "feature_metadata": feature_metadata,
        "manifest": str(Path(args.manifest).resolve()),
        "feature_indexes": [str(Path(path).resolve()) for path in args.feature_index],
        "runtime_environment": environment,
        "splits": {
            "training_and_selection": "source_train only",
            "final_evaluation": ["source_val", "target_val"],
            "target_used_for_training_or_selection": False,
        },
        "probe": {
            "head": "class-conditional linear start/end",
            "optimizer": "AdamW",
            "learning_rates": args.learning_rates,
            "weight_decays": args.weight_decays,
            "folds": args.folds,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "cv_seed": args.cv_seed,
            "shuffle_seed": args.shuffle_seed,
            "bootstrap_repeats": args.bootstrap_repeats,
            "bootstrap_seed": args.bootstrap_seed,
            "seeds": args.seeds,
            "trainable_parameters": [
                "start_weight",
                "start_bias",
                "end_weight",
                "end_bias",
            ],
        },
        "results": results,
        "completed_shift_macro": aggregate_complete_results(results),
    }
    write_json_atomic(output_dir / "rq2_temporal_probe_results.json", payload)
    return 3 if any(row["status"] == "shortcut_alarm" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
