"""Extract auditable time-aligned Qwen3-VL vision features for RQ2 windows."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from tools.run_clip_then_llava import frame_paths
from tools.run_clip_target_adapter import read_jsonl
from tools.run_qwen3_vl_classification import load_qwen3


SCHEMA = "rq2-qwen-time-aligned-vision-tower-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shard_for_video(video_id, shard_count):
    value = hashlib.sha256(str(video_id).encode("utf-8")).digest()
    return int.from_bytes(value[:8], "big") % shard_count


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def completed_ids(index_path, feature_root, args, manifest_hash):
    completed = set()
    if not Path(index_path).is_file():
        return completed
    for row in read_jsonl(index_path):
        feature_file = feature_root / row.get("feature_file", "")
        if (
            row.get("status") == "ok"
            and row.get("schema_version") == SCHEMA
            and row.get("model") == args.model_path
            and row.get("manifest_sha256") == manifest_hash
            and feature_file.is_file()
        ):
            completed.add(row["candidate_id"])
    return completed


def load_rgb(path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def frame_key(row, timestamp):
    return row["video_id"], round(float(timestamp), 6)


def frames_ready(row, args):
    return all(
        path.is_file()
        for path in frame_paths(
            args.frame_cache,
            args.cache_namespace,
            row["candidate_id"],
            int(row["frame_count"]),
        )
    )


def pack_rows(rows, max_batch_frames):
    batch = []
    keys = set()
    for row in rows:
        row_keys = {
            frame_key(row, timestamp) for timestamp in row["frame_timestamps_sec"]
        }
        added = len(row_keys.difference(keys))
        if batch and len(keys) + added > max_batch_frames:
            yield batch
            batch = []
            keys = set()
        batch.append(row)
        keys.update(row_keys)
    if batch:
        yield batch


def encode_rows(runtime, rows, args, executor):
    unique_paths = []
    key_to_index = {}
    row_indices = []
    for row in rows:
        paths = frame_paths(
            args.frame_cache,
            args.cache_namespace,
            row["candidate_id"],
            int(row["frame_count"]),
        )
        indices = []
        for timestamp, path in zip(row["frame_timestamps_sec"], paths):
            key = frame_key(row, timestamp)
            if key not in key_to_index:
                key_to_index[key] = len(unique_paths)
                unique_paths.append(path)
            indices.append(key_to_index[key])
        row_indices.append(indices)
    images = list(executor.map(load_rgb, unique_paths))
    inputs = runtime["processor"].image_processor(
        images=images,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    image_grid_thw = inputs["image_grid_thw"]
    if str(runtime["device"]).startswith("cuda"):
        pixel_values = pixel_values.pin_memory()
        image_grid_thw = image_grid_thw.pin_memory()
    pixel_values = pixel_values.to(runtime["device"], non_blocking=True)
    image_grid_thw = image_grid_thw.to(runtime["device"], non_blocking=True)
    with runtime["torch"].inference_mode():
        image_embeds, _ = runtime["model"].get_image_features(
            pixel_values, image_grid_thw
        )
    if len(image_embeds) != len(unique_paths):
        raise ValueError("Qwen feature count differs from frame count")
    features = runtime["torch"].stack(
        [embedding.float().mean(dim=0) for embedding in image_embeds]
    )
    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    features = features.cpu().numpy().astype(np.float16)
    return [features[np.asarray(indices)] for indices in row_indices], len(unique_paths)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-batch-frames", type=int, default=64)
    parser.add_argument("--loader-workers", type=int, default=8)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="process only windows whose complete frame cache already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be within a positive shard count")
    if args.max_batch_frames < 64 or args.loader_workers <= 0:
        raise ValueError("max-batch-frames must be at least 64 and loader-workers positive")
    manifest_path = Path(args.manifest)
    rows = [
        row for row in read_jsonl(manifest_path)
        if shard_for_video(row["video_id"], args.shard_count) == args.shard_index
    ]
    if not rows:
        raise ValueError("manifest is empty")
    manifest_hash = sha256_file(manifest_path)
    index_path = Path(args.output_index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(index_path, feature_root, args, manifest_hash)
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if args.available_only:
        before = len(pending)
        pending = [row for row in pending if frames_ready(row, args)]
        print(
            "available_only selected={}/{} shard={}/{}".format(
                len(pending), before, args.shard_index, args.shard_count
            ),
            flush=True,
        )
    if args.limit > 0:
        pending = pending[: args.limit]
    if not pending:
        print("no pending windows", flush=True)
        return 0
    runtime = load_qwen3(args)
    runtime["torch"].set_float32_matmul_precision("high")
    runtime["torch"].backends.cudnn.benchmark = True
    started = time.time()
    processed = 0
    encoded_frames = 0
    batches = list(pack_rows(pending, args.max_batch_frames))
    with ThreadPoolExecutor(max_workers=args.loader_workers) as executor:
        for batch_number, batch in enumerate(batches, 1):
            try:
                batch_features, unique_frames = encode_rows(
                    runtime, batch, args, executor
                )
                results = []
                for row, features in zip(batch, batch_features):
                    timestamps = np.asarray(
                        row["frame_timestamps_sec"], dtype=np.float32
                    )
                    if features.shape[0] != timestamps.shape[0]:
                        raise ValueError("feature/timestamp length mismatch")
                    feature_name = row["candidate_id"] + ".npz"
                    atomic_savez(
                        feature_root / feature_name,
                        features=features,
                        timestamps_sec=timestamps,
                    )
                    results.append({
                        "candidate_id": row["candidate_id"],
                        "instance_id": row["instance_id"],
                        "video_id": row["video_id"],
                        "status": "ok",
                        "schema_version": SCHEMA,
                        "model": args.model_path,
                        "model_revision": runtime["model_revision"],
                        "manifest_sha256": manifest_hash,
                        "feature_file": feature_name,
                        "frame_count": int(features.shape[0]),
                        "feature_dim": int(features.shape[1]),
                        "time_mapping": "one pooled vision embedding per sampled frame",
                        "extractor": "deduplicated-batched-v1",
                    })
            except Exception as error:
                unique_frames = 0
                results = [{
                    "candidate_id": row["candidate_id"],
                    "video_id": row.get("video_id"),
                    "status": "error",
                    "schema_version": SCHEMA,
                    "error": "{}: {}".format(type(error).__name__, error),
                } for row in batch]
            with index_path.open("a", encoding="utf-8") as stream:
                for result in results:
                    stream.write(json.dumps(result, sort_keys=True) + "\n")
            processed += len(batch)
            encoded_frames += unique_frames
            elapsed = max(time.time() - started, 1e-9)
            print(
                "batch={}/{} windows={}/{} unique_frames={} windows_s={:.3f} frames_s={:.1f}".format(
                    batch_number,
                    len(batches),
                    processed,
                    len(pending),
                    unique_frames,
                    processed / elapsed,
                    encoded_frames / elapsed,
                ),
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
