"""Extract auditable time-aligned Qwen3-VL vision features for RQ2 windows."""

import argparse
import hashlib
import json
import os
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


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def completed_ids(index_path, feature_root, args, manifest_hash):
    completed = set()
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


def encode_window(runtime, paths, args):
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    inputs = runtime["processor"].image_processor(
        images=images,
        min_pixels=int(args.min_pixels),
        max_pixels=int(args.max_pixels),
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"].to(runtime["device"])
    image_grid_thw = inputs["image_grid_thw"].to(runtime["device"])
    with runtime["torch"].inference_mode():
        image_embeds, _ = runtime["model"].get_image_features(
            pixel_values, image_grid_thw
        )
    if len(image_embeds) != len(paths):
        raise ValueError("Qwen feature count differs from frame count")
    features = runtime["torch"].stack(
        [embedding.float().mean(dim=0) for embedding in image_embeds]
    )
    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy().astype(np.float16)


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
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = Path(args.manifest)
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("manifest is empty")
    manifest_hash = sha256_file(manifest_path)
    index_path = Path(args.output_index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)
    completed = completed_ids(index_path, feature_root, args, manifest_hash)
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    runtime = load_qwen3(args)
    for number, row in enumerate(pending, 1):
        candidate_id = row["candidate_id"]
        try:
            count = int(row["frame_count"])
            paths = frame_paths(
                args.frame_cache, args.cache_namespace, candidate_id, count
            )
            features = encode_window(runtime, paths, args)
            timestamps = np.asarray(row["frame_timestamps_sec"], dtype=np.float32)
            if features.shape[0] != timestamps.shape[0]:
                raise ValueError("feature/timestamp length mismatch")
            feature_name = candidate_id + ".npz"
            atomic_savez(
                feature_root / feature_name,
                features=features,
                timestamps_sec=timestamps,
            )
            result = {
                "candidate_id": candidate_id,
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
            }
        except Exception as error:
            result = {
                "candidate_id": candidate_id,
                "video_id": row.get("video_id"),
                "status": "error",
                "schema_version": SCHEMA,
                "error": "{}: {}".format(type(error).__name__, error),
            }
        with index_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print(
            "{}/{} {} {}".format(
                number, len(pending), candidate_id, result["status"]
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
