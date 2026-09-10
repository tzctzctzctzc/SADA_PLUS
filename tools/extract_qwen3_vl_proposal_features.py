"""Extract prompt-free Qwen3-VL visual features for proposal frame sequences."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.run_clip_then_llava import frame_paths, read_jsonl
from tools.run_qwen3_vl_classification import (
    build_messages,
    load_qwen3,
    ordered_frame_paths,
)


FEATURE_SCHEMA = "qwen3-vl-proposal-visual-features-v1"
NEUTRAL_PROMPT = "Observe the ordered proposal frames."


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_token_groups(groups):
    """Mean-pool variable-length visual token groups into frame features."""
    pooled = []
    for group in groups:
        array = group.detach().float().cpu().numpy()
        if array.ndim != 2 or array.shape[0] == 0:
            raise ValueError("visual token group must have shape [tokens, dim]")
        pooled.append(array.mean(axis=0))
    return np.stack(pooled).astype(np.float16)


def split_deepstack_features(features, split_sizes):
    layers = []
    for feature in features:
        groups = feature.split(tuple(split_sizes), dim=0)
        layers.append(mean_token_groups(groups))
    if not layers:
        return np.empty((0, len(split_sizes), 0), dtype=np.float16)
    return np.stack(layers).astype(np.float16)


def extract_one(runtime, paths, args):
    messages = build_messages(
        paths, NEUTRAL_PROMPT, args.min_pixels, args.max_pixels
    )
    text = runtime["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = runtime["process_vision_info"](
        messages, image_patch_size=16
    )
    inputs = runtime["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime["device"])
    torch = runtime["torch"]
    with torch.inference_mode():
        image_embeds, deepstack = runtime["model"].get_image_features(
            inputs.pixel_values, inputs.image_grid_thw
        )
    split_sizes = (
        inputs.image_grid_thw.prod(-1)
        // runtime["model"].model.visual.spatial_merge_size ** 2
    ).tolist()
    if len(image_embeds) != len(paths) or len(split_sizes) != len(paths):
        raise ValueError("Qwen image feature count differs from proposal frame count")
    return {
        "final_frames": mean_token_groups(image_embeds),
        "deepstack_frames": split_deepstack_features(deepstack, split_sizes),
    }


def current_rows(index_path, feature_root, args, manifest_sha256):
    completed = set()
    for row in read_jsonl(index_path):
        feature_file = feature_root / row.get("feature_file", "")
        if (
            row.get("status") == "ok"
            and row.get("schema_version") == FEATURE_SCHEMA
            and row.get("model") == args.model_path
            and row.get("manifest_sha256") == manifest_sha256
            and row.get("frame_count") == args.frame_count
            and feature_file.is_file()
        ):
            completed.add(row["candidate_id"])
    return completed


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


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
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 1:
        raise ValueError("frame_count must be greater than one")
    manifest_path = Path(args.manifest)
    manifest = read_jsonl(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    index_path = Path(args.output_index)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    feature_root = Path(args.feature_root)
    feature_root.mkdir(parents=True, exist_ok=True)
    completed = current_rows(index_path, feature_root, args, manifest_sha256)
    pending = [row for row in manifest if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    runtime = load_qwen3(args)
    for number, row in enumerate(pending, 1):
        candidate_id = row["candidate_id"]
        try:
            paths = ordered_frame_paths(
                frame_paths(
                    args.frame_cache,
                    args.cache_namespace,
                    candidate_id,
                    frame_count=args.frame_count,
                )
            )
            arrays = extract_one(runtime, paths, args)
            feature_name = candidate_id + ".npz"
            atomic_savez(feature_root / feature_name, **arrays)
            result = {
                "candidate_id": candidate_id,
                "video_id": row.get("video_id"),
                "status": "ok",
                "schema_version": FEATURE_SCHEMA,
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "manifest_sha256": manifest_sha256,
                "frame_count": args.frame_count,
                "feature_file": feature_name,
                "final_dim": int(arrays["final_frames"].shape[-1]),
                "deepstack_layers": int(arrays["deepstack_frames"].shape[0]),
            }
        except Exception as error:
            result = {
                "candidate_id": candidate_id,
                "video_id": row.get("video_id"),
                "status": "error",
                "schema_version": FEATURE_SCHEMA,
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


if __name__ == "__main__":
    main()
