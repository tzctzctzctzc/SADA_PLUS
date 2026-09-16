"""Extract resumable prompt-free Qwen3-VL segment features for RQ1 probes."""

import argparse
import time

import numpy as np

from tools.rq1_feature_store import FeatureStore, summarize_frame_features
from tools.run_clip_target_adapter import read_jsonl
from tools.run_clip_then_llava import frame_paths
from tools.run_qwen3_vl_classification import load_qwen3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_probe_8f")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model-path", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--segment-batch-size", type=int, default=4)
    parser.add_argument("--pooling", choices=("static", "static_transition"), default="static")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def encode_batch(runtime, rows, indices, args):
    from PIL import Image

    images = []
    for index in indices:
        paths = frame_paths(
            args.frame_cache, args.cache_namespace, rows[index]["candidate_id"],
            frame_count=args.frame_count,
        )
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
    frame_features = np.stack([
        embedding.detach().float().mean(dim=0).cpu().numpy()
        for embedding in image_embeds
    ])
    if len(frame_features) != len(indices) * args.frame_count:
        raise ValueError("Qwen image feature count differs from requested frame count")
    return np.stack([
        summarize_frame_features(
            frame_features[offset:offset + args.frame_count], args.pooling
        )
        for offset in range(0, len(frame_features), args.frame_count)
    ])


def main():
    args = parse_args()
    if args.frame_count <= 1 or args.segment_batch_size <= 0:
        raise ValueError("frame_count must exceed one and batch size must be positive")
    rows = read_jsonl(args.manifest)
    runtime = load_qwen3(args)
    dimensions = int(runtime["model"].config.text_config.hidden_size)
    if args.pooling == "static_transition":
        dimensions *= 2
    store = FeatureStore(
        args.output_prefix, args.manifest, rows, args.model_path, dimensions,
        args.frame_count, args.pooling,
    )
    pending = store.pending_indices()
    if args.limit > 0:
        pending = pending[:args.limit]
    started = time.time()
    processed = 0
    for offset in range(0, len(pending), args.segment_batch_size):
        indices = pending[offset:offset + args.segment_batch_size]
        features = encode_batch(runtime, rows, indices, args)
        store.write(indices, features)
        processed += len(indices)
        elapsed = max(time.time() - started, 1e-9)
        print(
            "{}/{} segments {:.3f} segments/s".format(
                processed, len(pending), processed / elapsed
            ), flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
