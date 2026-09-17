"""Extract resumable Qwen3-VL segment features for RQ1 probes."""

import argparse
import hashlib
import time

import numpy as np

from tools.rq1_feature_store import FeatureStore, summarize_frame_features
from tools.run_clip_target_adapter import read_jsonl
from tools.run_clip_then_llava import frame_paths
from tools.run_qwen3_vl_classification import build_messages, load_qwen3


MULTIMODAL_PROBE_PROMPT = (
    "The images above are chronological frames sampled uniformly from one action segment. "
    "Focus on the dominant hand-object action and prepare a concise action representation."
)
MULTIMODAL_PROBE_PROMPT_SHA256 = hashlib.sha256(
    MULTIMODAL_PROBE_PROMPT.encode("utf-8")
).hexdigest()


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
    parser.add_argument(
        "--feature-interface",
        choices=("multimodal_last_token", "vision_tower"),
        default="multimodal_last_token",
    )
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


def encode_multimodal(runtime, rows, index, args):
    paths = frame_paths(
        args.frame_cache,
        args.cache_namespace,
        rows[index]["candidate_id"],
        frame_count=args.frame_count,
    )
    messages = build_messages(
        paths,
        MULTIMODAL_PROBE_PROMPT,
        args.min_pixels,
        args.max_pixels,
    )
    text = runtime["processor"].apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = runtime["process_vision_info"](
        messages,
        image_patch_size=16,
    )
    inputs = runtime["processor"](
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(runtime["device"])
    with runtime["torch"].inference_mode():
        output = runtime["model"](
            **inputs,
            use_cache=False,
            logits_to_keep=1,
            output_hidden_states=True,
            return_dict=True,
        )
    if not output.hidden_states:
        raise RuntimeError("Qwen did not return multimodal hidden states")
    feature = output.hidden_states[-1][0, -1].detach().float()
    feature = feature / feature.norm().clamp_min(1e-12)
    return feature.cpu().numpy()


def main():
    args = parse_args()
    if args.frame_count <= 1 or args.segment_batch_size <= 0:
        raise ValueError("frame_count must exceed one and batch size must be positive")
    rows = read_jsonl(args.manifest)
    runtime = load_qwen3(args)
    dimensions = int(runtime["model"].config.text_config.hidden_size)
    if args.feature_interface == "vision_tower" and args.pooling == "static_transition":
        dimensions *= 2
    pooling = (
        args.pooling
        if args.feature_interface == "vision_tower"
        else "multimodal_last_token"
    )
    model_identity = "{}|interface={}".format(
        args.model_path, args.feature_interface
    )
    if args.feature_interface == "multimodal_last_token":
        model_identity += "|prompt_sha256={}".format(
            MULTIMODAL_PROBE_PROMPT_SHA256
        )
    store = FeatureStore(
        args.output_prefix, args.manifest, rows, model_identity, dimensions,
        args.frame_count, pooling,
    )
    pending = store.pending_indices()
    if args.limit > 0:
        pending = pending[:args.limit]
    started = time.time()
    processed = 0
    step = args.segment_batch_size if args.feature_interface == "vision_tower" else 1
    for offset in range(0, len(pending), step):
        indices = pending[offset:offset + step]
        if args.feature_interface == "vision_tower":
            features = encode_batch(runtime, rows, indices, args)
        else:
            features = np.stack([
                encode_multimodal(runtime, rows, index, args)
                for index in indices
            ])
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
