"""Extract resumable VideoChat3 segment features for the RQ1 probe."""

import argparse
import hashlib
import time

import numpy as np

from tools.rq1_feature_store import FeatureStore
from tools.run_clip_target_adapter import read_jsonl
from tools.run_clip_then_llava import frame_paths


MULTIMODAL_PROBE_PROMPT = (
    "The images above are chronological frames sampled uniformly from one action segment. "
    "Focus on the dominant hand-object action and prepare a concise action representation."
)
PROMPT_SHA256 = hashlib.sha256(
    MULTIMODAL_PROBE_PROMPT.encode("utf-8")
).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_8f")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model-path", default="MCG-NJU/VideoChat3-4B")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--vision-attention",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--segment-batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def select_shard(indices, shard_count, shard_index):
    return [index for index in indices if index % shard_count == shard_index]


def load_videochat3(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    local_only = bool(args.offline)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    # Keep the published FlashAttention-2 path when available while retaining
    # the model's official SDPA/eager implementations as explicit fallbacks.
    config.vision_config.attn_impl = args.vision_attention
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="sdpa",
        output_loading_info=bool(getattr(args, 'pre_cuda_guard', None)),
    )
    if getattr(args, 'pre_cuda_guard', None):
        model, info = model
        if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('RQ3 official VideoChat3 load incomplete: ' + str(info))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, 'pre_cuda_guard', None):
        args.pre_cuda_guard()
    model.to(device).eval()
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
    }


def load_video(frame_files):
    from PIL import Image

    frames = []
    for path in frame_files:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert("RGB")).copy())
    return np.stack(frames)


def video_metadata(video):
    frames, height, width = video.shape[:3]
    return {
        "total_num_frames": int(frames),
        "fps": 1.0,
        "duration": float(frames),
        "width": int(width),
        "height": int(height),
    }


def encode_batch(runtime, rows, indices, args):
    videos = []
    messages = []
    metadata = []
    for index in indices:
        paths = frame_paths(
            args.frame_cache,
            args.cache_namespace,
            rows[index]["candidate_id"],
            frame_count=args.frame_count,
        )
        video = load_video(paths)
        videos.append(video)
        metadata.append(video_metadata(video))
        messages.append([{
            "role": "user",
            "content": [
                {"type": "video", "video": "placeholder.mp4"},
                {"type": "text", "text": MULTIMODAL_PROBE_PROMPT},
            ],
        }])

    processor = runtime["processor"]
    texts = [
        processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        for message in messages
    ]
    inputs = processor(
        text=texts,
        videos=videos,
        video_metadata=metadata,
        num_frames=args.frame_count,
        fps=None,
        do_resize=True,
        size={
            "shortest_edge": int(args.min_pixels),
            "longest_edge": int(args.max_pixels),
        },
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
        raise RuntimeError("VideoChat3 did not return multimodal hidden states")
    hidden = output.hidden_states[-1]
    attention_mask = inputs["attention_mask"].bool()
    positions = runtime["torch"].arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0).expand_as(attention_mask)
    last_positions = positions.masked_fill(~attention_mask, -1).max(dim=1).values
    batch_indices = runtime["torch"].arange(
        len(indices), device=hidden.device
    )
    features = hidden[batch_indices, last_positions].detach().float()
    features = features / features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return features.cpu().numpy()


def main():
    args = parse_args()
    if args.frame_count <= 1 or args.segment_batch_size <= 0:
        raise ValueError("frame_count must exceed one and batch size must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be within a positive shard count")
    rows = read_jsonl(args.manifest)
    runtime = load_videochat3(args)
    dimensions = int(runtime["model"].config.text_config.hidden_size)
    identity = (
        "{}|interface=multimodal_last_token|prompt_sha256={}"
        "|vision_attention={}|min_pixels={}|max_pixels={}"
    ).format(
        args.model_path,
        PROMPT_SHA256,
        args.vision_attention,
        args.min_pixels,
        args.max_pixels,
    )
    store = FeatureStore(
        args.output_prefix,
        args.manifest,
        rows,
        identity,
        dimensions,
        args.frame_count,
        "multimodal_last_token",
    )
    pending = select_shard(
        store.pending_indices(), args.shard_count, args.shard_index
    )
    if args.limit > 0:
        pending = pending[:args.limit]
    started = time.time()
    processed = 0
    for offset in range(0, len(pending), args.segment_batch_size):
        indices = pending[offset:offset + args.segment_batch_size]
        store.write(indices, encode_batch(runtime, rows, indices, args))
        processed += len(indices)
        elapsed = max(time.time() - started, 1e-9)
        print(
            "{}/{} segments {:.3f} segments/s".format(
                processed, len(pending), processed / elapsed
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
