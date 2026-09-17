"""Extract resumable MOSS-VL segment features for the RQ1 probe."""

import argparse
import hashlib
import tempfile
import time
from pathlib import Path

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
    parser.add_argument("--model-path", default="OpenMOSS-Team/MOSS-VL-Instruct-0708")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--vision-chunked-length", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def select_shard(indices, shard_count, shard_index):
    return [index for index in indices if index % shard_count == shard_index]


def load_moss_vl(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    local_only = bool(args.offline)
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_only,
        frame_extract_num_threads=1,
    )
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=local_only,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return {
        "torch": torch,
        "processor": processor,
        "model": model,
        "device": device,
    }


def write_frame_video(frame_files, output_path, fps=1):
    import av
    from PIL import Image

    images = []
    for path in frame_files:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert("RGB")).copy())
    height, width = images[0].shape[:2]
    if any(image.shape[:2] != (height, width) for image in images):
        raise ValueError("all cached segment frames must have the same size")
    with av.open(str(output_path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for image in images:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def encode_segment(runtime, rows, index, args):
    paths = frame_paths(
        args.frame_cache,
        args.cache_namespace,
        rows[index]["candidate_id"],
        frame_count=args.frame_count,
    )
    with tempfile.TemporaryDirectory(prefix="rq1_moss_vl_") as directory:
        video_path = Path(directory) / "segment.mp4"
        write_frame_video(paths, video_path, fps=1)
        query = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path)},
                    {"type": "text", "text": MULTIMODAL_PROBE_PROMPT},
                ],
            }],
            "media_kwargs": {
                "min_pixels": int(args.min_pixels),
                "max_pixels": int(args.max_pixels),
                "video_max_pixels": int(args.frame_count * args.max_pixels),
                "video_fps": 1.0,
                "min_frames": int(args.frame_count),
                "max_frames": int(args.frame_count),
            },
        }
        prepared = runtime["model"].offline_prepare_query_cpu(
            runtime["processor"], query, padding=False
        )
    inputs = runtime["model"]._offline_move_inputs_to_devices(
        prepared["inputs_cpu"]
    )
    with runtime["torch"].inference_mode():
        output = runtime["model"](
            **inputs,
            use_cache=False,
            logits_to_keep=1,
            vision_chunked_length=args.vision_chunked_length,
            output_hidden_states=True,
            return_dict=True,
        )
    if not output.hidden_states:
        raise RuntimeError("MOSS-VL did not return multimodal hidden states")
    attention_mask = inputs["attention_mask"].bool()
    positions = runtime["torch"].arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0).expand_as(attention_mask)
    last_position = positions.masked_fill(~attention_mask, -1).max(dim=1).values
    feature = output.hidden_states[-1][0, last_position[0]].detach().float()
    feature = feature / feature.norm().clamp_min(1e-12)
    return feature.cpu().numpy()[None, :]


def main():
    args = parse_args()
    if args.frame_count <= 1:
        raise ValueError("frame_count must exceed one")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard index must be within a positive shard count")
    rows = read_jsonl(args.manifest)
    runtime = load_moss_vl(args)
    dimensions = int(runtime["model"].config.text_config.hidden_size)
    identity = (
        "{}|interface=multimodal_last_token|prompt_sha256={}"
        "|attention=sdpa|min_pixels={}|max_pixels={}|video_fps=1"
    ).format(
        args.model_path,
        PROMPT_SHA256,
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
    for processed, index in enumerate(pending, start=1):
        store.write([index], encode_segment(runtime, rows, index, args))
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
