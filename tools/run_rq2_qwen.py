"""Predict action start/end gaps from 16 ordered context frames with Qwen3-VL."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from tools import run_qwen3_vl_classification as qwen_base
from tools.run_clip_target_adapter import ACTION_PHRASES, LABELS, read_jsonl, shard_for_video
from tools.run_clip_then_llava import frame_paths
from tools.run_qwen_vl_matrix import load_runtime


PROMPT_VERSION = "rq2-oracle-class-boundary-gaps-16f-v1"


def build_prompt(label, frame_count=16):
    phrase = ACTION_PHRASES[LABELS.index(label)]
    return (
        "The images are {} frames sampled uniformly in chronological order from a context "
        "window that contains the known action {} ({}). There are {} temporal gaps: gap 0 "
        "is before frame 1, gap k is between frames k and k+1, and gap {} is after frame {}. "
        "Locate the visible action interval. Return exactly START,END using integer gap indices "
        "with 0 <= START < END <= {}. Do not add any other text."
    ).format(frame_count, label, phrase, frame_count + 1, frame_count, frame_count, frame_count)


def prompt_hash(label, frame_count):
    return hashlib.sha256(build_prompt(label, frame_count).encode("utf-8")).hexdigest()


def parse_interval(text, frame_count=16):
    match = re.search(r"(?<!\d)(\d{1,2})\s*[,;]\s*(\d{1,2})(?!\d)", text)
    if not match:
        return None
    start, end = map(int, match.groups())
    return (start, end) if 0 <= start < end <= frame_count else None


def current(row, manifest_row, args):
    return bool(
        row.get("candidate_id") == manifest_row["candidate_id"]
        and row.get("status") in ("ok", "invalid")
        and row.get("model") == args.model_path
        and row.get("model_revision")
        and row.get("frame_count") == args.frame_count
        and row.get("prompt_version") == PROMPT_VERSION
        and row.get("prompt_sha256") == prompt_hash(manifest_row["gt_label"], args.frame_count)
    )


def infer_one(runtime, paths, prompt, args):
    messages = qwen_base.build_messages(paths, prompt, args.min_pixels, args.max_pixels)
    text = runtime["processor"].apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = runtime["process_vision_info"](
        messages, image_patch_size=16
    )
    inputs = runtime["processor"](
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(runtime["device"])
    with runtime["torch"].inference_mode():
        generated = runtime["model"].generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    suffix = generated[:, inputs.input_ids.shape[1]:]
    return runtime["processor"].batch_decode(
        suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq2_boundary_16f")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-family", choices=("qwen3-8b", "qwen3-32b"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto", "balanced"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    if args.frame_count <= 1 or args.max_new_tokens <= 0:
        raise ValueError("invalid frame or generation count")
    rows = [
        row for row in read_jsonl(args.manifest)
        if shard_for_video(row["video_id"], args.shard_count) == args.shard_index
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output) if output.is_file() else []
    by_id = {row.get("candidate_id"): row for row in prior}
    pending = [row for row in rows if not current(by_id.get(row["candidate_id"], {}), row, args)]
    if args.limit > 0:
        pending = pending[:args.limit]
    runtime = load_runtime(args)
    for index, row in enumerate(pending, 1):
        prompt = build_prompt(row["gt_label"], args.frame_count)
        try:
            paths = frame_paths(
                args.frame_cache, args.cache_namespace, row["candidate_id"], args.frame_count
            )
            response = infer_one(runtime, paths, prompt, args)
            interval = parse_interval(response, args.frame_count)
            result = {
                "candidate_id": row["candidate_id"],
                "video_id": row["video_id"],
                "status": "ok" if interval else "invalid",
                "model": args.model_path,
                "model_revision": runtime["model_revision"],
                "wire_api": "qwen3-vl-greedy-boundary-generation",
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash(row["gt_label"], args.frame_count),
                "frame_count": args.frame_count,
                "raw_response": response,
                "pred_start_position": interval[0] if interval else None,
                "pred_end_position": interval[1] if interval else None,
            }
        except Exception as error:
            result = {
                "candidate_id": row["candidate_id"], "video_id": row["video_id"],
                "status": "error", "error": "{}: {}".format(type(error).__name__, error),
            }
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, sort_keys=True) + "\n")
        print("{}/{} {} {}".format(index, len(pending), row["candidate_id"], result["status"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
