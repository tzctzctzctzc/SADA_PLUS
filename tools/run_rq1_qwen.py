"""Run one Qwen3-VL model on the shared RQ1 GT-segment frame cache."""

import argparse
from pathlib import Path

from tools import run_qwen3_vl_classification as base
from tools.run_qwen_vl_matrix import load_runtime, protocol


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_8f")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-family", choices=("qwen3-8b", "qwen3-32b"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto", "balanced"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument(
        "--prompt-mode",
        choices=("baseline", "phase_transition", "contrastive_transition"),
        default="baseline",
    )
    parser.add_argument("--frame-order", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0 or args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid frame or pixel limits")
    args.input_mode = "images"
    args.sample_fps = 1.0
    args.sample_fps_mode = "fixed"
    version, wire_api, prompt_hash = protocol(
        args.model_family,
        args.frame_count,
        args.prompt_mode,
        args.frame_order,
    )
    base.PROMPT_VERSION = version
    base.PROMPT_SHA256 = prompt_hash
    base.WIRE_API = wire_api
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    runtime = load_runtime(args)
    base.score_manifest(
        runtime,
        args.manifest,
        args.output,
        args.frame_cache,
        args.cache_namespace,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
