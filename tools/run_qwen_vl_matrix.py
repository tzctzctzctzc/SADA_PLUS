"""Run one local Qwen classifier across the six SADA shifts with one load."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tools import run_qwen3_vl_classification as base


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def protocol(model_family):
    if model_family == "qwen3-8b":
        version = "qwen3-vl-8b-ordered-images-digit-logits-v1"
        wire_api = "qwen3-vl-next-token-classification"
    elif model_family == "qwen3.8-27b":
        version = "qwen3.8-27b-ordered-images-digit-logits-v1"
        wire_api = "qwen3.8-vl-next-token-classification"
    else:
        raise ValueError("unknown model family: {}".format(model_family))
    prompt_sha = hashlib.sha256(
        json.dumps(
            {
                "version": version,
                "labels": base.LABELS,
                "phrases": base.ACTION_PHRASES,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return version, wire_api, prompt_sha


def load_runtime(args):
    if args.model_family == "qwen3-8b":
        return base.load_qwen3(args)
    from tools.run_qwen38_vl_classification import load_qwen38

    return load_qwen38(args)


def shift_args(args, shift):
    root = Path(args.scenario_root) / shift
    values = vars(args).copy()
    values.update({
        "source_manifest": str(root / "source_val_manifest.jsonl"),
        "target_manifest": str(root / "target_train_manifest.jsonl"),
        "source_frame_cache": str(Path(args.frame_cache) / "source"),
        "target_frame_cache": str(Path(args.frame_cache) / "target"),
        "source_cache_namespace": "{}_source".format(shift.lower()),
        "target_cache_namespace": shift.lower(),
        "source_output": str(Path(args.output_root) / shift / "source_predictions.jsonl"),
        "target_output": str(Path(args.output_root) / shift / "target_predictions.jsonl"),
        "source_gate_output": str(Path(args.output_root) / shift / "source_gate.json"),
    })
    return SimpleNamespace(**values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--model-family", choices=("qwen3-8b", "qwen3.8-27b"), required=True
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device-map", choices=("none", "auto"), default="none")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=224 * 224)
    parser.add_argument("--max-pixels", type=int, default=448 * 448)
    parser.add_argument("--score-temperature", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force-target", action="store_true")
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    parser.add_argument("--min-consensus-coverage", type=float, default=0.20)
    parser.add_argument("--min-precision-gain", type=float, default=0.03)
    parser.add_argument("--min-background-precision", type=float, default=0.80)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--min-candidate-mass", type=float, default=0.50)
    parser.add_argument("--max-prediction-share", type=float, default=0.60)
    parser.add_argument(
        "--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES)
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0 or args.min_pixels <= 0 or args.max_pixels < args.min_pixels:
        raise ValueError("invalid frame or pixel limits")
    version, wire_api, prompt_sha = protocol(args.model_family)
    base.PROMPT_VERSION = version
    base.PROMPT_SHA256 = prompt_sha
    base.WIRE_API = wire_api
    runtime = load_runtime(args)
    for shift in args.shifts:
        current = shift_args(args, shift)
        Path(current.output_root).joinpath(shift).mkdir(parents=True, exist_ok=True)
        base.score_manifest(
            runtime,
            current.source_manifest,
            current.source_output,
            current.source_frame_cache,
            current.source_cache_namespace,
            current,
        )
        gate = base.source_gate(
            current.source_manifest,
            current.source_output,
            current.source_gate_output,
            current,
        )
        print("{} source gate={}".format(shift, gate["adapter_pass"]), flush=True)
        if not gate["adapter_pass"] and not current.force_target:
            print("{} target skipped by source gate".format(shift), flush=True)
            continue
        base.score_manifest(
            runtime,
            current.target_manifest,
            current.target_output,
            current.target_frame_cache,
            current.target_cache_namespace,
            current,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
