"""Classify proposal frames with the local Qwen3.8-27B vision-language model."""

import hashlib
import json
import os

from tools import run_qwen3_vl_classification as base


PROMPT_VERSION = "qwen3.8-27b-ordered-images-digit-logits-v1"
WIRE_API = "qwen3.8-vl-next-token-classification"
PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {
            "version": PROMPT_VERSION,
            "labels": base.LABELS,
            "phrases": base.ACTION_PHRASES,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()


def load_qwen38(args):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model_kwargs = {
        "dtype": dtype,
        "attn_implementation": args.attn_implementation,
        "local_files_only": args.offline,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path, **model_kwargs
    )
    if args.device_map == "none":
        model.to("cuda")
    model.eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=args.offline
    )
    return {
        "torch": torch,
        "model": model,
        "processor": processor,
        "process_vision_info": process_vision_info,
        "digit_token_ids": base.resolve_digit_token_ids(processor.tokenizer),
        "device": model.device,
        "model_revision": getattr(model.config, "_commit_hash", None),
    }


def main():
    args = base.parse_args()
    base.PROMPT_VERSION = PROMPT_VERSION
    base.PROMPT_SHA256 = PROMPT_SHA256
    base.WIRE_API = WIRE_API
    runtime = load_qwen38(args)
    base.score_manifest(
        runtime,
        args.source_manifest,
        args.source_output,
        args.source_frame_cache,
        args.source_cache_namespace,
        args,
    )
    gate = base.source_gate(
        args.source_manifest,
        args.source_output,
        args.source_gate_output,
        args,
    )
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    if not gate["adapter_pass"] and not args.force_target:
        print("source gate 未通过，跳过 target；如需强制运行请加 --force-target", flush=True)
        return 0
    base.score_manifest(
        runtime,
        args.target_manifest,
        args.target_output,
        args.target_frame_cache,
        args.target_cache_namespace,
        args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
