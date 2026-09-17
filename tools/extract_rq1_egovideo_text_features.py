"""Extract several auditable text-prompt banks from the released EgoVideo encoder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.extract_rq1_egovideo_features import (
    MODEL_NAME,
    checkpoint_sha256,
    load_model,
)
from tools.run_clip_target_adapter import ACTION_PHRASES, LABELS


PROMPT_PRESETS = {
    "bare_label": tuple(label.lower().replace("_", " ") for label in LABELS),
    "bare_phrase": ACTION_PHRASES,
}


def parse_output_spec(spec):
    if "=" not in spec:
        raise ValueError("prompt output must be PRESET=PATH")
    preset, path = spec.split("=", 1)
    if preset not in PROMPT_PRESETS or not path:
        raise ValueError("unknown or empty prompt output: {}".format(spec))
    return preset, Path(path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egohod-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--prompt-output", action="append", required=True,
        help="PRESET=PATH; presets: {}".format(", ".join(PROMPT_PRESETS)),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    import clip
    import torch

    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    model = load_model(args.egohod_root, args.checkpoint, args.device)
    for spec in args.prompt_output:
        preset, output = parse_output_spec(spec)
        prompts = PROMPT_PRESETS[preset]
        tokens = clip.tokenize(list(prompts)).to(args.device)
        with torch.inference_mode():
            encoded = model.encode_text(tokens).float()
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, encoded.cpu().numpy().astype(np.float32))
        prompt_hash = hashlib.sha256(json.dumps({
            "labels": LABELS,
            "prompts": prompts,
        }, sort_keys=True).encode()).hexdigest()
        Path(str(output) + ".meta.json").write_text(json.dumps({
            "model": MODEL_NAME,
            "checkpoint_sha256": checkpoint_hash,
            "prompt_preset": preset,
            "prompt_sha256": prompt_hash,
            "labels": list(LABELS),
            "prompts": list(prompts),
            "logit_scale": float(model.logit_scale.exp().detach().cpu()),
        }, indent=2, sort_keys=True), encoding="utf-8")
        print("{} -> {}".format(preset, output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
