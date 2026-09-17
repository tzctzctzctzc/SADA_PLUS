"""Extract joint eight-frame EgoVideo-L embeddings for the RQ1 protocol."""

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

from tools.rq1_feature_store import FeatureStore
from tools.run_clip_target_adapter import (
    ACTION_PHRASES,
    LABELS,
    PROMPT_SHA256,
    PROMPT_TEMPLATES,
    PROMPT_VERSION,
    read_jsonl,
)
from tools.run_clip_then_llava import frame_paths


MODEL_NAME = "EgoVideo-L-HOD-large_best"
MODEL_DIMENSIONS = 512
NATIVE_FRAME_COUNT = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_8f")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--egohod-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--text-features-output", default=None)
    return parser.parse_args()


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_sdpa_compat(transformer_module):
    """Match flash-attn parameter names while using PyTorch fused SDPA."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class FlashMHACompat(nn.Module):
        def __init__(self, d_model, n_head, cross_attn=False,
                     qkv_proj_bias=True, out_proj_bias=True, dropout=0.0,
                     use_flash_attn=True):
            super().__init__()
            if cross_attn:
                raise ValueError("cross attention is not supported")
            self.n_head = int(n_head)
            self.head_dim = int(d_model) // self.n_head
            self.dropout = float(dropout)
            self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=qkv_proj_bias)
            self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)

        def forward(self, x):
            batch, tokens, dimensions = x.shape
            qkv = self.Wqkv(x).reshape(
                batch, tokens, 3, self.n_head, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            values = functional.scaled_dot_product_attention(
                qkv[0], qkv[1], qkv[2],
                dropout_p=self.dropout if self.training else 0.0,
            )
            values = values.transpose(1, 2).reshape(batch, tokens, dimensions)
            return self.out_proj(values)

    class FlashMlpCompat(nn.Module):
        def __init__(self, in_features, hidden_features, activation):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.activation = activation
            self.fc2 = nn.Linear(hidden_features, in_features)

        def forward(self, x):
            return self.fc2(self.activation(self.fc1(x)))

    transformer_module.FlashMHA = FlashMHACompat
    transformer_module.FlashMlp = FlashMlpCompat


def install_mmengine_weight_init_compat():
    """Provide the two init helpers used by EgoHOD without importing mmengine."""
    import torch.nn as nn

    weight_init = types.ModuleType("mmengine.model.weight_init")

    def constant_init(module, val, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def trunc_normal_init(module, mean=0, std=1, a=-2, b=2, bias=0):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.trunc_normal_(module.weight, mean=mean, std=std, a=a, b=b)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    weight_init.constant_init = constant_init
    weight_init.trunc_normal_init = trunc_normal_init
    model_module = types.ModuleType("mmengine.model")
    model_module.weight_init = weight_init
    mmengine_module = types.ModuleType("mmengine")
    mmengine_module.model = model_module
    sys.modules["mmengine"] = mmengine_module
    sys.modules["mmengine.model"] = model_module
    sys.modules["mmengine.model.weight_init"] = weight_init


def load_model(egohod_root, checkpoint, device):
    import torch

    source_root = str(Path(egohod_root).resolve())
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    install_mmengine_weight_init_compat()
    import model.transformer as transformer_module

    install_sdpa_compat(transformer_module)
    from model.clip import CLIP

    visual = transformer_module.VisionTransformer(
        336, 14, 1024, 24, 16, 4,
        output_dim=MODEL_DIMENSIONS,
        num_frames=NATIVE_FRAME_COUNT,
        use_fast_conv1=True,
        use_flash_attn=True,
    )
    textual = transformer_module.TextTransformer(
        context_length=77,
        vocab_size=49408,
        width=768,
        heads=12,
        layers=12,
        output_dim=MODEL_DIMENSIONS,
        causal_mask=True,
    )
    model = CLIP(
        embed_dim=MODEL_DIMENSIONS,
        vision_model=visual,
        text_model=textual,
        freeze_temperature=True,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("module."): value
        for key, value in payload["state_dict"].items()
    }
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device).half()
    return model


def image_transform():
    from torchvision.transforms import (
        CenterCrop,
        Compose,
        InterpolationMode,
        Normalize,
        Resize,
        ToTensor,
    )

    return Compose([
        Resize(336, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(336),
        ToTensor(),
        Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def save_text_features(model, output, checkpoint_hash, device):
    import clip
    import torch

    prompts = [
        template.format(phrase)
        for phrase in ACTION_PHRASES
        for template in PROMPT_TEMPLATES
    ]
    tokens = clip.tokenize(prompts).to(device)
    with torch.inference_mode():
        encoded = model.encode_text(tokens).float()
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        encoded = encoded.reshape(len(LABELS), len(PROMPT_TEMPLATES), -1).mean(1)
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, encoded.cpu().numpy().astype(np.float32))
    Path(str(output) + ".meta.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "checkpoint_sha256": checkpoint_hash,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "labels": list(LABELS),
        "phrases": list(ACTION_PHRASES),
        "templates": list(PROMPT_TEMPLATES),
        "logit_scale": float(model.logit_scale.exp().detach().cpu()),
    }, indent=2, sort_keys=True), encoding="utf-8")


def main():
    args = parse_args()
    if args.frame_count <= 0 or args.batch_size <= 0:
        raise ValueError("frame count and batch size must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")

    rows = read_jsonl(args.manifest)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    store = FeatureStore(
        args.output_prefix,
        args.manifest,
        rows,
        "{}@sha256:{}".format(MODEL_NAME, checkpoint_hash),
        MODEL_DIMENSIONS,
        args.frame_count,
        "joint_video_cls",
    )
    if args.initialize_only:
        print("initialized {} rows".format(len(rows)), flush=True)
        return 0

    import torch
    from PIL import Image

    model = load_model(args.egohod_root, args.checkpoint, args.device)
    if args.text_features_output:
        save_text_features(
            model, args.text_features_output, checkpoint_hash, args.device
        )
    transform = image_transform()
    pending = [
        index for index in store.pending_indices()
        if index % args.shard_count == args.shard_index
    ]
    if args.limit > 0:
        pending = pending[:args.limit]
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        indices = pending[offset:offset + args.batch_size]
        videos = []
        for index in indices:
            frames = []
            for path in frame_paths(
                args.frame_cache,
                args.cache_namespace,
                rows[index]["candidate_id"],
                args.frame_count,
            ):
                with Image.open(path) as image:
                    frames.append(transform(image.convert("RGB")))
            videos.append(torch.stack(frames, dim=1))
        batch = torch.stack(videos).to(args.device, non_blocking=True)
        with torch.inference_mode():
            features = model.encode_image(batch)[0].float()
            features = features / features.norm(dim=-1, keepdim=True)
        store.write(indices, features.cpu().numpy())
        processed = min(offset + len(indices), len(pending))
        elapsed = max(time.time() - started, 1e-9)
        print(
            "shard={}/{} {}/{} segments {:.3f} segments/s".format(
                args.shard_index, args.shard_count, processed, len(pending),
                processed / elapsed,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
