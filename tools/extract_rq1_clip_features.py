"""Extract resumable prompt-free CLIP segment features for RQ1 linear probes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.rq1_feature_store import FeatureStore, summarize_frame_features
from tools.run_clip_target_adapter import (
    LABELS,
    PROMPT_SHA256,
    PROMPT_VERSION,
    read_jsonl,
    text_features,
)
from tools.run_clip_then_llava import frame_paths


class SegmentFrameDataset:
    def __init__(self, rows, indices, frame_cache, cache_namespace,
                 frame_count, preprocess):
        self.rows = rows
        self.indices = list(indices)
        self.frame_cache = frame_cache
        self.cache_namespace = cache_namespace
        self.frame_count = int(frame_count)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        import torch
        from PIL import Image

        index = self.indices[item]
        images = []
        for path in frame_paths(
            self.frame_cache,
            self.cache_namespace,
            self.rows[index]["candidate_id"],
            frame_count=self.frame_count,
        ):
            with Image.open(path) as image:
                images.append(self.preprocess(image.convert("RGB")))
        return index, torch.stack(images)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_probe_8f")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=64)
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--pooling", choices=("static", "static_transition"), default="static")
    parser.add_argument("--text-features-output", default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.frame_count, args.segment_batch_size, args.image_batch_size) <= 0:
        raise ValueError("frame and batch sizes must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    import clip
    import torch
    from torch.utils.data import DataLoader

    rows = read_jsonl(args.manifest)
    model, preprocess = clip.load(
        args.model, device=args.device, jit=False, download_root=args.model_cache
    )
    model.eval()
    if args.text_features_output:
        text = text_features(model, clip, args.device)
        text_output = Path(args.text_features_output)
        text_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(text_output, text.detach().cpu().numpy().astype(np.float32))
        Path(str(text_output) + ".meta.json").write_text(json.dumps({
            "model": args.model,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": PROMPT_SHA256,
            "labels": list(LABELS),
            "logit_scale": float(model.logit_scale.exp().detach().cpu()),
        }, indent=2, sort_keys=True), encoding="utf-8")
    dimensions = int(model.visual.output_dim)
    if args.pooling == "static_transition":
        dimensions *= 2
    store = FeatureStore(
        args.output_prefix, args.manifest, rows, args.model, dimensions,
        args.frame_count, args.pooling,
    )
    pending = store.pending_indices()
    if args.limit > 0:
        pending = pending[:args.limit]
    dataset = SegmentFrameDataset(
        rows, pending, args.frame_cache, args.cache_namespace,
        args.frame_count, preprocess,
    )
    loader_kwargs = {
        "batch_size": args.segment_batch_size,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    if args.workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
            "multiprocessing_context": "spawn",
        })
    loader = DataLoader(dataset, **loader_kwargs)
    started = time.time()
    processed = 0
    for indices, videos in loader:
        indices = indices.tolist()
        images = videos.flatten(0, 1)
        encoded = []
        with torch.inference_mode():
            for start in range(0, len(images), args.image_batch_size):
                batch = images[start:start + args.image_batch_size].to(
                    args.device, non_blocking=True
                )
                values = model.encode_image(batch).float()
                values = values / values.norm(dim=-1, keepdim=True)
                encoded.append(values.cpu().numpy())
        encoded = np.concatenate(encoded)
        features = []
        for local_index in range(len(indices)):
            begin = local_index * args.frame_count
            frames = encoded[begin:begin + args.frame_count]
            features.append(summarize_frame_features(frames, args.pooling))
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
