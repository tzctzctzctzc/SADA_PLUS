"""Localize a known action with frame-wise CLIP similarity for RQ2."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.run_clip_target_adapter import (
    LABELS, PROMPT_SHA256, read_jsonl, shard_for_video, text_features,
)
from tools.run_clip_then_llava import frame_paths


PROMPT_VERSION = "rq2-clip-known-class-peak-half-range-16f-v1"
PROTOCOL_SHA256 = hashlib.sha256(
    (PROMPT_VERSION + "|" + PROMPT_SHA256).encode("utf-8")
).hexdigest()


def localize_peak(scores):
    values = np.asarray(scores, dtype=np.float64)
    peak = int(np.argmax(values))
    threshold = 0.5 * (float(values.min()) + float(values.max()))
    left = peak
    right = peak
    while left > 0 and values[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(values) and values[right + 1] >= threshold:
        right += 1
    return left, right + 1, threshold


def is_current(row, model, frame_count):
    return bool(
        row.get("status") == "ok"
        and row.get("model") == model
        and row.get("frame_count") == frame_count
        and row.get("protocol_sha256") == PROTOCOL_SHA256
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq2_boundary_16f")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=16)
    parser.add_argument("--segment-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import clip
    import torch
    from PIL import Image

    rows = [
        row for row in read_jsonl(args.manifest)
        if shard_for_video(row["video_id"], args.shard_count) == args.shard_index
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output) if output.is_file() else []
    completed = {
        row["candidate_id"] for row in prior if is_current(row, args.model, args.frame_count)
    }
    pending = [row for row in rows if row["candidate_id"] not in completed]
    model, preprocess = clip.load(
        args.model, device=args.device, jit=False, download_root=args.model_cache
    )
    model.eval()
    texts = text_features(model, clip, args.device)
    processed = 0
    for offset in range(0, len(pending), args.segment_batch_size):
        batch_rows = pending[offset:offset + args.segment_batch_size]
        images = []
        for row in batch_rows:
            for path in frame_paths(
                args.frame_cache, args.cache_namespace, row["candidate_id"], args.frame_count
            ):
                with Image.open(path) as image:
                    images.append(preprocess(image.convert("RGB")))
        encoded = []
        with torch.no_grad():
            for start in range(0, len(images), args.image_batch_size):
                tensor = torch.stack(images[start:start + args.image_batch_size]).to(args.device)
                part = model.encode_image(tensor).float()
                encoded.append(part / part.norm(dim=-1, keepdim=True))
        encoded = torch.cat(encoded)
        results = []
        for index, row in enumerate(batch_rows):
            class_id = int(row["gt_label_id"])
            begin = index * args.frame_count
            scores = (encoded[begin:begin + args.frame_count] @ texts[class_id]).cpu().tolist()
            pred_start, pred_end, threshold = localize_peak(scores)
            results.append({
                "candidate_id": row["candidate_id"], "video_id": row["video_id"],
                "status": "ok", "model": args.model,
                "wire_api": "clip-framewise-known-class-similarity",
                "prompt_version": PROMPT_VERSION, "protocol_sha256": PROTOCOL_SHA256,
                "frame_count": args.frame_count, "frame_similarities": scores,
                "relative_threshold": threshold,
                "pred_start_position": pred_start, "pred_end_position": pred_end,
            })
        with output.open("a", encoding="utf-8") as stream:
            for result in results:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
        processed += len(results)
        print("{}/{} segments".format(processed, len(pending)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
