"""Run CLIP on the shared RQ1 GT-segment frame cache."""

import argparse
import json
import time
from pathlib import Path

from tools.run_clip_target_adapter import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    proposal_timestamps,
    read_jsonl,
    score_rows,
    text_features,
)
from tools.run_clip_then_llava import frame_paths


def is_current(row, model, frame_count):
    return bool(
        row.get("status") == "ok"
        and row.get("model") == model
        and int(row.get("frame_count", -1)) == frame_count
        and row.get("prompt_sha256") == PROMPT_SHA256
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default="rq1_8f")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--segment-batch-size", type=int, default=32)
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.frame_count, args.segment_batch_size, args.image_batch_size) <= 0:
        raise ValueError("frame and batch sizes must be positive")
    import clip

    rows = read_jsonl(args.manifest)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output) if output.exists() else []
    completed = {
        row["candidate_id"]
        for row in prior
        if is_current(row, args.model, args.frame_count)
    }
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if args.limit > 0:
        pending = pending[:args.limit]
    model, preprocess = clip.load(
        args.model, device=args.device, jit=False, download_root=args.model_cache
    )
    model.eval()
    texts = text_features(model, clip, args.device)
    started = time.time()
    processed = 0
    for offset in range(0, len(pending), args.segment_batch_size):
        batch_rows = pending[offset:offset + args.segment_batch_size]
        prepared = []
        for row in batch_rows:
            paths = frame_paths(
                args.frame_cache,
                args.cache_namespace,
                row["candidate_id"],
                frame_count=args.frame_count,
            )
            prepared.append((row, paths, proposal_timestamps(row, args.frame_count)))
        scores = score_rows(
            model,
            preprocess,
            texts,
            prepared,
            args.device,
            args.image_batch_size,
        )
        with output.open("a", encoding="utf-8") as stream:
            for (row, _, timestamps), score in zip(prepared, scores):
                stream.write(json.dumps({
                    "candidate_id": row["candidate_id"],
                    "video_id": row["video_id"],
                    "status": "ok",
                    "model": args.model,
                    "wire_api": "openai-clip",
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": PROMPT_SHA256,
                    "frame_count": args.frame_count,
                    "frame_timestamps_sec": timestamps,
                    **score,
                }, sort_keys=True) + "\n")
        processed += len(batch_rows)
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
