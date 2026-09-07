"""Run CLIP proposal classification across S1--S6 with one model load."""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from tools.run_clip_target_adapter import (
    ACTION_PHRASES,
    LABELS,
    PROMPT_SHA256,
    PROMPT_VERSION,
    download_video,
    epic_55_splits,
    prepare_frames,
    score_rows,
    shard_for_video,
    video_path,
)
from tools.run_clip_then_llava import read_jsonl


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def score_manifest(model, preprocess, args, manifest_path, output_path,
                   frame_cache, cache_namespace, old_splits):
    rows = read_jsonl(manifest_path)
    grouped = defaultdict(list)
    for row in rows:
        if shard_for_video(row["video_id"], args.shard_count) == args.shard_index:
            grouped[row["video_id"]].append(row)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(output_path) if output_path.exists() else []
    completed = {row["candidate_id"] for row in prior if row.get("status") == "ok"}
    failed = 0
    with tempfile.TemporaryDirectory(prefix="clip-matrix-video-") as temporary:
        for video_id in sorted(grouped):
            pending = [
                row for row in grouped[video_id]
                if row["candidate_id"] not in completed
            ]
            if not pending:
                continue
            local_video = video_path(args.video_root, video_id)
            downloaded = False
            try:
                if local_video is None:
                    if not args.download_missing:
                        raise FileNotFoundError("video not found: {}".format(video_id))
                    local_video = download_video(
                        video_id,
                        old_splits,
                        Path(temporary) / (video_id + ".MP4"),
                        retries=args.download_retries,
                        connections=args.download_connections,
                    )
                    downloaded = True
                prepared = [
                    (
                        row,
                        *prepare_frames(
                            row,
                            local_video,
                            frame_cache,
                            cache_namespace,
                            args.frame_count,
                        ),
                    )
                    for row in pending
                ]
                scores = score_rows(
                    model,
                    preprocess,
                    text=args.text_features,
                    rows_and_frames=prepared,
                    device=args.device,
                    image_batch_size=args.image_batch_size,
                )
                with output_path.open("a", encoding="utf-8") as stream:
                    for (row, _, timestamps), score in zip(prepared, scores):
                        result = {
                            **row,
                            "status": "ok",
                            "model": args.model,
                            "wire_api": "openai-clip",
                            "prompt_version": PROMPT_VERSION,
                            "prompt_sha256": PROMPT_SHA256,
                            "frame_count": args.frame_count,
                            "frame_timestamps_sec": timestamps,
                            **score,
                        }
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                        completed.add(row["candidate_id"])
                print("{} proposals={} ok".format(video_id, len(pending)), flush=True)
            except Exception as error:
                failed += len(pending)
                print("{} proposals={} error={}".format(video_id, len(pending), error), flush=True)
            finally:
                if downloaded and local_video is not None and Path(local_video).exists():
                    Path(local_video).unlink()
    if failed:
        raise RuntimeError("{} proposal scores failed".format(failed))


def write_source_summary(manifest_path, predictions_path, output_path):
    predictions = {
        row["candidate_id"]: row
        for row in read_jsonl(predictions_path)
        if row.get("status") == "ok"
    }
    manifest = read_jsonl(manifest_path)
    known = [row for row in manifest if row.get("gt_assigned_class_ids")]
    correct = sum(
        predictions.get(row["candidate_id"], {}).get("prediction") in {
            LABELS[int(class_id)] for class_id in row["gt_assigned_class_ids"]
        }
        for row in known
    )
    counts = defaultdict(int)
    for row in predictions.values():
        counts[row.get("prediction", "UNKNOWN")] += 1
    summary = {
        "model": "openai-clip",
        "prompt_version": PROMPT_VERSION,
        "manifest_count": len(manifest),
        "success_count": len(predictions),
        "success_rate": len(predictions) / len(manifest) if manifest else 0.0,
        "known_action_count": len(known),
        "known_action_accuracy": correct / len(known) if known else None,
        "prediction_counts": dict(sorted(counts.items())),
    }
    Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-connections", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    import clip

    model, preprocess = clip.load(
        args.model,
        device=args.device,
        jit=False,
        download_root=args.model_cache,
    )
    model.eval()
    args.text_features = __import__("tools.run_clip_target_adapter", fromlist=["text_features"]).text_features(
        model, clip, args.device
    )
    old_splits = epic_55_splits(args.epic55_splits)
    for shift in SHIFT_NAMES:
        root = Path(args.scenario_root) / shift
        out = Path(args.output_root) / shift
        source_output = out / "source_predictions.jsonl"
        target_output = out / "target_predictions.jsonl"
        score_manifest(
            model, preprocess, args,
            root / "source_val_manifest.jsonl", source_output,
            Path(args.frame_cache) / "source", "{}_source".format(shift.lower()), old_splits,
        )
        write_source_summary(
            root / "source_val_manifest.jsonl", source_output, out / "source_summary.json"
        )
        score_manifest(
            model, preprocess, args,
            root / "target_train_manifest.jsonl", target_output,
            Path(args.frame_cache) / "target", shift.lower(), old_splits,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
