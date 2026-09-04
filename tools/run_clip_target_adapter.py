"""Score frozen SADA proposals with OpenAI CLIP image-text similarity."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


LABELS = (
    "TAKE", "PUT", "WASH", "OPEN", "CLOSE", "INSERT", "TURN_ON",
    "CUT", "TURN_OFF", "POUR",
)
ACTION_PHRASES = (
    "taking or picking up an object",
    "putting or placing an object down",
    "washing or cleaning an object",
    "opening an object",
    "closing an object",
    "inserting an object into a container",
    "turning on a device, light, flame, or flow",
    "cutting or slicing an object",
    "turning off a device, light, flame, or flow",
    "pouring contents from a container",
)
PROMPT_TEMPLATES = (
    "an egocentric video frame of a person {}",
    "a first-person view of a person {}",
    "a kitchen video frame showing a person {}",
)
PROMPT_VERSION = "openai-clip-epic-proposal-class-v1"
PROMPT_SHA256 = hashlib.sha256(
    json.dumps(
        {"labels": LABELS, "phrases": ACTION_PHRASES, "templates": PROMPT_TEMPLATES},
        sort_keys=True,
    ).encode()
).hexdigest()
EPIC_55_BASE_URL = "https://data.bris.ac.uk/datasets/3h91syskeag5572hl6tvuovwv4d"
EPIC_100_BASE_URL = "https://data.bris.ac.uk/datasets/2g1n6qdydwaa9u22shpxqzp0t8m"


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def proposal_timestamps(row, frame_count=8):
    start = float(row["proposal_start_sec"])
    end = float(row["proposal_end_sec"])
    if not 0.0 <= start < end:
        raise ValueError("proposal interval must satisfy 0 <= start < end")
    span = end - start
    return [start + span * (index + 0.5) / frame_count for index in range(frame_count)]


def shard_for_video(video_id, shard_count):
    digest = hashlib.sha256(str(video_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % shard_count


def video_path(root, video_id):
    if root is None:
        return None
    for suffix in (".MP4", ".mp4"):
        path = Path(root) / (video_id + suffix)
        if path.is_file() and path.stat().st_size:
            return path
    return None


def epic_55_splits(path):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {row["video_id"]: row["split"] for row in csv.DictReader(stream)}


def epic_video_url(video_id, old_splits):
    participant, sequence = video_id.split("_", 1)
    if len(sequence) == 3:
        return "/".join(
            (EPIC_100_BASE_URL, participant, "videos", video_id + ".MP4")
        )
    split = old_splits.get(video_id)
    if split is None:
        raise ValueError("EPIC-55 split is unavailable for {}".format(video_id))
    return "/".join(
        (EPIC_55_BASE_URL, "videos", split, participant, video_id + ".MP4")
    )


def download_video(video_id, old_splits, output, retries=3):
    url = epic_video_url(video_id, old_splits)
    output = Path(output)
    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=180) as source, partial.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)
            partial.replace(output)
            return output
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == retries:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def video_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def prepare_frames(row, video, cache_root, namespace, frame_count=8):
    duration = video_duration(video)
    output_dir = Path(cache_root) / namespace / row["candidate_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    timestamps = proposal_timestamps(row, frame_count)
    for index, timestamp in enumerate(timestamps, 1):
        timestamp = min(max(timestamp, 0.0), duration)
        frame = output_dir / "frame_{:02d}.jpg".format(index)
        if not frame.is_file() or not frame.stat().st_size:
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", "{:.3f}".format(timestamp), "-i", str(video),
                    "-frames:v", "1", "-q:v", "3", str(frame),
                ],
                check=True,
            )
        frames.append(frame)
    return frames, timestamps


def text_features(model, clip_module, device):
    import torch

    prompts = [
        template.format(phrase)
        for phrase in ACTION_PHRASES
        for template in PROMPT_TEMPLATES
    ]
    tokens = clip_module.tokenize(prompts).to(device)
    with torch.no_grad():
        encoded = model.encode_text(tokens).float()
    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    encoded = encoded.reshape(len(LABELS), len(PROMPT_TEMPLATES), -1).mean(dim=1)
    return encoded / encoded.norm(dim=-1, keepdim=True)


def score_rows(model, preprocess, text, rows_and_frames, device, image_batch_size):
    import torch
    from PIL import Image

    images = []
    counts = []
    for _, frames, _ in rows_and_frames:
        counts.append(len(frames))
        for frame in frames:
            with Image.open(frame) as image:
                images.append(preprocess(image.convert("RGB")))
    encoded_parts = []
    with torch.no_grad():
        for start in range(0, len(images), image_batch_size):
            batch = torch.stack(images[start:start + image_batch_size]).to(device)
            encoded_parts.append(model.encode_image(batch).float())
    encoded = torch.cat(encoded_parts)
    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    results = []
    offset = 0
    logit_scale = float(model.logit_scale.exp().detach().cpu())
    for count in counts:
        visual = encoded[offset:offset + count].mean(dim=0)
        visual = visual / visual.norm()
        similarities = visual @ text.T
        probabilities = (logit_scale * similarities).softmax(dim=-1)
        values, indices = probabilities.topk(k=2)
        results.append({
            "prediction": LABELS[int(indices[0])],
            "class_id": int(indices[0]),
            "confidence": float(values[0]),
            "margin": float(values[0] - values[1]),
            "class_probabilities": probabilities.cpu().tolist(),
            "class_similarities": similarities.cpu().tolist(),
            "clip_logit_scale": logit_scale,
        })
        offset += count
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", default=None)
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--epic55-splits", default=None)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--model", default="ViT-B/16")
    parser.add_argument("--model-cache", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-batch-size", type=int, default=128)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit-videos", type=int, default=0)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard")
    if args.frame_count <= 0 or args.image_batch_size <= 0:
        raise ValueError("frame count and batch size must be positive")
    if args.download_missing and args.epic55_splits is None:
        raise ValueError("downloading requires --epic55-splits")

    import clip

    rows = read_jsonl(args.manifest)
    grouped = defaultdict(list)
    for row in rows:
        if shard_for_video(row["video_id"], args.shard_count) == args.shard_index:
            grouped[row["video_id"]].append(row)
    video_ids = sorted(grouped)
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]
    namespace = args.cache_namespace or Path(args.manifest).stem
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prior_rows = read_jsonl(output) if output.exists() else []
    completed = {
        row["candidate_id"]
        for row in prior_rows
        if row.get("status") == "ok"
    }
    model, preprocess = clip.load(
        args.model,
        device=args.device,
        jit=False,
        download_root=args.model_cache,
    )
    model.eval()
    text = text_features(model, clip, args.device)
    old_splits = epic_55_splits(args.epic55_splits)
    failed = 0

    with tempfile.TemporaryDirectory(prefix="clip-video-") as temporary:
        for video_number, video_id in enumerate(video_ids, 1):
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
                    )
                    downloaded = True
                prepared = []
                for row in pending:
                    frames, timestamps = prepare_frames(
                        row,
                        local_video,
                        args.frame_cache,
                        namespace,
                        args.frame_count,
                    )
                    prepared.append((row, frames, timestamps))
                scores = score_rows(
                    model,
                    preprocess,
                    text,
                    prepared,
                    args.device,
                    args.image_batch_size,
                )
                with output.open("a", encoding="utf-8") as stream:
                    for (row, _, timestamps), score in zip(prepared, scores):
                        result = {
                            "candidate_id": row["candidate_id"],
                            "video_id": video_id,
                            "absolute_feature_index": row["absolute_feature_index"],
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
                print(
                    "{}/{} {} proposals={} ok".format(
                        video_number, len(video_ids), video_id, len(pending)
                    ),
                    flush=True,
                )
            except Exception as error:
                failed += len(pending)
                print(
                    "{}/{} {} proposals={} error={}".format(
                        video_number, len(video_ids), video_id, len(pending), error
                    ),
                    flush=True,
                )
            finally:
                if downloaded and local_video is not None and Path(local_video).exists():
                    Path(local_video).unlink()

    if failed:
        raise RuntimeError("{} proposal scores failed".format(failed))


if __name__ == "__main__":
    main()
