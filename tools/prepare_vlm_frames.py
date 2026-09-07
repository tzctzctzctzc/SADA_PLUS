"""Extract one shared chronological JPEG cache for all three VLM runners."""

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path

from tools.run_clip_target_adapter import (
    download_video,
    epic_55_splits,
    prepare_frames,
    video_path,
)
from tools.run_clip_then_llava import read_jsonl


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def prepare_manifest(manifest_path, frame_cache, namespace, args, old_splits):
    rows = read_jsonl(manifest_path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    complete = 0
    with tempfile.TemporaryDirectory(prefix="vlm-frame-video-") as temporary:
        for video_id in sorted(grouped):
            local_video = video_path(args.video_root, video_id)
            downloaded = False
            try:
                if local_video is None:
                    local_video = download_video(
                        video_id,
                        old_splits,
                        Path(temporary) / (video_id + ".MP4"),
                        retries=args.download_retries,
                        connections=args.download_connections,
                    )
                    downloaded = True
                for row in grouped[video_id]:
                    prepare_frames(
                        row,
                        local_video,
                        frame_cache,
                        namespace,
                        args.frame_count,
                    )
                    complete += 1
                print("{} {} proposals ready".format(namespace, complete), flush=True)
            finally:
                if downloaded and local_video is not None and Path(local_video).exists():
                    Path(local_video).unlink()
    return complete


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-connections", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0:
        raise ValueError("frame count must be positive")
    Path(args.video_root).mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    total = 0
    for shift in SHIFT_NAMES:
        root = Path(args.scenario_root) / shift
        total += prepare_manifest(
            root / "source_val_manifest.jsonl",
            Path(args.frame_cache) / "source",
            "{}_source".format(shift.lower()),
            args,
            old_splits,
        )
        total += prepare_manifest(
            root / "target_train_manifest.jsonl",
            Path(args.frame_cache) / "target",
            shift.lower(),
            args,
            old_splits,
        )
    print("prepared {} proposals".format(total), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
