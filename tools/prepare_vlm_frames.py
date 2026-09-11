"""Extract one shared chronological JPEG cache for all three VLM runners."""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _frame_dir_complete(directory, frame_count):
    directory = Path(directory)
    return directory.is_dir() and all(
        (directory / "frame_{:02d}.jpg".format(index)).is_file()
        and (directory / "frame_{:02d}.jpg".format(index)).stat().st_size
        for index in range(1, frame_count + 1)
    )


def _build_reuse_index(roots, frame_count):
    """Index old proposal caches so already extracted JPEGs are reused."""
    index = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for marker in root.rglob("frame_01.jpg"):
            directory = marker.parent
            candidate_id = directory.name
            if candidate_id not in index and _frame_dir_complete(directory, frame_count):
                index[candidate_id] = directory
    return index


def _link_reused_frames(source, destination):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(Path(source), destination, target_is_directory=True)


def _prepare_video(video_id, rows, args, old_splits):
    """Download one video, extract all requested proposals, then delete it."""
    complete = 0
    with tempfile.TemporaryDirectory(prefix="vlm-frame-video-") as temporary:
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
            for row, namespace, cache_root in rows:
                prepare_frames(
                    row,
                    local_video,
                    cache_root,
                    namespace,
                    args.frame_count,
                )
                complete += 1
            return video_id, complete, None
        except Exception as error:
            return video_id, complete, "{}: {}".format(type(error).__name__, error)
        finally:
            if downloaded and local_video is not None and Path(local_video).exists():
                Path(local_video).unlink()


def prepare_all_manifests(scenario_root, frame_cache, args, old_splits):
    """Prepare the requested shifts once per unique video, with bounded concurrency."""
    manifests = []
    for shift in args.shifts:
        root = Path(scenario_root) / shift
        if "source" in args.parts:
            manifests.append(
                (
                    root / "source_val_manifest.jsonl",
                    Path(frame_cache) / "source",
                    "{}_source".format(shift.lower()),
                ),
            )
        if "target" in args.parts:
            manifests.append(
                (
                    root / "target_train_manifest.jsonl",
                    Path(frame_cache) / "target",
                    shift.lower(),
                ),
            )

    reuse_index = _build_reuse_index(args.reuse_root, args.frame_count)
    jobs = defaultdict(list)
    total_rows = 0
    reused = 0
    for manifest_path, cache_root, namespace in manifests:
        for row in read_jsonl(manifest_path):
            total_rows += 1
            destination = Path(cache_root) / namespace / row["candidate_id"]
            if _frame_dir_complete(destination, args.frame_count):
                continue
            old_frames = reuse_index.get(row["candidate_id"])
            if old_frames is not None:
                _link_reused_frames(old_frames, destination)
                reused += 1
                continue
            jobs[row["video_id"]].append((row, namespace, cache_root))

    print(
        "{} proposals, reused {}, need {} unique videos".format(
            total_rows, reused, len(jobs)
        ),
        flush=True,
    )
    prepared = reused
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _prepare_video,
                video_id,
                rows,
                args,
                old_splits,
            )
            for video_id, rows in sorted(jobs.items())
        ]
        for number, future in enumerate(as_completed(futures), 1):
            video_id, count, error = future.result()
            prepared += count
            if error:
                print(
                    "video {}/{} {} failed after {} proposals: {}".format(
                        number, len(futures), video_id, count, error
                    ),
                    flush=True,
                )
            else:
                print(
                    "video {}/{} {} ready ({} proposals; total {})".format(
                        number, len(futures), video_id, count, prepared
                    ),
                    flush=True,
                )
    print("prepared {} / {} proposals".format(prepared, total_rows), flush=True)
    return prepared, total_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument(
        "--shifts",
        nargs="+",
        choices=SHIFT_NAMES,
        default=list(SHIFT_NAMES),
        help="shift(s) to prepare; defaults to all S1--S6",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("source", "target"),
        default=["source", "target"],
        help="manifest parts to prepare; defaults to source and target",
    )
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-connections", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-root",
        action="append",
        default=[],
        help="old frame-cache root(s) to reuse by candidate_id",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0:
        raise ValueError("frame count must be positive")
    Path(args.video_root).mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    prepared, total = prepare_all_manifests(
        args.scenario_root, args.frame_cache, args, old_splits
    )
    return 0 if prepared == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
