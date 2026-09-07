"""Download and retain the EPIC videos referenced by S1--S6 manifests."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.run_clip_target_adapter import download_video, epic_55_splits, video_path
from tools.run_clip_then_llava import read_jsonl


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def manifest_video_ids(root, shifts):
    video_ids = set()
    root = Path(root)
    for shift in shifts:
        for filename in ("source_val_manifest.jsonl", "target_train_manifest.jsonl"):
            video_ids.update(
                row["video_id"] for row in read_jsonl(root / shift / filename)
            )
    return sorted(video_ids)


def download_one(video_id, output_root, old_splits, retries, connections):
    existing = video_path(output_root, video_id)
    if existing is not None:
        return video_id, "cached", Path(existing).stat().st_size, None
    output = Path(output_root) / (video_id + ".MP4")
    try:
        path = download_video(
            video_id,
            old_splits,
            output,
            retries=retries,
            connections=connections,
        )
        return video_id, "downloaded", Path(path).stat().st_size, None
    except Exception as error:
        return video_id, "failed", 0, "{}: {}".format(type(error).__name__, error)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--connections", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-videos", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0 or args.connections <= 0:
        raise ValueError("workers and connections must be positive")
    output_root = Path(args.video_root)
    output_root.mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    video_ids = manifest_video_ids(args.scenario_root, args.shifts)
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]

    total_bytes = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                video_id,
                output_root,
                old_splits,
                args.retries,
                args.connections,
            )
            for video_id in video_ids
        ]
        for number, future in enumerate(as_completed(futures), 1):
            video_id, status, size, error = future.result()
            total_bytes += size
            if error:
                failed.append(video_id)
                print(
                    "{}/{} {} failed: {}".format(number, len(futures), video_id, error),
                    flush=True,
                )
            else:
                print(
                    "{}/{} {} {} {:.2f} GiB".format(
                        number, len(futures), video_id, status, size / 1024 ** 3
                    ),
                    flush=True,
                )
    print(
        "finished={} failed={} retained={:.2f} GiB".format(
            len(video_ids) - len(failed), len(failed), total_bytes / 1024 ** 3
        ),
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
