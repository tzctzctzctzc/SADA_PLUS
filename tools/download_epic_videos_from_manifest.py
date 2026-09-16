"""Download missing EPIC-KITCHENS videos referenced by a JSONL manifest."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools import run_clip_target_adapter as adapter
from tools.run_clip_target_adapter import download_video, epic_55_splits, video_path
from tools.run_clip_then_llava import read_jsonl


def load_md5s(path):
    if not path:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {
            Path(row[1]).stem: row[0].lower()
            for row in csv.reader(stream)
            if len(row) >= 2 and row[1].lower().endswith(".mp4")
        }


def download_one(video_id, output_root, old_splits, retries, connections, expected_md5):
    existing = video_path(output_root, video_id)
    control = Path(str(existing) + ".aria2") if existing is not None else None
    if existing is not None and not control.exists():
        return video_id, "cached", Path(existing).stat().st_size, None
    try:
        path = download_video(
            video_id,
            old_splits,
            Path(output_root) / (video_id + ".MP4"),
            retries=retries,
            connections=connections,
            expected_md5=expected_md5,
        )
        return video_id, "downloaded", Path(path).stat().st_size, None
    except Exception as error:
        return video_id, "failed", 0, "{}: {}".format(type(error).__name__, error)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--md5-csv", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--connections", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--use-http",
        action="store_true",
        help="Use the official Bristol HTTP endpoint when HTTPS video GETs stall.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.workers, args.connections, args.retries) <= 0:
        raise ValueError("worker, connection, and retry counts must be positive")
    rows = read_jsonl(args.manifest)
    if args.use_http:
        adapter.EPIC_55_BASE_URL = adapter.EPIC_55_BASE_URL.replace("https://", "http://")
        adapter.EPIC_100_BASE_URL = adapter.EPIC_100_BASE_URL.replace("https://", "http://")
    video_ids = sorted({row["video_id"] for row in rows})
    output_root = Path(args.video_root)
    output_root.mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    md5s = load_md5s(args.md5_csv)
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one, video_id, output_root, old_splits,
                args.retries, args.connections, md5s.get(video_id),
            )
            for video_id in video_ids
            if video_path(output_root, video_id) is None
        ]
        for number, future in enumerate(as_completed(futures), 1):
            video_id, status, size, error = future.result()
            if error:
                failed.append(video_id)
                print("{}/{} {} failed: {}".format(number, len(futures), video_id, error), flush=True)
            else:
                print(
                    "{}/{} {} {} {:.2f} GiB".format(
                        number, len(futures), video_id, status, size / 1024 ** 3
                    ), flush=True,
                )
    print(
        "requested={} missing={} failed={}".format(
            len(video_ids), len(futures), len(failed)
        ), flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
