"""Download and retain the EPIC videos referenced by S1--S6 manifests."""

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.run_clip_target_adapter import (
    download_video,
    epic_55_splits,
    file_md5,
    prepare_frames,
    video_path,
)
from tools.run_clip_then_llava import read_jsonl


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def manifest_rows(root, shifts):
    rows_by_video = {}
    root = Path(root)
    for shift in shifts:
        manifests = (
            ("source_val_manifest.jsonl", "source", "{}_source".format(shift.lower())),
            ("target_train_manifest.jsonl", "target", shift.lower()),
        )
        for filename, scope, namespace in manifests:
            for row in read_jsonl(root / shift / filename):
                rows_by_video.setdefault(row["video_id"], []).append(
                    (row, scope, namespace)
                )
    return rows_by_video


def video_md5s(path):
    if path is None:
        return {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {
            Path(row[1]).stem: row[0].lower()
            for row in csv.reader(stream)
            if len(row) >= 2 and row[1].lower().endswith(".mp4")
        }


def backup_corrupt(path):
    backup = Path(str(path) + ".corrupt_md5")
    suffix = 1
    while backup.exists():
        backup = Path(str(path) + ".corrupt_md5.{}".format(suffix))
        suffix += 1
    Path(path).replace(backup)
    return backup


def download_one(video_id, rows, output_root, old_splits, retries, connections,
                 frame_cache, frame_count, expected_md5=None):
    existing = video_path(output_root, video_id)
    control = Path(str(existing) + ".aria2") if existing is not None else None
    try:
        if (
            existing is not None
            and not control.exists()
            and expected_md5
            and file_md5(existing) != expected_md5
        ):
            backup_corrupt(existing)
            existing = None
        if existing is not None and not control.exists():
            path = Path(existing)
            status = "cached"
        else:
            path = download_video(
                video_id,
                old_splits,
                Path(output_root) / (video_id + ".MP4"),
                retries=retries,
                connections=connections,
                expected_md5=expected_md5,
            )
            status = "downloaded"
        if frame_cache:
            for row, scope, namespace in rows:
                prepare_frames(
                    row,
                    path,
                    Path(frame_cache) / scope,
                    namespace,
                    frame_count,
                )
            status += "+frames"
        return video_id, status, Path(path).stat().st_size, None
    except Exception as error:
        return video_id, "failed", 0, "{}: {}".format(type(error).__name__, error)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument(
        "--md5-csv",
        default=None,
        help="official checksum CSV; cached and downloaded videos are verified when set",
    )
    parser.add_argument("--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES))
    parser.add_argument(
        "--video-ids",
        nargs="+",
        default=None,
        help="download only these manifest video IDs",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--connections", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--frame-cache", default=None)
    parser.add_argument("--frame-count", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0 or args.connections <= 0 or args.frame_count <= 0:
        raise ValueError("workers, connections, and frame count must be positive")
    output_root = Path(args.video_root)
    output_root.mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    expected_md5s = video_md5s(args.md5_csv)
    rows_by_video = manifest_rows(args.scenario_root, args.shifts)
    if args.video_ids:
        missing_ids = sorted(set(args.video_ids).difference(rows_by_video))
        if missing_ids:
            raise ValueError("video IDs absent from manifests: {}".format(missing_ids))
        video_ids = sorted(set(args.video_ids))
    else:
        video_ids = sorted(rows_by_video)
    if args.limit_videos > 0:
        video_ids = video_ids[:args.limit_videos]

    total_bytes = 0
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                download_one,
                video_id,
                rows_by_video[video_id],
                output_root,
                old_splits,
                args.retries,
                args.connections,
                args.frame_cache,
                args.frame_count,
                expected_md5s.get(video_id),
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
