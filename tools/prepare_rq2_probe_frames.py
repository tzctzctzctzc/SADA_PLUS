"""Prepare variable-length 2 FPS frame caches for the RQ2 temporal probe."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.prepare_rq1_frames import frame_path, frames_complete
from tools.run_clip_target_adapter import _extract_frame, read_jsonl


def find_video(video_root, video_id):
    for suffix in (".MP4", ".mp4"):
        candidate = Path(video_root) / (video_id + suffix)
        if (
            candidate.is_file()
            and candidate.stat().st_size
            and not Path(str(candidate) + ".aria2").exists()
        ):
            return candidate
    return None


def prepare_video(video_id, rows, video_root, output_root, force=False):
    import cv2

    video = find_video(video_root, video_id)
    if video is None:
        return video_id, 0, "video missing"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return video_id, 0, "OpenCV could not open video"
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else None
    prepared = 0
    try:
        for row in sorted(rows, key=lambda item: item["candidate_id"]):
            frame_count = int(row["frame_count"])
            timestamps = list(map(float, row["frame_timestamps_sec"]))
            if len(timestamps) != frame_count:
                raise ValueError("frame timestamps do not match frame_count")
            output_dir = Path(output_root) / row["candidate_id"]
            if not force and frames_complete(output_dir, frame_count):
                prepared += 1
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            for index, timestamp in enumerate(timestamps, 1):
                output = frame_path(output_dir, index)
                if not force and output.is_file() and output.stat().st_size:
                    continue
                safe_time = max(0.0, timestamp)
                if duration is not None:
                    safe_time = min(
                        safe_time, max(0.0, duration - 1.0 / max(fps, 1.0))
                    )
                capture.set(cv2.CAP_PROP_POS_MSEC, safe_time * 1000.0)
                ok, image = capture.read()
                if not ok or image is None or not cv2.imwrite(str(output), image):
                    if output.exists():
                        output.unlink()
                    _extract_frame(video, safe_time, output)
                if not output.is_file() or not output.stat().st_size:
                    raise RuntimeError("empty frame {}".format(output))
            prepared += 1
    except Exception as error:
        return video_id, prepared, "{}: {}".format(type(error).__name__, error)
    finally:
        capture.release()
    return video_id, prepared, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    output_root = Path(args.frame_cache) / args.cache_namespace
    output_root.mkdir(parents=True, exist_ok=True)
    completed = 0
    errors = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                prepare_video,
                video_id,
                video_rows,
                args.video_root,
                output_root,
                args.force,
            ): video_id
            for video_id, video_rows in grouped.items()
        }
        for index, future in enumerate(as_completed(futures), 1):
            video_id, count, error = future.result()
            completed += count
            if error:
                errors[video_id] = error
            print(
                "video {}/{} {} windows={} total={} status={}".format(
                    index,
                    len(futures),
                    video_id,
                    count,
                    completed,
                    "error" if error else "ok",
                ),
                flush=True,
            )
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "cache_namespace": args.cache_namespace,
        "unique_videos": len(grouped),
        "total_windows": len(rows),
        "prepared_windows": completed,
        "errors": errors,
    }
    Path(args.frame_cache).joinpath(
        "{}_prepare_report.json".format(args.cache_namespace)
    ).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if completed == len(rows) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
