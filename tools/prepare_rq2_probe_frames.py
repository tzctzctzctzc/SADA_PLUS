"""Prepare variable-length 2 FPS frame caches for the RQ2 temporal probe."""

import argparse
import json
import os
import shutil
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


def timestamp_key(timestamp):
    return int(round(float(timestamp) * 1_000_000))


def link_frame(source, destination, force=False):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not force and destination.stat().st_size:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def prepare_video(
    video_id, rows, video_root, output_root, force=False, decode_mode="seek"
):
    import cv2

    video = find_video(video_root, video_id)
    if video is None:
        return video_id, 0, {}, "video missing"
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return video_id, 0, {}, "OpenCV could not open video"
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 and total_frames > 0 else None
    prepared = 0
    frame_references = 0
    unique_requests = {}
    row_requests = []
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
            requests = []
            for index, timestamp in enumerate(timestamps, 1):
                key = timestamp_key(timestamp)
                unique_requests.setdefault(key, timestamp)
                requests.append((key, frame_path(output_dir, index)))
                frame_references += 1
            row_requests.append((output_dir, frame_count, requests))
        bank = Path(output_root) / "_frame_bank" / video_id
        bank.mkdir(parents=True, exist_ok=True)
        pending_bank = {}
        for key, timestamp in unique_requests.items():
            output = bank / "t_{:016d}.jpg".format(key)
            if force and output.exists():
                output.unlink()
            if not output.is_file() or not output.stat().st_size:
                pending_bank[key] = (timestamp, output)
        if (
            decode_mode == "sequential"
            and pending_bank
            and fps > 0
            and total_frames > 0
        ):
            by_frame = defaultdict(list)
            maximum_index = max(0, int(total_frames) - 1)
            for key, (timestamp, output) in pending_bank.items():
                safe_time = max(0.0, float(timestamp))
                if duration is not None:
                    safe_time = min(safe_time, max(0.0, duration - 1.0 / max(fps, 1.0)))
                frame_index = min(max(int(round(safe_time * fps)), 0), maximum_index)
                by_frame[frame_index].append((key, safe_time, output))
            first_index = min(by_frame)
            last_index = max(by_frame)
            capture.set(cv2.CAP_PROP_POS_FRAMES, first_index)
            frame_index = first_index
            while frame_index <= last_index:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                requests = by_frame.get(frame_index, [])
                for _, _, output in requests:
                    if not cv2.imwrite(str(output), image):
                        raise RuntimeError("could not write frame {}".format(output))
                frame_index += 1
        for key, (timestamp, output) in sorted(pending_bank.items()):
            if not output.is_file() or not output.stat().st_size:
                safe_time = max(0.0, float(timestamp))
                if duration is not None:
                    safe_time = min(safe_time, max(0.0, duration - 1.0 / max(fps, 1.0)))
                capture.set(cv2.CAP_PROP_POS_MSEC, safe_time * 1000.0)
                ok, image = capture.read()
                if not ok or image is None or not cv2.imwrite(str(output), image):
                    if output.exists():
                        output.unlink()
                    _extract_frame(video, safe_time, output)
                if not output.is_file() or not output.stat().st_size:
                    raise RuntimeError("empty frame {}".format(output))
        for output_dir, frame_count, requests in row_requests:
            for key, destination in requests:
                link_frame(
                    bank / "t_{:016d}.jpg".format(key), destination, force=force
                )
            if not frames_complete(output_dir, frame_count):
                raise RuntimeError("incomplete linked frame window {}".format(output_dir))
            prepared += 1
    except Exception as error:
        return video_id, prepared, {}, "{}: {}".format(type(error).__name__, error)
    finally:
        capture.release()
    return video_id, prepared, {
        "frame_references": frame_references,
        "unique_frames": len(unique_requests),
    }, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--frame-cache", required=True)
    parser.add_argument("--cache-namespace", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--decode-mode", choices=("sequential", "seek"), default="seek"
    )
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
    frame_references = 0
    unique_frames = 0
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
                args.decode_mode,
            ): video_id
            for video_id, video_rows in grouped.items()
        }
        for index, future in enumerate(as_completed(futures), 1):
            video_id, count, stats, error = future.result()
            completed += count
            frame_references += int(stats.get("frame_references", 0))
            unique_frames += int(stats.get("unique_frames", 0))
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
        "decode_mode": args.decode_mode,
        "unique_videos": len(grouped),
        "total_windows": len(rows),
        "prepared_windows": completed,
        "frame_references_prepared": frame_references,
        "unique_frames_decoded": unique_frames,
        "decode_reuse_factor": (
            frame_references / unique_frames if unique_frames else None
        ),
        "errors": errors,
    }
    Path(args.frame_cache).joinpath(
        "{}_prepare_report.json".format(args.cache_namespace)
    ).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if completed == len(rows) and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
