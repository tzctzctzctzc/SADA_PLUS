"""Prepare one shared eight-frame cache for all RQ1 models."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tools.run_clip_target_adapter import _extract_frame, proposal_timestamps, read_jsonl


def frame_path(output_dir, index):
    return Path(output_dir) / "frame_{:02d}.jpg".format(index)


def frames_complete(output_dir, frame_count):
    from PIL import Image

    for index in range(1, frame_count + 1):
        path = frame_path(output_dir, index)
        if not path.is_file() or not path.stat().st_size:
            return False
        try:
            with Image.open(path) as image:
                image.load()
        except (OSError, ValueError):
            return False
    return True


def prepare_video(video_id, rows, video_root, output_root, frame_count, force=False):
    import cv2

    video = None
    for suffix in (".MP4", ".mp4"):
        candidate = Path(video_root) / (video_id + suffix)
        if (
            candidate.is_file()
            and candidate.stat().st_size
            and not Path(str(candidate) + ".aria2").exists()
        ):
            video = candidate
            break
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
        for row in sorted(rows, key=lambda item: float(item["proposal_start_sec"])):
            output_dir = Path(output_root) / row["candidate_id"]
            if not force and frames_complete(output_dir, frame_count):
                prepared += 1
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamps = proposal_timestamps(row, frame_count)
            for index, timestamp in enumerate(timestamps, 1):
                output = frame_path(output_dir, index)
                if not force and output.is_file() and output.stat().st_size:
                    continue
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
    parser.add_argument("--cache-namespace", default="rq1_8f")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Only prepare this video ID; may be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frames for the selected videos.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.frame_count <= 0 or args.workers <= 0:
        raise ValueError("frame count and workers must be positive")
    rows = read_jsonl(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    if args.video_id:
        requested = set(args.video_id)
        unknown = requested.difference(grouped)
        if unknown:
            raise ValueError("video IDs not in manifest: {}".format(sorted(unknown)))
        grouped = defaultdict(list, {
            video_id: grouped[video_id] for video_id in sorted(requested)
        })
    total_segments = sum(len(video_rows) for video_rows in grouped.values())
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
                args.frame_count,
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
                "video {}/{} {} segments={} total={} status={}".format(
                    index, len(futures), video_id, count, completed, "error" if error else "ok"
                ),
                flush=True,
            )
    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "frame_count": args.frame_count,
        "unique_videos": len(grouped),
        "total_segments": total_segments,
        "prepared_segments": completed,
        "errors": errors,
    }
    Path(args.frame_cache).joinpath("{}_prepare_report.json".format(args.cache_namespace)).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if completed == total_segments and not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
