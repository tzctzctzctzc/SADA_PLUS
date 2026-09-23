"""Audit sampled MP4 frames and recover undecodable frames from an official RGB tar."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

from tools.rq2_dense_protocol import sampled_source_indices


def exact_mp4_frame(video_path, index):
    code = (
        "import cv2,sys\n"
        "c=cv2.VideoCapture(sys.argv[1]); c.set(cv2.CAP_PROP_POS_FRAMES,int(sys.argv[2]));"
        "ok,f=c.read(); c.release();"
        "sys.exit(2) if not ok else None;"
        "sys.stdout.buffer.write(f.tobytes())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(video_path), str(index)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False,
    )
    if result.returncode != 0:
        return None
    frame = np.frombuffer(result.stdout, dtype=np.uint8)
    return frame.reshape(1080, 1920, 3) if frame.size == 1080 * 1920 * 3 else None


def tar_frame(archive, index):
    name = "./frame_{:010d}.jpg".format(index)
    member = archive.getmember(name)
    encoded = np.frombuffer(archive.extractfile(member).read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("cannot decode official RGB member: " + name)
    return name, frame


def small(frame):
    return cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)


def md5_file(path):
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", type=Path, required=True)
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--recover", action="store_true")
    args = parser.parse_args()
    video_id = args.video_path.stem
    if video_id != "P30_08":
        raise ValueError("this audited recovery is specific to P30_08")
    tar_md5 = md5_file(args.tar_path)
    video_md5 = md5_file(args.video_path)
    if tar_md5 != "7dea2ad436564ea7713c86b5fe5a62e5":
        raise RuntimeError("official RGB tar MD5 mismatch")
    if video_md5 != "bd32d6e5c5815beb42310563b0ce750e":
        raise RuntimeError("official MP4 MD5 mismatch")
    cap = cv2.VideoCapture(str(args.video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if abs(fps - 60000 / 1001) > 0.01:
        raise RuntimeError("unexpected MP4 frame rate")
    candidates = sorted({
        int(index)
        for token in range(max(0, args.start // 32 - 2), args.stop // 32 + 3)
        for index in sampled_source_indices(token, 8, fps, frame_count)
        if args.start <= index <= args.stop
    })
    manifest_path = args.recovery_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    matches = []
    with tarfile.open(args.tar_path, "r") as archive:
        for index in candidates:
            decoded = exact_mp4_frame(args.video_path, index)
            tar_index = round(index * 1001 / 1000) + 1
            if decoded is None:
                failures.append((index, tar_index))
            else:
                reference = small(decoded)
                errors = []
                for offset in (-1, 0, 1):
                    _, candidate = tar_frame(archive, tar_index + offset)
                    errors.append((round(float(np.abs(reference - small(candidate)).mean()), 3), tar_index + offset))
                matches.append((index, tar_index, min(errors)))
        print(json.dumps({"fps": fps, "candidates": len(candidates), "failed": failures,
                          "healthy_alignment": matches}, indent=2), flush=True)
        if not args.recover:
            return
        if not matches or any(abs(best - predicted) > 1 for _, predicted, (_, best) in matches):
            raise RuntimeError("official RGB alignment is not validated")
        for index, tar_index in failures:
            key = "{}:{}".format(video_id, index)
            if key in manifest["frames"]:
                continue
            before = max((item for item in matches if item[0] < index), default=None)
            after = min((item for item in matches if item[0] > index), default=None)
            if before is None or after is None:
                raise RuntimeError("recovery frame lacks healthy alignment brackets")
            member, frame = tar_frame(archive, tar_index)
            frame_dir = args.recovery_root / video_id
            frame_dir.mkdir(parents=True, exist_ok=True)
            frame_path = frame_dir / (str(index) + ".npy")
            if frame_path.exists():
                raise RuntimeError("recovery target already exists: " + str(frame_path))
            with frame_path.open("wb") as stream:
                np.save(stream, frame, allow_pickle=False)
            manifest["frames"][key] = {
                "color_order": "bgr", "dtype": "uint8", "frame_index": index,
                "frame_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
                "mapping_method": "mp4_pts_frame_rate_alignment_round_time_times_60_plus_one",
                "mapping_validation": [
                    {"mp4_frame_index": mp4_index, "predicted_tar_index": predicted,
                     "best_tar_index": best, "mean_abs_error_64x36": error}
                    for mp4_index, predicted, (error, best) in (before, after)
                ],
                "mp4_fps": {"numerator": 60000, "denominator": 1001},
                "mp4_frame_time_sec": index * 1001 / 60000,
                "path": "{}/{}.npy".format(video_id, index),
                "pixel_array_semantics": "opencv_bgr_before_reader_color_conversion",
                "rgb_tar_frame_index": tar_index, "rgb_tar_md5": tar_md5,
                "rgb_tar_member": member, "shape": list(frame.shape),
                "source_kind": "official_rgb_frame_tar_time_aligned_to_mp4_pts",
                "video_md5": video_md5,
            }
        backup = manifest_path.with_name("manifest.before_p30_08_scan.json")
        if not backup.exists():
            backup.write_bytes(manifest_path.read_bytes())
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, manifest_path)
        print("manifest_sha256=" + hashlib.sha256(manifest_path.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
