#!/usr/bin/env python3
"""Resume EPIC video downloads with parallel HTTP byte ranges.

This is for the new server where aria2 is unavailable and the existing
``.aria2`` files contain useful out-of-order progress.  Completed pieces are
kept; only missing pieces are fetched and written at their byte offsets.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


EPIC_55_BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
EPIC_100_BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
VIDEO_ID_PATTERN = re.compile(r"^P\d{2}_\d{2,3}$")


def read_ids(manifest_root):
    ids = set()
    for shift in ("S1", "S2", "S3", "S4", "S5", "S6"):
        for name in ("source_val_manifest.jsonl", "target_train_manifest.jsonl"):
            path = Path(manifest_root) / shift / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    ids.add(json.loads(line)["video_id"])
    return sorted(ids)


def read_splits(path):
    with open(path, newline="", encoding="utf-8") as stream:
        return {row["video_id"]: row["split"] for row in csv.DictReader(stream)}


def read_only_ids(path):
    ids = sorted({line.strip() for line in Path(path).read_text().splitlines() if line.strip()})
    invalid = [video_id for video_id in ids if not VIDEO_ID_PATTERN.fullmatch(video_id)]
    if invalid:
        raise ValueError("invalid video IDs: %s" % invalid)
    return ids


def read_md5s(path):
    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as stream:
        return {
            Path(row[1]).stem: row[0].lower()
            for row in csv.reader(stream)
            if len(row) >= 2 and row[1].lower().endswith(".mp4")
        }


def file_md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_corrupt(path):
    backup = Path(str(path) + ".corrupt_md5")
    suffix = 1
    while backup.exists():
        backup = Path(str(path) + ".corrupt_md5.%d" % suffix)
        suffix += 1
    Path(path).replace(backup)
    return backup


def video_url(video_id, splits):
    participant, sequence = video_id.split("_", 1)
    if len(sequence) == 3:
        return "%s/%s/videos/%s.MP4" % (EPIC_100_BASE, participant, video_id)
    split = splits.get(video_id)
    if split is None:
        raise KeyError("missing EPIC-55 split for %s" % video_id)
    return "%s/videos/%s/%s/%s.MP4" % (EPIC_55_BASE, split, participant, video_id)


def aria_state(path):
    """Return piece size, total size and completed-piece indexes."""
    data = Path(path).read_bytes()
    if len(data) < 34:
        raise ValueError("short aria2 control file: %s" % path)
    version = int.from_bytes(data[:2], "big")
    if version != 1:
        raise ValueError("unsupported aria2 control version %s" % version)
    info_len = int.from_bytes(data[6:10], "big")
    offset = 10 + info_len
    piece = int.from_bytes(data[offset:offset + 4], "big")
    total = int.from_bytes(data[offset + 4:offset + 12], "big")
    bit_len = int.from_bytes(data[offset + 20:offset + 24], "big")
    bitfield = data[offset + 24:offset + 24 + bit_len]
    done = {
        index
        for index in range((total + piece - 1) // piece)
        if index // 8 < len(bitfield)
        and (bitfield[index // 8] & (1 << (7 - (index % 8))))
    }
    return piece, total, done


def curl_range(url, start, end, output, timeout):
    command = [
        "curl", "--fail", "--silent", "--show-error", "--location",
        "--retry", "8", "--retry-all-errors", "--retry-delay", "2",
        "--connect-timeout", "30",
        "--max-time", str(timeout), "--range", "%d-%d" % (start, end),
        "--output", str(output), url,
    ]
    subprocess.run(command, check=True)


def download_piece(url, target, piece, total, index, timeout):
    start = index * piece
    end = min(total, start + piece) - 1
    temp = Path("%s.range.%06d" % (target, index))
    if temp.exists() and temp.stat().st_size == end - start + 1:
        pass
    else:
        temp.unlink(missing_ok=True)
        curl_range(url, start, end, temp, timeout)
    expected = end - start + 1
    if temp.stat().st_size != expected:
        raise RuntimeError("short range %s: %s/%s" % (index, temp.stat().st_size, expected))
    fd = os.open(target, os.O_WRONLY)
    try:
        with temp.open("rb") as stream:
            offset = start
            while True:
                block = stream.read(4 * 1024 * 1024)
                if not block:
                    break
                written = os.pwrite(fd, block, offset)
                if written != len(block):
                    raise OSError("short pwrite for piece %s" % index)
                offset += written
    finally:
        os.close(fd)
    temp.unlink(missing_ok=True)
    return index


def download_one(video_id, root, splits, workers, timeout, expected_md5=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = root / (video_id + ".MP4")
    sidecar = Path(str(target) + ".aria2")
    relay_part = Path(str(target) + ".relay.part")
    upload_part = Path(str(target) + ".upload.part")
    if target.exists() and not sidecar.exists():
        if expected_md5 and file_md5(target) != expected_md5:
            backup_corrupt(target)
        else:
            # A completed file may retain progress markers from an interrupted
            # relay; the official digest is the authoritative completion test.
            Path(str(target) + ".ranges.json").unlink(missing_ok=True)
            relay_part.unlink(missing_ok=True)
            upload_part.unlink(missing_ok=True)
            return video_id, "complete", target.stat().st_size, target.stat().st_size
    if relay_part.exists() and not sidecar.exists():
        return video_id, "relay_partial_skip", 0, relay_part.stat().st_size
    if upload_part.exists() and not sidecar.exists():
        return video_id, "upload_partial_skip", 0, upload_part.stat().st_size
    url = video_url(video_id, splits)
    progress = Path(str(target) + ".ranges.json")
    if sidecar.exists():
        piece, total, done = aria_state(sidecar)
    else:
        # A fresh file is handled as one range; curl writes it atomically below.
        piece, total, done = 16 * 1024 * 1024, 0, set()
        probe = subprocess.run(
            ["curl", "--fail", "--silent", "--show-error", "--location", "--head", url],
            check=True, capture_output=True, text=True,
        )
        for line in probe.stdout.splitlines():
            if line.lower().startswith("content-length:"):
                total = int(line.split(":", 1)[1].strip())
                break
        if not total:
            raise RuntimeError("no Content-Length for %s" % video_id)
    if total <= 0:
        raise RuntimeError("invalid total size for %s" % video_id)
    # ``truncate`` does not create a missing path on all supported Python
    # versions; create the sparse target explicitly before range writes.
    target.touch(exist_ok=True)
    os.truncate(target, total)
    if progress.exists():
        try:
            done.update(int(x) for x in json.loads(progress.read_text()))
        except (ValueError, TypeError, json.JSONDecodeError):
            progress.unlink(missing_ok=True)
    pieces = set(range((total + piece - 1) // piece))
    missing = sorted(pieces - done)
    if not missing:
        sidecar.unlink(missing_ok=True)
        progress.unlink(missing_ok=True)
        if expected_md5 and file_md5(target) != expected_md5:
            backup = backup_corrupt(target)
            raise RuntimeError("MD5 mismatch; preserved as %s" % backup)
        return video_id, "completed", total, total
    completed = len(done) * piece
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_piece, url, str(target), piece, total, index, timeout): index
            for index in missing
        }
        for future in as_completed(futures):
            index = future.result()
            done.add(index)
            progress.write_text(json.dumps(sorted(done)), encoding="utf-8")
            completed += min(piece, total - index * piece)
            print("%s piece=%d/%d bytes=%d/%d" % (video_id, len(done), len(pieces), completed, total), flush=True)
    sidecar.unlink(missing_ok=True)
    progress.unlink(missing_ok=True)
    if expected_md5 and file_md5(target) != expected_md5:
        backup = backup_corrupt(target)
        raise RuntimeError("MD5 mismatch; preserved as %s" % backup)
    return video_id, "completed", total, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--md5-csv", default=None)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--piece-workers", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=3600)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only", nargs="+", help="download only the listed video IDs")
    selection.add_argument(
        "--only-file",
        help="one video ID per line; IDs need not occur in the matrix manifests",
    )
    args = parser.parse_args()
    splits = read_splits(args.splits)
    expected_md5s = read_md5s(args.md5_csv)
    if args.only_file:
        ids = read_only_ids(args.only_file)
    elif args.only:
        ids = sorted(set(args.only))
    else:
        ids = read_ids(args.manifest_root)
    print("videos=%d workers=%d piece_workers=%d" % (len(ids), args.workers, args.piece_workers), flush=True)
    counts = {}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                video_id,
                args.video_root,
                splits,
                args.piece_workers,
                args.timeout,
                expected_md5s.get(video_id),
            ): video_id
            for video_id in ids
        }
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                _, status, size, covered = future.result()
            except Exception as exc:
                counts["failed"] = counts.get("failed", 0) + 1
                failures.append(video_id)
                print("%s failed error=%s" % (video_id, exc), flush=True)
                continue
            counts[status] = counts.get(status, 0) + 1
            if status.endswith("_skip"):
                failures.append(video_id)
            print("%s %s size=%d covered=%d" % (video_id, status, size, covered), flush=True)
    print("counts=%s failures=%s" % (counts, sorted(failures)), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
