"""Upload verified EPIC videos to an isolated remote workspace with resume."""

import argparse
import csv
import os
import posixpath
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko

from tools.run_clip_target_adapter import file_md5, video_path
from tools.run_clip_then_llava import read_jsonl


SHIFT_NAMES = ("S1", "S2", "S3", "S4", "S5", "S6")


def manifest_video_ids(scenario_root, shifts):
    ids = set()
    root = Path(scenario_root)
    for shift in shifts:
        for filename in ("source_val_manifest.jsonl", "target_train_manifest.jsonl"):
            for row in read_jsonl(root / shift / filename):
                ids.add(row["video_id"])
    return sorted(ids)


def video_md5s(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return {
            Path(row[1]).stem: row[0].lower()
            for row in csv.reader(stream)
            if len(row) >= 2 and row[1].lower().endswith(".mp4")
        }


class UploadWorker:
    def __init__(self, args, expected_md5s):
        self.args = args
        self.expected_md5s = expected_md5s
        self.print_lock = threading.Lock()

    def _connect(self):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.args.host,
            port=self.args.port,
            username=self.args.user,
            key_filename=os.path.expanduser(self.args.key_file),
            look_for_keys=False,
            allow_agent=False,
            timeout=60,
            banner_timeout=60,
            auth_timeout=60,
        )
        return client

    def upload_one(self, video_id):
        source = video_path(self.args.video_root, video_id)
        if source is None:
            return video_id, "missing", 0, None
        source = Path(source)
        if Path(str(source) + ".aria2").exists():
            return video_id, "incomplete", 0, None

        expected = self.expected_md5s.get(video_id)
        if expected is None:
            return video_id, "missing_md5", 0, None
        if file_md5(source) != expected:
            return video_id, "bad_md5", 0, None

        remote_path = posixpath.join(
            self.args.remote_dir.rstrip("/"), video_id + ".MP4"
        )
        partial = remote_path + ".upload.part"
        local_size = source.stat().st_size
        client = self._connect()
        try:
            sftp = client.open_sftp()
            try:
                try:
                    if sftp.stat(remote_path).st_size == local_size:
                        return video_id, "cached", local_size, None
                except IOError:
                    pass

                try:
                    remote_size = sftp.stat(partial).st_size
                except IOError:
                    remote_size = 0
                if remote_size > local_size:
                    sftp.remove(partial)
                    remote_size = 0

                with self.print_lock:
                    print(
                        "start {} {:.2f}/{:.2f} GiB".format(
                            video_id, remote_size / 1024 ** 3, local_size / 1024 ** 3
                        ),
                        flush=True,
                    )
                with source.open("rb") as local_file:
                    local_file.seek(remote_size)
                    mode = "ab" if remote_size else "wb"
                    with sftp.open(partial, mode) as remote_file:
                        remote_file.set_pipelined(True)
                        while True:
                            chunk = local_file.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            remote_file.write(chunk)

                if sftp.stat(partial).st_size != local_size:
                    raise RuntimeError("remote size mismatch after upload")
                try:
                    sftp.remove(remote_path)
                except IOError:
                    pass
                sftp.rename(partial, remote_path)
                return video_id, "uploaded", local_size, None
            finally:
                sftp.close()
        except Exception as error:
            return video_id, "failed", 0, "{}: {}".format(type(error).__name__, error)
        finally:
            client.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--md5-csv", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES))
    parser.add_argument("--video-ids", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    required = manifest_video_ids(args.scenario_root, args.shifts)
    if args.video_ids:
        requested = set(args.video_ids)
        unknown = sorted(requested.difference(required))
        if unknown:
            raise ValueError("video IDs absent from manifests: {}".format(unknown))
        required = sorted(requested)

    worker = UploadWorker(args, video_md5s(args.md5_csv))
    counts = {}
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker.upload_one, video_id) for video_id in required]
        for number, future in enumerate(as_completed(futures), 1):
            video_id, status, size, error = future.result()
            counts[status] = counts.get(status, 0) + 1
            total_bytes += size
            message = "{}/{} {} {}".format(number, len(futures), video_id, status)
            if error:
                message += ": " + error
            print(message, flush=True)
    print(
        "counts={} transferred_or_cached={:.2f} GiB".format(
            counts, total_bytes / 1024 ** 3
        ),
        flush=True,
    )
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
