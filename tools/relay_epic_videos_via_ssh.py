"""Download EPIC videos on the fast remote host and relay them to local disk."""

import argparse
import json
import os
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import paramiko

from tools.run_clip_target_adapter import epic_55_splits, epic_video_url, video_path
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


class RelayWorker:
    def __init__(self, args, old_splits):
        self.args = args
        self.old_splits = old_splits
        self.local_lock = threading.Lock()

    def _connect(self):
        password = os.environ.get("EPIC_RELAY_PASSWORD")
        if not password:
            raise RuntimeError("EPIC_RELAY_PASSWORD is not set")
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            self.args.host,
            port=self.args.port,
            username=self.args.user,
            password=password,
            look_for_keys=False,
            allow_agent=False,
            timeout=60,
            banner_timeout=60,
            auth_timeout=60,
        )
        return client

    def download_one(self, video_id):
        destination = Path(self.args.video_root) / (video_id + ".MP4")
        partial = Path(str(destination) + ".relay.part")
        remote_name = video_id + ".MP4"
        remote_dir = self.args.remote_dir.rstrip("/")
        remote_path = remote_dir + "/" + remote_name
        remote_control = remote_path + ".aria2"
        url = epic_video_url(video_id, self.old_splits)

        client = self._connect()
        try:
            command = "mkdir -p {directory}; if [ ! -s {path} ] || [ -e {control} ]; then aria2c --allow-overwrite=true --auto-file-renaming=false --file-allocation=none --continue=true --timeout=60 --connect-timeout=20 --max-tries=2 --retry-wait=5 --max-connection-per-server={connections} --split={connections} --min-split-size=1M --console-log-level=warn --dir {directory} --out {name} {url}; fi".format(
                directory=shlex.quote(remote_dir),
                path=shlex.quote(remote_path),
                control=shlex.quote(remote_control),
                connections=self.args.connections,
                name=shlex.quote(remote_name),
                url=shlex.quote(url),
            )
            _, stdout, stderr = client.exec_command(command, timeout=self.args.remote_timeout)
            exit_code = stdout.channel.recv_exit_status()
            if exit_code != 0:
                message = stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError("remote aria2 exit {}: {}".format(exit_code, message))

            sftp = client.open_sftp()
            try:
                remote_size = sftp.stat(remote_path).st_size
                if remote_size <= 0:
                    raise RuntimeError("remote file is empty: {}".format(remote_path))
                try:
                    sftp.stat(remote_control)
                except IOError:
                    pass
                else:
                    raise RuntimeError("remote aria2 file is incomplete: {}".format(remote_path))
                local_size = partial.stat().st_size if partial.exists() else 0
                if local_size > remote_size:
                    partial.unlink()
                    local_size = 0
                with sftp.open(remote_path, "rb") as remote_file:
                    remote_file.seek(local_size)
                    remote_file.prefetch(remote_size)
                    with partial.open("ab") as local_file:
                        while True:
                            chunk = remote_file.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            local_file.write(chunk)
            finally:
                sftp.close()

            local_size = partial.stat().st_size
            if local_size != remote_size:
                raise RuntimeError(
                    "size mismatch for {}: local={} remote={}".format(
                        video_id, local_size, remote_size
                    )
                )
            os.replace(partial, destination)
            control = Path(str(destination) + ".aria2")
            if control.exists():
                control.unlink()

            cleanup = "rm -f {} {}".format(
                shlex.quote(remote_path), shlex.quote(remote_control)
            )
            _, cleanup_stdout, cleanup_stderr = client.exec_command(cleanup)
            if cleanup_stdout.channel.recv_exit_status() != 0:
                raise RuntimeError(
                    "remote cleanup failed: {}".format(
                        cleanup_stderr.read().decode("utf-8", errors="replace")
                    )
                )
            return video_id, remote_size, None
        except Exception as error:
            if partial.exists():
                partial.unlink()
            return video_id, 0, "{}: {}".format(type(error).__name__, error)
        finally:
            client.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--epic55-splits", required=True)
    parser.add_argument("--shifts", nargs="+", choices=SHIFT_NAMES, default=list(SHIFT_NAMES))
    parser.add_argument("--host", default="100.119.228.59")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="gyy_dlib")
    parser.add_argument("--remote-dir", default="/home1/gyy/vlms/epic_video_relay_tmp")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--remote-timeout", type=int, default=21600)
    parser.add_argument("--limit-videos", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers <= 0 or args.connections <= 0:
        raise ValueError("workers and connections must be positive")
    Path(args.video_root).mkdir(parents=True, exist_ok=True)
    old_splits = epic_55_splits(args.epic55_splits)
    all_ids = manifest_video_ids(args.scenario_root, args.shifts)
    missing = []
    for video_id in all_ids:
        existing = video_path(args.video_root, video_id)
        if existing is None or Path(str(existing) + ".aria2").exists():
            missing.append(video_id)
    if args.limit_videos > 0:
        missing = missing[: args.limit_videos]
    print(
        "required={} missing={} workers={} connections={}".format(
            len(all_ids), len(missing), args.workers, args.connections
        ),
        flush=True,
    )
    worker = RelayWorker(args, old_splits)
    failed = []
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker.download_one, video_id) for video_id in missing]
        for number, future in enumerate(as_completed(futures), 1):
            video_id, size, error = future.result()
            if error:
                failed.append(video_id)
                print(
                    "{}/{} {} failed: {}".format(number, len(futures), video_id, error),
                    flush=True,
                )
            else:
                total_bytes += size
                print(
                    "{}/{} {} relayed {:.2f} GiB (total {:.2f} GiB)".format(
                        number, len(futures), video_id, size / 1024 ** 3, total_bytes / 1024 ** 3
                    ),
                    flush=True,
                )
    print(
        "finished={} failed={} relayed={:.2f} GiB".format(
            len(missing) - len(failed), len(failed), total_bytes / 1024 ** 3
        ),
        flush=True,
    )
    if failed:
        print(json.dumps({"failed": failed}, ensure_ascii=False), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
