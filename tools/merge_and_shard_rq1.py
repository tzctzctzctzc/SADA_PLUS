"""Merge completed RQ1 predictions and shard newly ready segments."""

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def frames_complete(root, candidate_id, frame_count):
    directory = Path(root) / candidate_id
    return all(
        (directory / "frame_{:02d}.jpg".format(index)).is_file()
        and (directory / "frame_{:02d}.jpg".format(index)).stat().st_size > 0
        for index in range(1, frame_count + 1)
    )


def merge_rows(paths):
    merged = {}
    for path in paths:
        for row in read_jsonl(path):
            candidate_id = row.get("candidate_id")
            if row.get("status") == "ok" and candidate_id:
                merged[candidate_id] = row
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-root")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--exclude-video-id", action="append", default=[])
    parser.add_argument("--shard-prefix")
    parser.add_argument("--num-shards", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    by_id = merge_rows(args.input)
    ordered = [by_id[row["candidate_id"]] for row in manifest if row["candidate_id"] in by_id]
    write_jsonl_atomic(args.output, ordered)

    pending = []
    if args.num_shards:
        if not args.frame_root or not args.shard_prefix:
            raise ValueError("frame-root and shard-prefix are required with num-shards")
        excluded = set(args.exclude_video_id)
        pending = [
            row for row in manifest
            if row["candidate_id"] not in by_id
            and row["video_id"] not in excluded
            and frames_complete(args.frame_root, row["candidate_id"], args.frame_count)
        ]
        for shard in range(args.num_shards):
            rows = [row for index, row in enumerate(pending) if index % args.num_shards == shard]
            write_jsonl_atomic("{}_{}.jsonl".format(args.shard_prefix, shard), rows)

    print(json.dumps({
        "manifest_rows": len(manifest),
        "merged_rows": len(ordered),
        "ready_pending_rows": len(pending),
        "num_shards": args.num_shards,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
