"""Merge resumable RQ2 shards in manifest order."""

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    merged = {}
    for path in args.input:
        for row in read_jsonl(path):
            if row.get("candidate_id") and row.get("status") in ("ok", "invalid"):
                merged[row["candidate_id"]] = row
    ordered = [merged[row["candidate_id"]] for row in manifest if row["candidate_id"] in merged]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"manifest_rows": len(manifest), "merged_rows": len(ordered)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
