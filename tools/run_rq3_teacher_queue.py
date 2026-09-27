"""逐shift真实教师打分并冻结缓存；等待共同候选，最多占用共享两卡额度。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.build_rq3_cache import build_cache
from tools.rq3_protocol import MODELS, content_hash, load_cache, validate_interface_audit, audit_cache_predictions
from tools.score_rq3_vlm_candidates import check_code

CODE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--shifts", nargs="+", choices=["S" + str(i) for i in range(1, 7)],
                        default=["S" + str(i) for i in range(1, 7)])
    parser.add_argument("--smoke-first", action="store_true")
    args = parser.parse_args()
    for key in ("settings", "frozen_protocol", "candidate_root", "teacher_root", "cache_root"):
        setattr(args, key, getattr(args, key).resolve())
    check_code(args.expected_head)
    root = args.teacher_root / args.model
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    import fcntl
    with (root / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for shift in args.shifts:
            manifest_path = args.candidate_root / shift / "manifest.json"
            announced = False
            while True:
                check_code(args.expected_head)
                try:
                    manifest = json.loads(manifest_path.read_text())
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    if not announced:
                        print("WAIT_COMMON_CANDIDATES", args.model, shift, flush=True)
                        announced = True
                    time.sleep(30)
            base = [sys.executable, "-B", "-m", "tools.score_rq3_vlm_candidates",
                    "--model", args.model, "--settings", str(args.settings),
                    "--frozen-protocol", str(args.frozen_protocol), "--manifest", str(manifest_path),
                    "--frame-root", str(manifest_path.parent / "frames"),
                    "--expected-head", args.expected_head, "--gpu", "auto", "--wait-gpu"]

            def score(output, full):
                if (output / "interface_audit.json").exists():
                    return
                command = base + ["--output-root", str(output)] + (["--full"] if full else [])
                log = logs / (shift + (".full." if full else ".smoke.") + str(time.time_ns()) + ".log")
                print("START_TEACHER", args.model, shift, "full" if full else "smoke", flush=True)
                with log.open("x") as stream:
                    subprocess.run(command, cwd=CODE_ROOT,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1"),
                                   stdout=stream, stderr=subprocess.STDOUT, check=True)

            if args.smoke_first and shift == args.shifts[0]:
                smoke = root / (shift + "_smoke")
                score(smoke, False)
                validate_interface_audit(json.loads((smoke / "input_protocol.json").read_text()), args.model)
                print("REAL_INTERFACE_SMOKE_PASS", args.model, flush=True)
            output = root / shift
            cache = args.cache_root / args.model / (shift + ".json")
            if not cache.exists():
                score(output, True)
                protocol = json.loads((output / "input_protocol.json").read_text())
                validate_interface_audit(protocol, args.model)
                predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
                payload = build_cache(manifest, predictions, protocol, args.model, shift,
                                      manifest["teacher_checkpoint_sha256"])
                cache.parent.mkdir(parents=True, exist_ok=True)
                with cache.open("x") as stream:
                    json.dump(payload, stream, sort_keys=True)
            payload, _ = load_cache(cache, args.model, shift, manifest["target_video_ids"])
            audit = validate_interface_audit(payload["input_protocol"], args.model)
            audit_cache_predictions(payload, audit)
            if payload["candidate_sha256"] != content_hash(manifest):
                raise ValueError("teacher cache differs from current common candidates")
            print("FULL_CACHE_VERIFIED", args.model, shift, len(payload["entries"]), flush=True)
    print("TEACHER_SIX_SHIFT_COMPLETE", args.model, flush=True)


if __name__ == "__main__":
    main()
