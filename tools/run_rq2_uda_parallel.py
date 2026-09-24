"""Run independent formal RQ2 UDA experiments on one assigned GPU."""

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path


SPECS = [(shift, seed) for seed in (6, 7, 8) for shift in range(1, 7)]
GPU_BY_MODEL = {"egovideo_hod": 0, "internvideo2_clip_1b": 1}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def result_dir(root, model, shift, seed):
    return root / model / f"S{shift}_seed{seed}"


def checked_existing(root, model, shift, seed):
    output = result_dir(root, model, shift, seed)
    run_name = f"rq2_uda_{model}_S{shift}_seed{seed}_sada"
    checkpoint_root = Path("/vepfs/vlms/SADA_PLUS/model_results") / run_name
    if output.exists():
        required = ("manifest.json", "source_val.json", "target_val.json")
        if not all((output / filename).is_file() and (output / filename).stat().st_size for filename in required):
            raise RuntimeError(f"incomplete UDA output: {output}")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("shift") != f"S{shift}" or manifest.get("seed") != seed:
            raise RuntimeError(f"UDA output identity mismatch: {output}")
        for filename in ("source_val.json", "target_val.json"):
            digest = hashlib.sha256((output / filename).read_bytes()).hexdigest()
            if manifest.get("outputs", {}).get(filename) != digest:
                raise RuntimeError(f"UDA output hash mismatch: {output / filename}")
        return True
    if checkpoint_root.exists():
        raise RuntimeError(f"training exists without evaluated output: {checkpoint_root}")
    return False


def run_one(script, model, gpu, root, revision, shift, seed):
    spec = f"S{shift}:{seed}"
    log = root / "logs" / f"{model}_{spec.replace(':', '_')}.launcher.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp()} start {spec} gpu={gpu} revision={revision}\n")
        stream.flush()
        result = subprocess.run(
            ["bash", str(script), model, str(gpu), str(root), revision, "launch", spec],
            stdout=stream, stderr=subprocess.STDOUT, check=False,
        )
        stream.write(f"{timestamp()} exit {spec} code={result.returncode}\n")
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=GPU_BY_MODEL, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.gpu != GPU_BY_MODEL[args.model] or not 1 <= args.workers <= 8:
        raise ValueError("invalid RQ2 model/GPU pair or worker count")

    script = Path(__file__).with_name("run_rq2_uda_queue.sh").resolve()
    root = args.artifact_root.resolve()
    root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    if not checked_existing(root, args.model, 1, 6):
        raise RuntimeError("S1 seed6 recovery evaluation must finish first")
    pending_specs = [
        (shift, seed) for shift, seed in SPECS if (shift, seed) != (1, 6)
        and not checked_existing(root, args.model, shift, seed)
    ]
    print(f"{timestamp()} {args.model} pending={len(pending_specs)} workers={args.workers}", flush=True)
    remaining = iter(pending_specs)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        running = {}

        def submit_next():
            try:
                shift, seed = next(remaining)
            except StopIteration:
                return False
            future = executor.submit(
                run_one, script, args.model, args.gpu, root, args.expected_head, shift, seed
            )
            running[future] = (shift, seed)
            print(f"{timestamp()} start S{shift} seed={seed}", flush=True)
            return True

        for _ in range(args.workers):
            if not submit_next():
                break
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            failures = []
            for future in done:
                shift, seed = running.pop(future)
                try:
                    code = future.result()
                    if code or not checked_existing(root, args.model, shift, seed):
                        failures.append(f"S{shift} seed={seed} exit={code}")
                    else:
                        print(f"{timestamp()} complete S{shift} seed={seed}", flush=True)
                except Exception as exc:
                    failures.append(f"S{shift} seed={seed}: {exc}")
            if failures:
                raise RuntimeError("UDA queue stopped: " + "; ".join(failures))
            while len(running) < args.workers and submit_next():
                pass
    print(f"{timestamp()} {args.model} complete 18/18", flush=True)


if __name__ == "__main__":
    main()
