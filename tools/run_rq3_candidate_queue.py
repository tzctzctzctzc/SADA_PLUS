"""六shift共同候选准备：核验复用完整项，导出缺项，再在CPU上准确抽帧。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.rq3_gpu import gpu_argument, gpu_slot
from tools.run_rq3_matrix import baseline_root, completed_baseline
from tools.score_rq3_vlm_candidates import check_code

CODE_ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--gpu", type=gpu_argument, default="auto")
    parser.add_argument("--wait-gpu", action="store_true")
    parser.add_argument("--audit-only", action="store_true", help="只核验已有基线/候选，报告缺项，不导出或抽帧")
    parser.add_argument("--data-root", type=Path, default=Path("/vepfs/vlms/SADA_PLUS"))
    parser.add_argument("--baseline-parent", type=Path, default=Path("/vepfs/vlms"))
    parser.add_argument("--video-root", type=Path, default=Path("/vepfs/vlms/data/EPIC_KITCHENS/videos"))
    parser.add_argument("--recovery-root", type=Path,
                        default=Path("/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery_qwen_0924"))
    parser.add_argument("--shifts", nargs="+", choices=["S" + str(i) for i in range(1, 7)],
                        default=["S" + str(i) for i in range(1, 7)])
    args = parser.parse_args()
    for name in ("output_root", "data_root", "baseline_parent", "video_root", "recovery_root"):
        setattr(args, name, getattr(args, name).resolve())
    check_code(args.expected_head)
    os.chdir(args.data_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    logs = args.output_root / "logs"
    logs.mkdir(exist_ok=True)
    (args.output_root / "proposals").mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    env["PYTHONPATH"] = str(CODE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    def run(command, name, uuid=None):
        log = logs / (name + "." + str(time.time_ns()) + ".log")
        with log.open("x") as output:
            subprocess.run(command, cwd=args.data_root, env=dict(env, CUDA_VISIBLE_DEVICES=uuid or ""),
                           stdout=output, stderr=subprocess.STDOUT, check=True)

    import fcntl
    statuses = {}
    with (args.data_root / ".rq3_candidates.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for shift in args.shifts:
            check_code(args.expected_head)
            baseline = baseline_root(args.baseline_parent, shift)
            _, checkpoint = completed_baseline(baseline, shift)
            print("BASELINE_VERIFIED", shift, flush=True)
            complete = json.loads((baseline / "complete.json").read_text())
            proposals = args.output_root / "proposals" / (shift + ".jsonl")
            meta = Path(str(proposals) + ".meta.json")
            destination = args.output_root / shift
            if proposals.exists() != meta.exists() or (destination.exists() and not proposals.exists()):
                raise ValueError("partial candidate/proposal output preserved; audit required: " + shift)
            if args.audit_only and not (destination / "manifest.json").is_file():
                if destination.exists():
                    raise ValueError("partial frame output preserved; audit required: " + shift)
                statuses[shift] = "missing-candidates"
                print("MISSING_CANDIDATES", shift, flush=True)
                continue
            if not proposals.exists():
                with gpu_slot(args.gpu, args.data_root, args.wait_gpu) as selected_gpu:
                    check_code(args.expected_head)
                    command = [sys.executable, "-B", str(CODE_ROOT / "tools/export_vlm_target_adapter_proposals.py"),
                               "--config", complete["so"], "--checkpoint", str(checkpoint),
                               "--trusted-matched-baseline-root", str(baseline),
                               "--video-root", str(args.video_root), "--include-missing-videos",
                               "--output", str(proposals), "--domain", "target", "--split", "train",
                               "--device", "cuda:0", "--weights", "ema", "--levels", "0", "1", "2",
                               "--max-per-video", "30", "--max-proposals", "200", "--seed", "6"]
                    print("START_PROPOSALS", shift, flush=True)
                    run(command, shift + ".proposals", selected_gpu["uuid"])
            check_code(args.expected_head)
            command = [sys.executable, "-B", str(CODE_ROOT / "tools/prepare_rq3_candidates.py"),
                       "--proposals", str(proposals), "--baseline-root", str(baseline), "--shift", shift,
                       "--video-root", str(args.video_root), "--recovery-root", str(args.recovery_root),
                       "--output-root", str(destination), "--reuse-verified"]
            run(command, shift + ".frames")
            statuses[shift] = "verified"
            print("CANDIDATES_VERIFIED", shift, flush=True)
    print(json.dumps({"audit_only": args.audit_only, "shifts": statuses,
                      "all_candidates_ready": all(value == "verified" for value in statuses.values())}), flush=True)


if __name__ == "__main__":
    main()
