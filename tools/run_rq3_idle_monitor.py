"""RQ3续跑监控：连续空闲至少60分钟后自动训练，最多两张物理GPU。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tools.run_rq3_matrix import CODE_ROOT, completed_entries, matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--completed-root", type=Path, action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=Path("/vepfs/vlms/SADA_PLUS"))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--idle-minutes", type=float, default=60)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.idle_minutes < 60:
        parser.error("GPU必须连续空闲至少60分钟")
    args.completed_root = [p.resolve() for p in args.completed_root]
    completed = completed_entries(args.completed_root)
    remaining = [entry for entry in matrix() if entry not in completed]
    print("RQ3_IDLE_MONITOR_PLAN completed=%d remaining=%d max_gpus=2 idle_minutes=%g" %
          (len(completed), len(remaining), args.idle_minutes), flush=True)
    if not args.execute or not remaining:
        return
    from tools.run_rq2_matched_control import check_code
    import fcntl
    check_code(args.expected_head)
    args.output_root = args.output_root.resolve()
    args.data_root = args.data_root.resolve()
    children = []
    streams = []
    stopping = False

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with (args.data_root / ".rq3_idle_monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.output_root.mkdir(parents=True, exist_ok=False)
        try:
            for shard in range(2):
                root = args.output_root / (args.output_root.name + "_shard%d" % shard)
                command = [sys.executable, "-B", "-m", "tools.run_rq3_matrix",
                           "--cache-root", str(args.cache_root.resolve()),
                           "--output-root", str(root), "--expected-head", args.expected_head,
                           "--data-root", str(args.data_root),
                           "--shard-count", "2", "--shard-index", str(shard),
                           "--gpu", "auto", "--wait-gpu", "--execute",
                           "--gpu-idle-minutes", str(args.idle_minutes)]
                for completed_root in args.completed_root:
                    command.extend(["--completed-root", str(completed_root)])
                stream = (args.output_root / ("shard%d.log" % shard)).open("x")
                streams.append(stream)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                           PYTHONUNBUFFERED="1", PYTHONPATH=str(CODE_ROOT))
                children.append(subprocess.Popen(command, cwd=CODE_ROOT, env=env,
                                                 stdout=stream, stderr=subprocess.STDOUT,
                                                 start_new_session=True))
            state = {"monitor_pid": os.getpid(), "worker_pids": [c.pid for c in children],
                     "idle_minutes": args.idle_minutes, "max_gpus": 2,
                     "completed_before_start": len(completed), "remaining": len(remaining),
                     "expected_head": args.expected_head, "status": "running"}
            state_path = args.output_root / "monitor_state.json"
            state_path.write_text(json.dumps(state, sort_keys=True))
            print("RQ3_IDLE_MONITOR_STARTED " + json.dumps(state, sort_keys=True), flush=True)
            while any(c.poll() is None for c in children):
                time.sleep(5)
            codes = [c.returncode for c in children]
            state.update(status="stopped" if stopping else "complete" if codes == [0, 0] else "failed",
                         exit_codes=codes)
            state_path.write_text(json.dumps(state, sort_keys=True))
            print("RQ3_IDLE_MONITOR_FINISHED " + json.dumps(state, sort_keys=True), flush=True)
            if not stopping and codes != [0, 0]:
                raise SystemExit(1)
        finally:
            stop(None, None)
            for child in children:
                child.wait()
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    main()
