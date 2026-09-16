#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P028.exitcode' EXIT
dependency_pid="$1"
if kill -0 "$dependency_pid" 2>/dev/null; then
  printf 'Waiting for P027 PID %s\n' "$dependency_pid"
  tail --pid="$dependency_pid" -f /dev/null
fi
test "$(cat runs/P027.exitcode)" = 0
exec 9>runs/gpu3.lock
flock 9
envs/qwen3/bin/python -u src/run_qwen3_resolution.py > logs/P028_qwen3.log 2>&1
envs/main/bin/python scripts/analyze_P028.py > logs/P028_analysis.log 2>&1
