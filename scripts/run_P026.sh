#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P026.exitcode' EXIT
dependency_pid="$1"
if kill -0 "$dependency_pid" 2>/dev/null; then
  printf 'Waiting for P025 PID %s\n' "$dependency_pid"
  tail --pid="$dependency_pid" -f /dev/null
fi
test "$(cat runs/P025.exitcode)" = 0
envs/main/bin/python scripts/analyze_P026.py > logs/P026_analysis.log 2>&1
