#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P030.exitcode' EXIT
dependency_pid="$1"
if kill -0 "$dependency_pid" 2>/dev/null; then
  printf 'Waiting for image downloader PID %s\n' "$dependency_pid"
  tail --pid="$dependency_pid" -f /dev/null
fi
test "$(cat runs/P030_images.exitcode)" = 0
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_pinpoint_references.py > logs/P030_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P030.py > logs/P030_analysis.log 2>&1
