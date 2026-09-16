#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P025.exitcode' EXIT
dependency_pid="$1"
if kill -0 "$dependency_pid" 2>/dev/null; then
 printf 'Waiting for P023 PID %s\n' "$dependency_pid"
 tail --pid="$dependency_pid" -f /dev/null
fi
test "$(cat runs/P023.exitcode)" = 0
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_pointwise_imagecode.py > logs/P025_pointwise.log 2>&1
envs/main/bin/python scripts/analyze_P025.py > logs/P025_analysis.log 2>&1
