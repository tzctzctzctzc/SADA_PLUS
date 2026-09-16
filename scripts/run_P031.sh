#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P031.exitcode' EXIT
while kill -0 512794 2>/dev/null; do sleep 20; done
test "$(cat runs/P030.exitcode)" = "0"
while kill -0 513275 2>/dev/null; do sleep 20; done
test "$(cat runs/Coral_download.exitcode)" = "0"
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_coral_transfer.py > logs/P031_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P031.py > logs/P031_analysis.log 2>&1
