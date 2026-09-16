#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P024.exitcode' EXIT
envs/main/bin/python -u src/run_pointwise_controls.py > logs/P024_pointwise.log 2>&1
envs/main/bin/python scripts/analyze_P024.py > logs/P024_analysis.log 2>&1
