#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P009.exitcode' EXIT
envs/main/bin/python -u src/run_boundary_restore.py > logs/P009_boundary_restore.log 2>&1
envs/main/bin/python scripts/analyze_P009.py > logs/P009_analysis.log 2>&1
