#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s
" "$rc" > runs/P035.exitcode' EXIT
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_source_bound_predicates.py > logs/P035_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P035.py > logs/P035_analysis.log 2>&1
