#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P013.exitcode' EXIT
envs/main/bin/python -u src/run_cache_paths.py > logs/P013_cache_paths.log 2>&1
envs/main/bin/python scripts/analyze_P013.py > logs/P013_analysis.log 2>&1
