#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P014.exitcode' EXIT
envs/main/bin/python -u src/run_image_isolation.py > logs/P014_image_isolation.log 2>&1
envs/main/bin/python scripts/analyze_P014.py > logs/P014_analysis.log 2>&1
