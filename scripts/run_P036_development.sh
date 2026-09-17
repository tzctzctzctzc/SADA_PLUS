#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
rm -f runs/P036_development.exitcode
trap 'rc=$?; printf "%s\n" "$rc" > runs/P036_development.exitcode' EXIT
envs/main/bin/python -u scripts/fetch_P036_images.py development > logs/P036_development_fetch.log 2>&1
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_P036.py development > logs/P036_development_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P036.py development > logs/P036_development_analysis.log 2>&1
