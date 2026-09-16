#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P011.exitcode' EXIT
envs/main/bin/python -u src/run_geometry_transport.py > logs/P011_geometry_transport.log 2>&1
envs/main/bin/python scripts/analyze_P011.py > logs/P011_analysis.log 2>&1
