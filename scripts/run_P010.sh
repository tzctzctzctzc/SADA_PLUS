#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P010.exitcode' EXIT
envs/main/bin/python -u src/run_muir_protocol.py > logs/P010_muir_protocol.log 2>&1
envs/main/bin/python scripts/analyze_P010.py > logs/P010_analysis.log 2>&1
