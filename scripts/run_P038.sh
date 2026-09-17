#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
rm -f runs/P038.exitcode
trap 'rc=$?; printf "%s\n" "$rc" > runs/P038.exitcode' EXIT
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_P038.py > logs/P038_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P038.py > logs/P038_analysis.log 2>&1
