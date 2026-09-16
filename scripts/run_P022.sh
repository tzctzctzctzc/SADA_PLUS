#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P022.exitcode' EXIT
envs/main/bin/python -u src/run_raw_score_validation.py > logs/P022_validation.log 2>&1
