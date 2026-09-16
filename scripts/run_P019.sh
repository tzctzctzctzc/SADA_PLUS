#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P019.exitcode' EXIT
envs/main/bin/python -u src/run_backend_diagnostic.py > logs/P019_backend_diagnostic.log 2>&1
