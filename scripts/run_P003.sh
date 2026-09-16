#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P003.exitcode' EXIT
envs/main/bin/python -u src/run_boundary_probe.py > logs/P003_boundary_probe.log 2>&1
