#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P029.exitcode' EXIT
test "$(cat runs/P028.exitcode)" = 0
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_clause_diagnostic.py > logs/P029_clause.log 2>&1
