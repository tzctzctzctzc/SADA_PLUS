#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock -n 9 || exit 75
trap 'rc=$?; printf "%s\n" "$rc" > runs/P002.exitcode' EXIT
for mode in baseline scale; do
 envs/main/bin/python -u src/run_mantis.py --mode "$mode" --output "runs/P002_mantis_${mode}" > "logs/P002_mantis_${mode}.log" 2>&1
done
