#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P006.exitcode' EXIT
for mode in baseline scale canonical_start; do
 envs/main/bin/python -u src/run_permutation_probe.py --mode "$mode" --output "runs/P006_permutation_${mode}" > "logs/P006_permutation_${mode}.log" 2>&1
done
