#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P005.exitcode' EXIT
for mode in baseline scale; do
 envs/main/bin/python -u src/run_multi_image.py --dataset muirbench --split dev --mode "$mode" --output "runs/P005_muir_dev_${mode}" > "logs/P005_muir_dev_${mode}.log" 2>&1
done
