#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P004.exitcode' EXIT
envs/main/bin/python -u src/run_multi_image.py --mode canonical_start --output runs/P004_mantis_7b_canonical_start > logs/P004_mantis_7b_canonical_start.log 2>&1
for mode in baseline scale; do
 envs/main/bin/python -u src/run_multi_image.py --mode "$mode" --model-name Qwen2.5-VL-3B-Instruct --output "runs/P004_mantis_3b_${mode}" > "logs/P004_mantis_3b_${mode}.log" 2>&1
done
