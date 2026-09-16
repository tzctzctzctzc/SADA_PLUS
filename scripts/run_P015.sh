#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P015.exitcode' EXIT
for mode in baseline scale; do
 envs/main/bin/python -u src/run_lad_bridge.py --config configs/P015_lad_bridge.json --mode "$mode" --output "runs/P015_lad_${mode}" > "logs/P015_lad_${mode}.log" 2>&1
done
envs/main/bin/python scripts/analyze_P007.py --config configs/P015_lad_bridge.json > logs/P015_analysis.log 2>&1
