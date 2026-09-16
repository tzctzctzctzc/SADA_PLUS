#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P007.exitcode' EXIT
for mode in baseline scale; do
 envs/main/bin/python -u src/run_lad_bridge.py --mode "$mode" --output "runs/P007_lad_${mode}" > "logs/P007_lad_${mode}.log" 2>&1
done
envs/main/bin/python scripts/analyze_P007.py > logs/P007_analysis.log 2>&1
