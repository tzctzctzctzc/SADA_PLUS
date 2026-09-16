#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P017.exitcode' EXIT
for resolution in square336 native2048; do
 envs/main/bin/python -u src/run_lad_resolution.py --mode baseline --resolution "$resolution" --output "runs/P017_lad_$resolution" > "logs/P017_lad_$resolution.log" 2>&1
done
envs/main/bin/python scripts/analyze_P007.py --config configs/P017_lad_resolution.json > logs/P017_analysis.log 2>&1
