#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P008.exitcode' EXIT
for size in 3B 7B; do
 envs/main/bin/python -u src/run_protocol_cross.py --model-name "Qwen2.5-VL-${size}-Instruct" --output "runs/P008_protocol_${size}" > "logs/P008_protocol_${size}.log" 2>&1
done
envs/main/bin/python scripts/analyze_P008.py > logs/P008_analysis.log 2>&1
