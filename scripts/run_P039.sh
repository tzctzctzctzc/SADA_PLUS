#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
export CUDA_VISIBLE_DEVICES=GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f
exec 9>runs/gpu1.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P039.exitcode' EXIT
envs/main/bin/python -u src/run_P039.py > logs/P039_shadow_binding_bridges.log 2>&1
envs/main/bin/python scripts/analyze_P039.py > logs/P039_analysis.log 2>&1
