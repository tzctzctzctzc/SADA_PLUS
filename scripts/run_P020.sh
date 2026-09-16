#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
exec 9>runs/gpu3.lock
flock 9
trap 'rc=$?; printf "%s\n" "$rc" > runs/P020.exitcode' EXIT
envs/main/bin/python -u src/run_siglip_retrieval.py > logs/P020_siglip_retrieval.log 2>&1
envs/main/bin/python scripts/analyze_P020.py > logs/P020_analysis.log 2>&1
