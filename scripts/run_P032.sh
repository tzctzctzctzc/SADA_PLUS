#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P032.exitcode' EXIT
while kill -0 512794 2>/dev/null; do sleep 20; done
test "$(cat runs/P030.exitcode)" = "0"
envs/main/bin/python scripts/analyze_P032_reference_contrast.py > logs/P032_analysis.log 2>&1
