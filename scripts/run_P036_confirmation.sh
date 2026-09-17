#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
rm -f runs/P036_confirmation.exitcode
trap 'rc=$?; printf "%s\n" "$rc" > runs/P036_confirmation.exitcode' EXIT
envs/main/bin/python - <<'PY'
import json
result=json.load(open('results/P036_development.json'))
assert result['continuation_gate_evaluation']['all_pass']
PY
envs/main/bin/python -u scripts/fetch_P036_images.py confirmation > logs/P036_confirmation_fetch.log 2>&1
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_P036.py confirmation > logs/P036_confirmation_scoring.log 2>&1
envs/main/bin/python scripts/analyze_P036.py confirmation > logs/P036_confirmation_analysis.log 2>&1
