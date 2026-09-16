#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P023.exitcode' EXIT
download_pid="$1"
if kill -0 "$download_pid" 2>/dev/null; then
  printf 'Waiting for ImageCoDe download PID %s\n' "$download_pid"
  tail --pid="$download_pid" -f /dev/null
fi
test "$(cat runs/ImageCoDe_download.exitcode)" = 0
envs/main/bin/python scripts/extract_P023.py > logs/P023_data.log 2>&1
exec 9>runs/gpu3.lock
flock 9
envs/main/bin/python -u src/run_imagecode_probe.py > logs/P023_vlm.log 2>&1
envs/main/bin/python -u src/run_imagecode_siglip.py > logs/P023_siglip.log 2>&1
envs/main/bin/python scripts/analyze_P023.py > logs/P023_analysis.log 2>&1
