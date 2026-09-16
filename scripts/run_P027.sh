#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/P027.exitcode' EXIT
evaluation_pid="$1"
environment_pid="$2"
for dependency_pid in "$evaluation_pid" "$environment_pid"; do
  if kill -0 "$dependency_pid" 2>/dev/null; then
    printf 'Waiting for prerequisite PID %s\n' "$dependency_pid"
    tail --pid="$dependency_pid" -f /dev/null
  fi
done
test "$(cat runs/P025.exitcode)" = 0
test "$(cat runs/qwen3_env_setup.exitcode)" = 0
test "$(cat runs/Qwen3_Reranker_download.exitcode)" = 0
exec 9>runs/gpu3.lock
flock 9
envs/qwen3/bin/python -u src/run_qwen3_imagecode.py > logs/P027_qwen3.log 2>&1
envs/main/bin/python scripts/analyze_P027.py > logs/P027_analysis.log 2>&1
