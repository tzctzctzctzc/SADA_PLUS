#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/Qwen3_Reranker_download.exitcode' EXIT
envs/io/bin/python -u scripts/fetch_manifest.py --manifest literature/Qwen3-VL-Reranker-2B_manifest.json --destination models/Qwen3-VL-Reranker-2B > logs/Qwen3_Reranker_download.log 2>&1
