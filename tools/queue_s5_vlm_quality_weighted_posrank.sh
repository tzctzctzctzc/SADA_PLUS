#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
artifact_root=${SADA_QUALITY_ARTIFACT_ROOT:-$root/artifacts/vlm_quality_weighting_20260912}
checkpoint=${SADA_PARENT:-$artifact_root/checkpoint-epoch10-seed123.pt}
launcher_log=$artifact_root/logs/quality_posrank_launcher.log
mkdir -p "$artifact_root/logs" "$artifact_root/pids"

while [[ ! -f "$checkpoint" ]]; do
    sleep 20
done

while true; do
    gpu=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | awk -F', ' '$2 < 1024 && $3 < 10 {print $1; exit}')
    if [[ -n "$gpu" ]]; then
        echo "$(date -Iseconds) start on GPU $gpu" >>"$launcher_log"
        cd "$root"
        GPU="$gpu" bash tools/run_s5_vlm_quality_weighted_posrank.sh \
            >>"$launcher_log" 2>&1
        exit $?
    fi
    sleep 60
done
