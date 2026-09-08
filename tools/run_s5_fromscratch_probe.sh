#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
config=configs/old_vs_new/version_final_ckpt.yaml
cache_root=artifacts/s5_strategy_matrix/caches
artifact_root=artifacts/s5_fromscratch_20260908
IFS=',' read -r -a gpus <<<"${GPUS:-0,2,3,4}"

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 4 )); then
    echo "GPUS must contain one to four comma-separated GPU ids" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

tasks=(
    "control 123"
    "control 456"
    "control 789"
    "qwen32 soft_boundary 123"
    "qwen32 soft_boundary 456"
    "qwen32 soft_boundary 789"
)

run_one() {
    local gpu=$1 model=$2 strategy=$3 seed=$4
    local label="$model"
    if [[ "$model" != control ]]; then
        label="${model}_${strategy}"
    fi
    local name="s5_fromscratch_${label}_e15_seed${seed}_20260908"
    local log="$artifact_root/logs/${name}.log"
    if [[ -f "$log" ]] && grep -q 'Training finished!' "$log"; then
        echo "skip completed $name"
        return
    fi
    echo "start gpu=$gpu $name"
    if [[ "$model" == control ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$name" -s "$seed" -i 0 -d \
            --stop-after-epoch 15 >"$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$name" -s "$seed" -i 0 -d \
            --vlm-target-adapter-cache "$cache_root/${model}_target_adapter.json" \
            --vlm-target-adapter-policy "$strategy" \
            --vlm-target-adapter-external-selection \
            --vlm-target-adapter-budget 0.05 \
            --vlm-target-adapter-levels 0,1,2 \
            --vlm-target-adapter-confidence-threshold 0.0 \
            --stop-after-epoch 15 >"$log" 2>&1
    fi
    find "model_results/$name" -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
        -type f -delete
    echo "done gpu=$gpu $name"
}

worker() {
    local worker_index=$1 gpu=$2 index task
    for ((index=worker_index; index<${#tasks[@]}; index+=${#gpus[@]})); do
        read -r -a task <<<"${tasks[$index]}"
        if (( ${#task[@]} == 2 )); then
            run_one "$gpu" "${task[0]}" "none" "${task[1]}"
        else
            run_one "$gpu" "${task[0]}" "${task[1]}" "${task[2]}"
        fi
    done
}

worker_pids=()
for index in "${!gpus[@]}"; do
    worker "$index" "${gpus[$index]}" \
        >"$artifact_root/logs/worker_${gpus[$index]}.log" 2>&1 &
    worker_pids+=("$!")
    echo "$!" >"$artifact_root/pids/worker_${gpus[$index]}.pid"
done
printf '%s\n' "${worker_pids[@]}" >"$artifact_root/pids/workers.pid"
wait "${worker_pids[@]}"
