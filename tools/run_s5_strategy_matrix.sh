#!/usr/bin/env bash
set -euo pipefail

root=/home1/gyy/vlms/SADA_CLIP_BENCH_20260904_r1
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
parent=/home1/gyy/vlms/SADA_VALIDITY_HEAD_20260808/artifacts/oracle_forgetting_20260817/s2_official_epoch10_normalized.pt
config=configs/old_vs_new/version_final_ckpt.yaml
artifact_root=artifacts/s5_strategy_matrix
cache_root="$artifact_root/caches"
IFS=',' read -r -a gpus <<<"${GPUS:-0,2,3,4}"

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 4 )); then
    echo "GPUS must contain one to four comma-separated GPU ids" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

tasks=()
for model in clip qwen8 qwen32; do
    for strategy in hard soft verify soft_boundary embedding; do
        # The fair calibrated-soft 8B/32B three-seed runs already exist.
        if [[ "$strategy" == soft && "$model" != clip ]]; then
            continue
        fi
        for seed in 123 456 789; do
            tasks+=("$model $strategy $seed")
        done
    done
done

run_one() {
    local gpu=$1 model=$2 strategy=$3 seed=$4
    local name="s5_${model}_${strategy}_e15_seed${seed}_20260908"
    local log="$artifact_root/logs/${name}.log"
    if [[ -f "$log" ]] && grep -q 'Training finished!' "$log"; then
        echo "skip completed $name"
        return
    fi
    echo "start gpu=$gpu $name"
    if [[ "$strategy" == embedding ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$name" -s "$seed" -i 0 -d \
            --proposal-semantic-continue-from-sada "$parent" \
            --proposal-semantic-init-weights student \
            --proposal-semantic-mode vlm \
            --proposal-semantic-target-cache "$cache_root/${model}_proposal_semantic.json" \
            --proposal-semantic-target-start-epoch 11 \
            --proposal-semantic-levels 0,1,2 \
            --proposal-semantic-max-proposals 256 \
            --proposal-semantic-confidence 0.1 \
            --proposal-semantic-entropy 1.0 \
            --proposal-semantic-source-weight 1.0 \
            --proposal-semantic-alignment-weight 0.1 \
            --proposal-semantic-direct-weight 0.0 \
            --stop-after-epoch 15 >"$log" 2>&1
    else
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$name" -r "$parent" -s "$seed" -i 0 -d \
            --vlm-target-adapter-cache "$cache_root/${model}_target_adapter.json" \
            --vlm-target-adapter-policy "$strategy" \
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
        run_one "$gpu" "${task[0]}" "${task[1]}" "${task[2]}"
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

