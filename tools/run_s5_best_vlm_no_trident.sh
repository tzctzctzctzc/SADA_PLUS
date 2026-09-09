#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
config=configs/old_vs_new/version_final_ckpt.yaml
cache=artifacts/s5_strategy_matrix/caches/qwen32_proposal_semantic.json
artifact_root=artifacts/s5_best_vlm_no_trident_20260909
IFS=',' read -r -a gpus <<<"${GPUS:-2,4,7}"
seeds=(123 456 789)

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 3 )); then
    echo "GPUS must contain one to three comma-separated GPU ids" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

run_seed() {
    local gpu=$1 seed=$2
    local parent_name="s5_repro_sada_no_trident_e15_seed${seed}_20260909"
    local parent_log="$artifact_root/logs/${parent_name}.log"
    local parent_dir="model_results/$parent_name"

    if ! grep -q 'Training finished!' "$parent_log" 2>/dev/null; then
        echo "start parent gpu=$gpu seed=$seed"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$parent_name" -s "$seed" -i 0 -d \
            --stop-after-epoch 15 >"$parent_log" 2>&1
    fi

    local parent
    parent=$(find "$parent_dir" \
        -path '*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt' \
        -type f | head -1)
    if [[ -z "$parent" ]]; then
        echo "source-loss best parent checkpoint is missing for seed=$seed" >&2
        return 1
    fi

    local parent_epoch target_start stop_epoch
    parent_epoch=$("$python" -c \
        "import torch; print(int(torch.load('$parent', map_location='cpu')['epoch']))")
    target_start=$((parent_epoch + 1))
    stop_epoch=$((parent_epoch + 5))

    # The source-loss best parent is retained; the duplicate latest checkpoint is expendable.
    find "$parent_dir" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch*.pt' \
        -type f -delete

    local name="s5_qwen32_top1_posrank_no_trident_from_repro_seed${seed}_20260909"
    local log="$artifact_root/logs/${name}.log"
    if grep -q 'Training finished!' "$log" 2>/dev/null; then
        echo "skip completed VLM seed=$seed"
        return
    fi

    echo "start VLM gpu=$gpu seed=$seed parent_epoch=$parent_epoch"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" -s "$seed" -i 0 -d \
        --proposal-semantic-continue-from-sada "$parent" \
        --proposal-semantic-init-weights student \
        --proposal-semantic-mode vlm_logit_confident \
        --proposal-semantic-target-cache "$cache" \
        --proposal-semantic-external-selection \
        --proposal-semantic-target-start-epoch "$target_start" \
        --proposal-semantic-levels 0,1,2 \
        --proposal-semantic-score-threshold 0.3 \
        --proposal-semantic-max-proposals 256 \
        --proposal-semantic-confidence 0.7 \
        --proposal-semantic-source-weight 0.0 \
        --proposal-semantic-alignment-weight 0.0 \
        --proposal-semantic-direct-weight 0.2 \
        --proposal-semantic-direct-loss positive_bce_rank \
        --proposal-semantic-direct-pooling top1 \
        --proposal-semantic-logit-max-delta 1.0 \
        --proposal-semantic-freeze-base \
        --diagnostic-alignment-levels 0,1,2 \
        --diagnostic-test-target-val-only \
        --stop-after-epoch "$stop_epoch" >"$log" 2>&1
    echo "done VLM gpu=$gpu seed=$seed"
}

worker() {
    local worker_index=$1 gpu=$2 index
    for ((index=worker_index; index<${#seeds[@]}; index+=${#gpus[@]})); do
        run_seed "$gpu" "${seeds[$index]}"
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
