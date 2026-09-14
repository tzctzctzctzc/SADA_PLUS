#!/usr/bin/env bash
set -euo pipefail

# Train an original SADA parent from epoch 0, select it by source-val loss,
# then freeze the parent and train the fixed Qwen32 target adapter for 5 epochs.
# Example:
#   SHIFTS=S3,S4 GPUS=3,4,5,6 bash tools/run_shift_best_vlm_no_trident.sh

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
cache_root=${CACHE_ROOT:-artifacts/shift_vlm_best_20260912/caches}
artifact_root=${SADA_VLM_ARTIFACT_ROOT:-artifacts/shift_vlm_no_trident_20260914}
IFS=',' read -r -a shifts <<<"${SHIFTS:-S3,S4}"
IFS=',' read -r -a gpus <<<"${GPUS:-3,4,5,6}"
seeds=(123 456 789)

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 4 )); then
    echo "GPUS must contain one to four comma-separated GPU ids" >&2
    exit 2
fi

config_for_shift() {
    case "$1" in
        S3) echo configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        S4) echo configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        *) echo "this launcher currently supports only S3 and S4: $1" >&2; return 2 ;;
    esac
}

tasks=()
for seed in "${seeds[@]}"; do
    for shift in "${shifts[@]}"; do
        config_for_shift "$shift" >/dev/null
        tasks+=("$shift:$seed")
    done
done

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids" "$artifact_root/checkpoints"
ulimit -n 65535

run_task() {
    local gpu=$1 task=$2 shift=${2%%:*} seed=${2##*:}
    local shift_lower=${shift,,} config cache parent_name parent_log parent_dir
    local parent parent_epoch target_start stop_epoch name log

    config=$(config_for_shift "$shift")
    cache="$cache_root/$shift/qwen32_proposal_semantic.json"
    test -s "$cache" || { echo "missing cache: $cache" >&2; return 2; }

    parent_name="${shift_lower}_repro_sada_no_trident_e15_seed${seed}_20260914"
    parent_log="$artifact_root/logs/${parent_name}.log"
    parent_dir="model_results/$parent_name"

    if ! grep -q 'Training finished!' "$parent_log" 2>/dev/null; then
        echo "start parent shift=$shift gpu=$gpu seed=$seed"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            -c "$config" -n "$parent_name" -s "$seed" -i 0 -d \
            --stop-after-epoch 15 >"$parent_log" 2>&1
    fi

    parent=$(find "$parent_dir" \
        -path '*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt' \
        -type f -print -quit)
    if [[ -z "$parent" ]]; then
        echo "source-loss best parent checkpoint is missing for shift=$shift seed=$seed" >&2
        return 1
    fi

    parent_epoch=$("$python" -c \
        "import torch; print(int(torch.load('$parent', map_location='cpu')['epoch']))")
    target_start=$((parent_epoch + 1))
    stop_epoch=$((parent_epoch + 5))

    # Retain the source-selected parent; its duplicate latest checkpoint is expendable.
    find "$parent_dir" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch*.pt' \
        -type f -delete

    name="${shift_lower}_qwen32_top1_posrank_no_trident_from_repro_seed${seed}_20260914"
    log="$artifact_root/logs/${name}.log"
    if grep -q 'Training finished!' "$log" 2>/dev/null; then
        echo "skip completed VLM shift=$shift seed=$seed"
        return
    fi

    echo "start VLM shift=$shift gpu=$gpu seed=$seed parent_epoch=$parent_epoch"
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

    grep -q 'Training finished!' "$log" || {
        echo "VLM training did not finish for shift=$shift seed=$seed" >&2
        return 1
    }
    echo "done VLM shift=$shift gpu=$gpu seed=$seed"
}

worker() {
    local worker_index=$1 gpu=$2 index
    for ((index=worker_index; index<${#tasks[@]}; index+=${#gpus[@]})); do
        run_task "$gpu" "${tasks[$index]}"
    done
}

pids=()
for index in "${!gpus[@]}"; do
    worker "$index" "${gpus[$index]}" \
        >"$artifact_root/logs/worker_${gpus[$index]}.log" 2>&1 &
    pids+=("$!")
    echo "$!" >"$artifact_root/pids/worker_${gpus[$index]}.pid"
done
printf '%s\n' "${pids[@]}" >"$artifact_root/pids/workers.pid"

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
