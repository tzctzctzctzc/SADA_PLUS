#!/usr/bin/env bash
set -euo pipefail

# VLM improvement plan, stage 0:
# reproduce the locked S5 Trident-SADA + Qwen3-VL-32B 8-image result while
# saving the final adapter checkpoint. Target GT is never used for training or
# checkpoint selection; the fixed epoch-15 checkpoint is the reported model.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
cache=${SADA_QWEN32_CACHE:-artifacts/s5_strategy_matrix/caches/qwen32_proposal_semantic.json}
artifact_root=${SADA_STAGE0_ARTIFACT_ROOT:-artifacts/vlm_improvement_stage0_20260911}
expected_cache_sha256=0f5171f68f2fabd18f051b03f41a34cd792847d0f059e86d3d4a2ada94948446
IFS=',' read -r -a gpus <<<"${GPUS:-0,1,2}"

seeds=(123 456 789)
parents=(
    model_results/tridet_s5_parent_e10_repro_all_seed123_20260909/260909_040935_883/saved_ckpt/best_ckpt/checkpoint-epoch10.pt
    model_results/tridet_s5_parent_e10_repro_all_seed456_20260909/260909_045249_710/saved_ckpt/best_ckpt/checkpoint-epoch10.pt
    model_results/tridet_s5_parent_e10_repro_all_seed789_20260909/260909_045249_758/saved_ckpt/best_ckpt/checkpoint-epoch10.pt
)

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 3 )); then
    echo "GPUS must contain one to three GPU ids" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids" "$artifact_root/checkpoints"
ulimit -n 65535

actual_cache_sha256=$(sha256sum "$cache" | awk '{print $1}')
if [[ "$actual_cache_sha256" != "$expected_cache_sha256" ]]; then
    echo "Qwen32 cache hash mismatch: $actual_cache_sha256" >&2
    exit 1
fi
for parent in "${parents[@]}"; do
    test -f "$parent" || { echo "missing parent checkpoint: $parent" >&2; exit 1; }
done

run_seed() {
    local gpu=$1 index=$2 seed=${seeds[$2]} parent=${parents[$2]}
    local name="s5_stage0_trident_qwen32_8img_top1_posrank_seed${seed}_20260911"
    local log="$artifact_root/logs/${name}.log"
    local final_ckpt

    final_ckpt=$(find "model_results/$name" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
        -type f -print -quit 2>/dev/null || true)
    if [[ -n "$final_ckpt" ]]; then
        echo "skip completed seed=$seed checkpoint=$final_ckpt"
        return
    fi

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" -s "$seed" -i 0 -d --trident-head \
        --proposal-semantic-continue-from-sada "$parent" \
        --proposal-semantic-init-weights student \
        --proposal-semantic-mode vlm_logit_confident \
        --proposal-semantic-target-cache "$cache" \
        --proposal-semantic-external-selection \
        --proposal-semantic-target-start-epoch 11 \
        --proposal-semantic-levels 0,1,2 \
        --proposal-semantic-max-proposals 256 \
        --proposal-semantic-confidence 0.7 \
        --proposal-semantic-source-weight 0.0 \
        --proposal-semantic-alignment-weight 0.0 \
        --proposal-semantic-direct-weight 0.2 \
        --proposal-semantic-logit-max-delta 1.0 \
        --proposal-semantic-freeze-base \
        --proposal-semantic-direct-loss positive_bce_rank \
        --proposal-semantic-direct-pooling top1 \
        --diagnostic-alignment-levels 0,1,2 \
        --stop-after-epoch 15 >"$log" 2>&1

    final_ckpt=$(find "model_results/$name" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
        -type f -print -quit)
    test -n "$final_ckpt" || { echo "final checkpoint missing for seed=$seed" >&2; exit 1; }
    ln -sfn "$(realpath "$final_ckpt")" "$artifact_root/checkpoints/seed${seed}_epoch15.pt"
    sha256sum "$final_ckpt" >"$artifact_root/checkpoints/seed${seed}_epoch15.sha256"
}

worker() {
    local worker_index=$1 gpu=$2 index
    for ((index=worker_index; index<${#seeds[@]}; index+=${#gpus[@]})); do
        run_seed "$gpu" "$index"
    done
}

pids=()
for index in "${!gpus[@]}"; do
    worker "$index" "${gpus[$index]}" &
    pids+=("$!")
    echo "$!" >"$artifact_root/pids/worker_${gpus[$index]}.pid"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo "stage 0 failed; inspect per-seed logs" >&2
    exit "$status"
fi

echo "stage 0 complete"
