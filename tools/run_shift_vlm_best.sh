#!/usr/bin/env bash
set -euo pipefail

# Continue a fixed epoch-10 Trident-SADA parent with the frozen S5 VLM recipe.
# Example: SHIFT=S3 GPUS=0,1,2 bash tools/run_shift_vlm_best.sh

shift=${SHIFT:?set SHIFT to S1, S2, S3, S4, S5, or S6}
root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
parent_root=${PARENT_ROOT:-artifacts/shift_trident_parents_20260912/checkpoints}
cache_root=${CACHE_ROOT:-artifacts/shift_vlm_best_20260912/caches}
artifact_root=${SADA_VLM_ARTIFACT_ROOT:-artifacts/shift_vlm_best_20260912}
IFS=',' read -r -a gpus <<<"${GPUS:-0,1,2}"
seeds=(123 456 789)

config_for_shift() {
    case "$1" in
        S1) echo configs/dark_vs_all_old_videos_full/version_final_ckpt.yaml ;;
        S2) echo configs/white_vs_all_old_videos_full/version_final_ckpt.yaml ;;
        S3) echo configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        S4) echo configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        S5) echo configs/old_vs_new/version_final_ckpt.yaml ;;
        S6) echo configs/new_vs_old/version_paper_s6_dim512.yaml ;;
        *) echo "unknown shift: $1" >&2; return 2 ;;
    esac
}

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 3 )); then
    echo "GPUS must contain one to three GPU ids" >&2
    exit 2
fi

shift_lower=${shift,,}
config=$(config_for_shift "$shift")
cache="$cache_root/$shift/qwen32_proposal_semantic.json"
cd "$root"
test -s "$cache" || { echo "missing cache: $cache" >&2; exit 2; }
mkdir -p "$artifact_root/logs" "$artifact_root/pids" "$artifact_root/checkpoints"
ulimit -n 65535

run_seed() {
    local gpu=$1 index=$2 seed=${seeds[$2]} parent name log final_ckpt
    parent="$parent_root/${shift_lower}_seed${seed}/epoch10.pt"
    test -e "$parent" || { echo "missing epoch-10 parent: $parent" >&2; return 2; }
    name="${shift_lower}_trident_qwen32_8img_top1_posrank_seed${seed}_20260912"
    log="$artifact_root/logs/${name}.log"

    final_ckpt=$(find "model_results/$name" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
        -type f -print -quit 2>/dev/null || true)
    if [[ -n "$final_ckpt" ]]; then
        echo "skip completed shift=$shift seed=$seed checkpoint=$final_ckpt"
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
    test -n "$final_ckpt" || { echo "epoch-15 checkpoint missing for $shift seed=$seed" >&2; return 1; }
    ln -sfn "$(realpath "$final_ckpt")" \
        "$artifact_root/checkpoints/${shift_lower}_seed${seed}_epoch15.pt"
    sha256sum "$final_ckpt" > \
        "$artifact_root/checkpoints/${shift_lower}_seed${seed}_epoch15.sha256"
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
    echo "$!" >"$artifact_root/pids/${shift_lower}_worker_${gpus[$index]}.pid"
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
