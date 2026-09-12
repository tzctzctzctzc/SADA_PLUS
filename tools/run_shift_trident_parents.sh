#!/usr/bin/env bash
set -euo pipefail

# Train the paired 10-epoch Trident-SADA parents used by the frozen VLM recipe.
# Example: SHIFTS=S3,S4 GPUS=0,1,2,3 bash tools/run_shift_trident_parents.sh

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
artifact_root=${SADA_PARENT_ARTIFACT_ROOT:-artifacts/shift_trident_parents_20260912}
IFS=',' read -r -a shifts <<<"${SHIFTS:-S3,S4}"
IFS=',' read -r -a gpus <<<"${GPUS:-0,1,2,3}"
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

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 4 )); then
    echo "GPUS must contain one to four GPU ids" >&2
    exit 2
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids" "$artifact_root/checkpoints"
ulimit -n 65535

tasks=()
for shift in "${shifts[@]}"; do
    config=$(config_for_shift "$shift")
    test -f "$config" || { echo "missing config: $config" >&2; exit 2; }
    for seed in "${seeds[@]}"; do
        tasks+=("$shift:$seed:$config")
    done
done

run_task() {
    local gpu=$1 task=$2 shift seed config shift_lower name log checkpoint_dir checkpoint
    IFS=':' read -r shift seed config <<<"$task"
    shift_lower=${shift,,}
    name="${shift_lower}_trident_parent_e10_repro_seed${seed}_20260912"
    log="$artifact_root/logs/${name}.log"
    checkpoint_dir="$artifact_root/checkpoints/${shift_lower}_seed${seed}"
    mkdir -p "$checkpoint_dir"

    checkpoint=$(find "$checkpoint_dir" -maxdepth 1 -type l -o -type f 2>/dev/null | head -1 || true)
    if [[ -n "$checkpoint" && -e "$checkpoint" ]]; then
        echo "skip completed shift=$shift seed=$seed checkpoint=$checkpoint"
        return
    fi

    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" -s "$seed" -i 0 -d --trident-head \
        --stop-after-epoch 10 >"$log" 2>&1

    checkpoint=$(find "model_results/$name" \
        -path '*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt' \
        -type f -print | sort -V | tail -1)
    test -n "$checkpoint" || { echo "best checkpoint missing for $shift seed=$seed" >&2; return 1; }
    ln -sfn "$(realpath "$checkpoint")" "$checkpoint_dir/best.pt"
    sha256sum "$checkpoint" >"$checkpoint_dir/best.sha256"
}

worker() {
    local worker_index=$1 gpu=$2 index
    for ((index=worker_index; index<${#tasks[@]}; index+=${#gpus[@]})); do
        run_task "$gpu" "${tasks[$index]}"
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
exit "$status"
