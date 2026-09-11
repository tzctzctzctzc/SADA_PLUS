#!/usr/bin/env bash
set -euo pipefail

# S5 diagnostic ladder.  Every mode except control uses target GT and therefore
# measures headroom only; none of these results is an admissible UDA result.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
artifact_root=${SADA_ORACLE_ROOT:-artifacts/s5_oracle_ladder_20260911}
max_gpus=${MAX_GPUS:-4}
poll_seconds=${POLL_SECONDS:-60}
seed=${SEED:-123}

tasks=(
    target_only
    source_target
    joint
    class
    boundary
    control
)

if (( max_gpus < 1 || max_gpus > 4 )); then
    echo "MAX_GPUS must be between 1 and 4" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

run_one() {
    local gpu=$1 task=$2
    local name="s5_oracle_${task}_trident_e50_seed${seed}_20260911"
    local log="$artifact_root/logs/${name}.log"
    local run_root="model_results/$name"
    local best

    best=$(find "$run_root" -path '*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt' \
        -type f -print -quit 2>/dev/null || true)
    if [[ -n "$best" ]] && grep -q 'Training finished!' "$log" 2>/dev/null; then
        echo "skip completed task=$task checkpoint=$best"
        return
    fi

    args=(
        -c "$config"
        -n "$name"
        -s "$seed"
        -i 0
        -d
        --trident-head
        --stop-after-epoch 50
    )
    case "$task" in
        control)
            ;;
        class|boundary|joint)
            args+=(--oracle-target-task "$task")
            ;;
        target_only|source_target)
            args+=(--supervised-upper-bound "$task")
            ;;
        *)
            echo "unknown Oracle task: $task" >&2
            return 2
            ;;
    esac

    echo "start gpu=$gpu task=$task name=$name"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        "${args[@]}" >"$log" 2>&1

    best=$(find "$run_root" -path '*/saved_ckpt/best_ckpt/checkpoint-epoch*.pt' \
        -type f -print -quit)
    test -n "$best" || { echo "best checkpoint missing: $task" >&2; return 1; }
    find "$run_root" -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch*.pt' \
        -type f -delete
    sha256sum "$best" >"$artifact_root/${task}_best.sha256"
    echo "done gpu=$gpu task=$task checkpoint=$best"
}

declare -A active_pid_by_gpu=()
declare -A active_task_by_gpu=()
next_task=0
failed=0

while (( next_task < ${#tasks[@]} || ${#active_pid_by_gpu[@]} > 0 )); do
    for gpu in "${!active_pid_by_gpu[@]}"; do
        pid=${active_pid_by_gpu[$gpu]}
        if ! kill -0 "$pid" 2>/dev/null; then
            if ! wait "$pid"; then
                echo "failed gpu=$gpu task=${active_task_by_gpu[$gpu]}" >&2
                failed=1
            fi
            unset 'active_pid_by_gpu[$gpu]'
            unset 'active_task_by_gpu[$gpu]'
        fi
    done

    if (( next_task < ${#tasks[@]} )); then
        mapfile -t idle_gpus < <(
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                --format=csv,noheader,nounits \
            | awk -F, '$2 + 0 < 1000 && $3 + 0 < 10 {gsub(/ /, "", $1); print $1}'
        )
        for gpu in "${idle_gpus[@]}"; do
            (( ${#active_pid_by_gpu[@]} >= max_gpus )) && break
            (( next_task >= ${#tasks[@]} )) && break
            [[ -n ${active_pid_by_gpu[$gpu]+x} ]] && continue
            task=${tasks[$next_task]}
            run_one "$gpu" "$task" \
                >>"$artifact_root/logs/controller.log" 2>&1 &
            active_pid_by_gpu[$gpu]=$!
            active_task_by_gpu[$gpu]=$task
            echo "$!" >"$artifact_root/pids/${task}.pid"
            next_task=$((next_task + 1))
        done
    fi

    if (( next_task < ${#tasks[@]} || ${#active_pid_by_gpu[@]} > 0 )); then
        sleep "$poll_seconds"
    fi
done

if (( failed != 0 )); then
    exit 1
fi
echo "S5 Oracle ladder complete"
