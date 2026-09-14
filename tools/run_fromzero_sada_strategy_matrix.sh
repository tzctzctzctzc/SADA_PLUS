#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
artifact_root=${BENCHMARK_ARTIFACT_ROOT:-artifacts/fromzero_sada_strategy_matrix_20260914}
epochs=${EPOCHS:-50}
min_free_mib=${MIN_FREE_MIB:-30000}
max_util=${MAX_GPU_UTIL:-10}
IFS=',' read -r -a shifts <<<"${SHIFTS:-S3,S4,S5}"
IFS=',' read -r -a models <<<"${MODELS:-clip,qwen8,qwen32}"
IFS=',' read -r -a strategies <<<"${STRATEGIES:-hard,soft,verify,soft_boundary,embedding,positive_rank}"
IFS=',' read -r -a seeds <<<"${SEEDS:-123,456,789}"
IFS=',' read -r -a gpus <<<"${GPUS:-4}"

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 4 )); then
    echo "GPUS must contain one to four comma-separated GPU ids" >&2
    exit 2
fi

config_for_shift() {
    case "$1" in
        S3) echo configs/dark_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        S4) echo configs/white_vs_all_new_videos_full/version_final_ckpt.yaml ;;
        S5) echo configs/old_vs_new/version_final_ckpt.yaml ;;
        *) echo "unsupported shift: $1" >&2; return 2 ;;
    esac
}

wait_for_gpu() {
    local gpu=$1 free util
    while true; do
        read -r free util < <(
            nvidia-smi --id="$gpu" \
                --query-gpu=memory.free,utilization.gpu \
                --format=csv,noheader,nounits \
                | tr -d ' ' | tr ',' ' '
        )
        if (( free >= min_free_mib && util <= max_util )); then
            return
        fi
        echo "wait gpu=$gpu free_mib=$free util=$util%" >&2
        sleep 30
    done
}

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids" "$artifact_root/caches"
ulimit -n 65535

tasks=()
for seed in "${seeds[@]}"; do
    for shift in "${shifts[@]}"; do
        config_for_shift "$shift" >/dev/null
        tasks+=("$shift control control $seed")
        for model in "${models[@]}"; do
            for strategy in "${strategies[@]}"; do
                tasks+=("$shift $model $strategy $seed")
            done
        done
    done
done

git_commit=$(git rev-parse HEAD)
protocol="$artifact_root/protocol.tsv"
printf 'key\tvalue\n' >"$protocol"
printf 'architecture\toriginal_sada_no_trident\n' >>"$protocol"
printf 'training_initialization\trandom_epoch0\n' >>"$protocol"
printf 'vlm_start_epoch\t1\n' >>"$protocol"
printf 'proposal_provenance\texternal_frozen_teacher\n' >>"$protocol"
printf 'checkpoint_selection\tsource_validation_loss\n' >>"$protocol"
printf 'target_gt_role\treporting_only\n' >>"$protocol"
printf 'git_commit\t%s\n' "$git_commit" >>"$protocol"
printf 'epochs\t%s\n' "$epochs" >>"$protocol"
printf 'hostname\t%s\n' "$(hostname)" >>"$protocol"
printf 'python_version\t%s\n' "$($python --version 2>&1)" >>"$protocol"
printf 'cuda_driver\t%s\n' "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')" >>"$protocol"
printf 'shifts\t%s\n' "${SHIFTS:-S3,S4,S5}" >>"$protocol"
printf 'models\t%s\n' "${MODELS:-clip,qwen8,qwen32}" >>"$protocol"
printf 'strategies\t%s\n' "${STRATEGIES:-hard,soft,verify,soft_boundary,embedding,positive_rank}" >>"$protocol"
printf 'seeds\t%s\n' "${SEEDS:-123,456,789}" >>"$protocol"

manifest="$artifact_root/run_manifest.tsv"
printf 'run_name\tshift\tmodel\tstrategy\tseed\tconfig\tcache\tcache_sha256\n' >"$manifest"
for task in "${tasks[@]}"; do
    read -r shift model strategy seed <<<"$task"
    shift_lower=${shift,,}
    name="fromzero_sada_${shift_lower}_${model}_${strategy}_e${epochs}_seed${seed}_20260914"
    config=$(config_for_shift "$shift")
    cache='none'
    cache_sha256='none'
    if [[ "$model" != control ]]; then
        suffix=target_adapter
        if [[ "$strategy" == embedding || "$strategy" == positive_rank ]]; then
            suffix=proposal_semantic
        fi
        cache="$artifact_root/caches/$shift/${model}_${suffix}.json"
        test -s "$cache" || {
            echo "missing cache: $cache" >&2
            exit 3
        }
        cache_sha256=$(sha256sum "$cache" | cut -d' ' -f1)
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$shift" "$model" "$strategy" "$seed" \
        "$config" "$cache" "$cache_sha256" >>"$manifest"
done

status_log="$artifact_root/run_status.tsv"
if [[ ! -e "$status_log" ]]; then
    printf 'timestamp\trun_name\tstatus\tgpu\trc\n' >"$status_log"
fi

run_one() {
    local gpu=$1 task=$2
    local shift model strategy seed shift_lower name config cache log rc
    read -r shift model strategy seed <<<"$task"
    shift_lower=${shift,,}
    name="fromzero_sada_${shift_lower}_${model}_${strategy}_e${epochs}_seed${seed}_20260914"
    config=$(config_for_shift "$shift")
    log="$artifact_root/logs/${name}.log"

    if [[ -f "$log" ]] && grep -q 'Training finished!' "$log"; then
        printf '%s\t%s\tcompleted_existing\t%s\t0\n' \
            "$(date --iso-8601=seconds)" "$name" "$gpu" >>"$status_log"
        return
    fi

    wait_for_gpu "$gpu"
    printf '%s\t%s\tstarted\t%s\t0\n' \
        "$(date --iso-8601=seconds)" "$name" "$gpu" >>"$status_log"
    printf 'run_name=%s\nshift=%s\nmodel=%s\nstrategy=%s\nseed=%s\n' \
        "$name" "$shift" "$model" "$strategy" "$seed" >"$log"
    printf 'training_initialization=random_epoch0\nproposal_provenance=external_frozen_teacher\n' \
        >>"$log"

    common=(
        -c "$config" -n "$name" -s "$seed" -i 0 -d
        --diagnostic-alignment-levels 0,1,2,3,4,5
        --benchmark-audit-log
        --stop-after-epoch "$epochs"
    )

    if [[ "$model" == control ]]; then
        if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            "${common[@]}" >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    elif [[ "$strategy" == embedding ]]; then
        cache="$artifact_root/caches/$shift/${model}_proposal_semantic.json"
        if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            "${common[@]}" \
            --proposal-semantic-mode vlm_cotrain \
            --proposal-semantic-target-cache "$cache" \
            --proposal-semantic-external-selection \
            --proposal-semantic-target-start-epoch 1 \
            --proposal-semantic-levels 0,1,2 \
            --proposal-semantic-max-proposals 256 \
            --proposal-semantic-confidence 0.1 \
            --proposal-semantic-entropy 1.0 \
            --proposal-semantic-source-weight 1.0 \
            --proposal-semantic-alignment-weight 0.1 \
            --proposal-semantic-direct-weight 0.0 >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    elif [[ "$strategy" == positive_rank ]]; then
        cache="$artifact_root/caches/$shift/${model}_proposal_semantic.json"
        if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            "${common[@]}" \
            --proposal-semantic-mode vlm_logit_confident \
            --proposal-semantic-target-cache "$cache" \
            --proposal-semantic-external-selection \
            --proposal-semantic-target-start-epoch 1 \
            --proposal-semantic-levels 0,1,2 \
            --proposal-semantic-score-threshold 0.3 \
            --proposal-semantic-max-proposals 256 \
            --proposal-semantic-confidence 0.7 \
            --proposal-semantic-source-weight 0.0 \
            --proposal-semantic-alignment-weight 0.0 \
            --proposal-semantic-direct-weight 0.2 \
            --proposal-semantic-direct-loss positive_bce_rank \
            --proposal-semantic-direct-pooling top1 \
            --proposal-semantic-logit-max-delta 1.0 >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    else
        cache="$artifact_root/caches/$shift/${model}_target_adapter.json"
        if CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
            "${common[@]}" \
            --vlm-target-adapter-cache "$cache" \
            --vlm-target-adapter-policy "$strategy" \
            --vlm-target-adapter-external-selection \
            --vlm-target-adapter-budget 0.05 \
            --vlm-target-adapter-levels 0,1,2 \
            --vlm-target-adapter-confidence-threshold 0.0 >>"$log" 2>&1; then
            rc=0
        else
            rc=$?
        fi
    fi

    if (( rc == 0 )) && grep -q 'Training finished!' "$log"; then
        find "model_results/$name" \
            -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch*.pt' \
            -type f -delete
        printf '%s\t%s\tcompleted\t%s\t0\n' \
            "$(date --iso-8601=seconds)" "$name" "$gpu" >>"$status_log"
        return
    fi

    printf '%s\t%s\tfailed\t%s\t%s\n' \
        "$(date --iso-8601=seconds)" "$name" "$gpu" "$rc" >>"$status_log"
    return 1
}

worker() {
    local worker_index=$1 gpu=$2 index status=0
    for ((index=worker_index; index<${#tasks[@]}; index+=${#gpus[@]})); do
        run_one "$gpu" "${tasks[$index]}" || status=1
    done
    return "$status"
}

worker_pids=()
for index in "${!gpus[@]}"; do
    worker "$index" "${gpus[$index]}" \
        >"$artifact_root/logs/worker_${gpus[$index]}.log" 2>&1 &
    worker_pids+=("$!")
    echo "$!" >"$artifact_root/pids/worker_${gpus[$index]}.pid"
done
printf '%s\n' "${worker_pids[@]}" >"$artifact_root/pids/workers.pid"

status=0
for pid in "${worker_pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
