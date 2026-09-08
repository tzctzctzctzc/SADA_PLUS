#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
parent=/home1/gyy/vlms/SADA_VALIDITY_HEAD_20260808/artifacts/oracle_forgetting_20260817/s2_official_epoch10_normalized.pt
cache=artifacts/s5_strategy_matrix/caches/qwen32_proposal_semantic.json
config=configs/old_vs_new/version_final_ckpt.yaml
artifact_root=artifacts/s5_vlm_logit_adapter_20260908
IFS=',' read -r -a gpus <<<"${GPUS:-2,3,4}"

if (( ${#gpus[@]} < 1 || ${#gpus[@]} > 3 )); then
    echo "GPUS must contain one to three ids; GPU 0 is reserved for the active probe" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535
policies=(soft confident agree)

run_one() {
    local gpu=$1 policy=$2
    local name="s5_qwen32_logit_${policy}_l012_e15_seed123_20260908"
    local log="$artifact_root/logs/${name}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" -s 123 -i 0 -d \
        --proposal-semantic-continue-from-sada "$parent" \
        --proposal-semantic-init-weights student \
        --proposal-semantic-mode "vlm_logit_${policy}" \
        --proposal-semantic-target-cache "$cache" \
        --proposal-semantic-target-start-epoch 11 \
        --proposal-semantic-levels 0,1,2 \
        --proposal-semantic-max-proposals 256 \
        --proposal-semantic-confidence 0.5 \
        --proposal-semantic-source-weight 0.0 \
        --proposal-semantic-alignment-weight 0.0 \
        --proposal-semantic-direct-weight 0.1 \
        --proposal-semantic-logit-max-delta 2.0 \
        --diagnostic-alignment-levels 0,1,2 \
        --stop-after-epoch 15 >"$log" 2>&1
    find "model_results/$name" \
        -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
        -type f -delete
}

worker() {
    local worker_index=$1 gpu=$2 index
    for ((index=worker_index; index<${#policies[@]}; index+=${#gpus[@]})); do
        run_one "$gpu" "${policies[$index]}"
    done
}

for index in "${!gpus[@]}"; do
    worker "$index" "${gpus[$index]}" &
    echo "$!" >"$artifact_root/pids/worker_${gpus[$index]}.pid"
done
wait
