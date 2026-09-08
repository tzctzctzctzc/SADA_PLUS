#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
config=configs/old_vs_new/version_final_ckpt.yaml
parent=/home1/gyy/vlms/SADA_VALIDITY_HEAD_20260808/artifacts/oracle_forgetting_20260817/s2_official_epoch10_normalized.pt
artifact_root=artifacts/s5_gradient_harmonization_20260908
IFS=',' read -r control_gpu method_gpu <<<"${GPUS:-5,7}"

if [[ -z "${control_gpu:-}" || -z "${method_gpu:-}" ]]; then
    echo "GPUS must contain exactly two comma-separated GPU ids" >&2
    exit 1
fi

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

run_one() {
    local gpu=$1 mode=$2
    local name="s5_gh_${mode}_l012_e15_seed123_20260908"
    local log="$artifact_root/logs/${name}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" -r "$parent" -s 123 -i 0 -d \
        --diagnostic-alignment-levels 0,1,2 \
        --gradient-harmonization "$mode" \
        --stop-after-epoch 15 >"$log" 2>&1
}

run_one "$control_gpu" off &
control_pid=$!
echo "$control_pid" >"$artifact_root/pids/control.pid"

run_one "$method_gpu" task_protect &
method_pid=$!
echo "$method_pid" >"$artifact_root/pids/task_protect.pid"

wait "$control_pid" "$method_pid"
