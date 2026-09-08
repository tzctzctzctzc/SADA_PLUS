#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
config=configs/old_vs_new/version_final_ckpt.yaml
parent=/home1/gyy/vlms/SADA_VALIDITY_HEAD_20260808/artifacts/oracle_forgetting_20260817/s2_official_epoch10_normalized.pt
artifact_root=artifacts/s5_transport_adapter_20260908
IFS=',' read -r control_gpu uot_gpu adapter_gpu uot_adapter_gpu <<<"${GPUS:-0,2,3,4}"

cd "$root"
mkdir -p "$artifact_root/logs" "$artifact_root/pids"
ulimit -n 65535

run_one() {
    local gpu=$1 mode=$2 start_flag=$3
    local name="s5_transport_${mode}_l012_e15_seed123_20260908"
    local log="$artifact_root/logs/${name}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
        -c "$config" -n "$name" "$start_flag" "$parent" -s 123 -i 0 -d \
        --diagnostic-alignment-levels 0,1,2 \
        --transport-alignment "$mode" \
        --transport-levels 0,1,2 \
        --stop-after-epoch 15 >"$log" 2>&1
}

run_one "$control_gpu" off -r &
echo $! >"$artifact_root/pids/control.pid"
run_one "$uot_gpu" uot -r &
echo $! >"$artifact_root/pids/uot.pid"
run_one "$adapter_gpu" adapter_grl --transport-continue-from-sada &
echo $! >"$artifact_root/pids/adapter_grl.pid"
run_one "$uot_adapter_gpu" uot_adapter --transport-continue-from-sada &
echo $! >"$artifact_root/pids/uot_adapter.pid"

wait
