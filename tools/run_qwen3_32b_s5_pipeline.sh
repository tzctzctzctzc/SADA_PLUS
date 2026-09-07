#!/usr/bin/env bash
set -euo pipefail

root=/home1/gyy/vlms/SADA_CLIP_BENCH_20260904_r1
model_dir=/home1/gyy/vlms/qwen3vl32b_bf16
download_pid_file=/home1/gyy/vlms/qwen3vl32b_bf16_download.pid
qwen_python=/home1/gyy/vlms/qwen3_vl_env/bin/python
sada_python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
parent_checkpoint=/home1/gyy/vlms/SADA_VALIDITY_HEAD_20260808/artifacts/oracle_forgetting_20260817/s2_official_epoch10_normalized.pt
prediction_root=artifacts/matrix_qwen3_vl_32b
training_root=artifacts/sada_s5_label_training

cd "$root"
mkdir -p "$prediction_root" "$training_root/logs" "$training_root/pids"

if [[ -s "$download_pid_file" ]]; then
    download_pid=$(<"$download_pid_file")
    while kill -0 "$download_pid" 2>/dev/null; do
        sleep 30
    done
fi

shard_count=$(find "$model_dir" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l)
incomplete_count=$(find "$model_dir" -type f -name '*.incomplete' | wc -l)
if [[ "$shard_count" -ne 14 || "$incomplete_count" -ne 0 ]]; then
    echo "model download incomplete: shards=$shard_count incomplete=$incomplete_count"
    exit 1
fi

while true; do
    gpu_csv=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits \
        | awk -F, '$2 + 0 < 1000 && $3 + 0 < 10 {gsub(/ /, "", $1); print $1}' \
        | head -4 | paste -sd, -
    )
    gpu_count=$(awk -F, '{print NF}' <<<"$gpu_csv")
    if [[ "$gpu_count" -ge 3 ]]; then
        break
    fi
    sleep 30
done
echo "selected_gpus=$gpu_csv count=$gpu_count"

CUDA_VISIBLE_DEVICES="$gpu_csv" PYTHONPATH=. "$qwen_python" -u \
    tools/run_qwen_vl_matrix.py \
    --scenario-root artifacts/matrix_manifests \
    --frame-cache artifacts/matrix_frames \
    --output-root "$prediction_root" \
    --model-family qwen3-32b \
    --model-path "$model_dir" \
    --offline --device-map balanced --attn-implementation sdpa --dtype bfloat16 \
    --frame-count 8 --min-pixels 50176 --max-pixels 200704 \
    --force-target --shifts S5

PYTHONPATH=. "$sada_python" tools/calibrate_vlm_temperature.py \
    --source-predictions "$prediction_root/S5/source_predictions.jsonl" \
    --output "$training_root/qwen3_vl_32b_source_temperature.json"

gate_override=()
if ! "$sada_python" -c \
    "import json; assert json.load(open('$prediction_root/S5/source_gate.json'))['adapter_pass']"; then
    gate_override=(--allow-gate-failure)
fi

PYTHONPATH=. "$sada_python" tools/build_vlm_target_adapter_cache.py \
    --manifest artifacts/matrix_manifests/S5/target_train_manifest.jsonl \
    --predictions "$prediction_root/S5/target_predictions.jsonl" \
    --source-gate-report "$prediction_root/S5/source_gate.json" \
    --temperature-calibration-report "$training_root/qwen3_vl_32b_source_temperature.json" \
    --output "$training_root/qwen3_vl_32b_target_cache_calibrated_soft.json" \
    "${gate_override[@]}"

train_gpu=${gpu_csv%%,*}
ulimit -n 65535
CUDA_VISIBLE_DEVICES="$train_gpu" PYTHONPATH=. "$sada_python" -u train_da.py \
    -c configs/old_vs_new/version_final_ckpt.yaml \
    -n qwen3vl32b_s5_calibrated_soft_e15_seed123_20260908 \
    -r "$parent_checkpoint" -s 123 -i 0 -d \
    --vlm-target-adapter-cache "$training_root/qwen3_vl_32b_target_cache_calibrated_soft.json" \
    --vlm-target-adapter-policy soft \
    --vlm-target-adapter-budget 0.05 \
    --vlm-target-adapter-levels 0,1,2 \
    --vlm-target-adapter-confidence-threshold 0.0 \
    --stop-after-epoch 15
