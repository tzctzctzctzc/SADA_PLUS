#!/usr/bin/env bash
set -euo pipefail

root=${ROOT:-/home1/gyy/vlms/SADA_VLM_IMPROVEMENT_20260911}
qwen_python=${QWEN_PYTHON:-/home1/gyy/vlms/qwen3_vl_env/bin/python}
model_dir=${MODEL_DIR:-/home1/gyy/vlms/qwen3vl32b_bf16}
stage_root=${STAGE_ROOT:-artifacts/vlm_improvement_stage1_20260911}
frame_cache=${FRAME_CACHE:-$stage_root/frames_16}
output_root=$stage_root/qwen32_16img
baseline=artifacts/matrix_qwen3_vl_32b/S5/source_predictions.jsonl

cd "$root"
mkdir -p "$output_root/S5"

expected=$(find "$frame_cache/source/s5_source" -mindepth 2 -maxdepth 2 \
    -type f -name 'frame_16.jpg' | wc -l)
if [[ "$expected" -ne 200 ]]; then
    echo "16-frame source cache incomplete: complete_proposals=$expected expected=200"
    exit 1
fi

while true; do
    gpu_csv=$(
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
            --format=csv,noheader,nounits \
        | awk -F, '$2 + 0 < 1000 && $3 + 0 < 10 {gsub(/ /, "", $1); print $1}' \
        | head -3 | paste -sd, -
    )
    gpu_count=$(awk -F, '{print NF}' <<<"$gpu_csv")
    if [[ "$gpu_count" -ge 3 ]]; then
        break
    fi
    sleep 60
done
echo "selected_gpus=$gpu_csv"

/usr/bin/time -v -o "$output_root/resource_usage.txt" \
    env CUDA_VISIBLE_DEVICES="$gpu_csv" PYTHONPATH=. "$qwen_python" -u \
    tools/run_qwen_vl_matrix.py \
    --scenario-root artifacts/matrix_manifests \
    --frame-cache "$frame_cache" \
    --output-root "$output_root" \
    --model-family qwen3-32b \
    --model-path "$model_dir" \
    --offline --device-map balanced --attn-implementation sdpa --dtype bfloat16 \
    --frame-count 16 --min-pixels 50176 --max-pixels 200704 \
    --source-only --shifts S5 \
    2>&1 | tee "$output_root/run.log"

PYTHONPATH=. "$qwen_python" tools/compare_vlm_source_predictions.py \
    --baseline "$baseline" \
    --candidate "$output_root/S5/source_predictions.jsonl" \
    --output "$output_root/S5/compare_against_8img.json" \
    --confidence-threshold 0.7 \
    | tee "$output_root/S5/compare_against_8img.log"
