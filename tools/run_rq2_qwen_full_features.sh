#!/usr/bin/env bash
set -euo pipefail

root=${RQ2_ROOT:-/vepfs/vlms/SADA_PLUS}
artifact=${RQ2_ARTIFACT:-$root/artifacts/rq2_probe_20260919}
python=${RQ2_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq2/frames}
gpu_a=${RQ2_GPU_A:-0}
gpu_b=${RQ2_GPU_B:-1}
minimum_free_mb=${RQ2_MINIMUM_FREE_MB:-30000}

if [[ "$gpu_a" == "$gpu_b" ]]; then
    echo "RQ2_GPU_A and RQ2_GPU_B must differ" >&2
    exit 2
fi

manifest=$artifact/rq2_probe_manifest.jsonl
namespace=rq2_probe_counterfactual_2fps_dedup_v1
feature_root=$artifact/features/qwen3_vl_8b_full
index_a=$artifact/qwen3_vl_8b_full_shard0.jsonl
index_b=$artifact/qwen3_vl_8b_full_shard1.jsonl

mkdir -p "$artifact/logs" "$feature_root"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -f "$artifact/.frames_full_complete" ]]; then
    "$python" tools/prepare_rq2_probe_frames.py \
        --manifest "$manifest" --video-root "$video_root" \
        --frame-cache "$frame_cache" --cache-namespace "$namespace" \
        --workers 8 --decode-mode seek \
        >"$artifact/logs/full_prepare_frames.log" 2>&1
    touch "$artifact/.frames_full_complete"
fi

wait_for_gpu() {
    local gpu=$1
    while true; do
        local values used free util
        values=$(nvidia-smi -i "$gpu" \
            --query-gpu=memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits)
        IFS=',' read -r used free util <<<"$values"
        free=${free// /}
        util=${util// /}
        if (( free >= minimum_free_mb && util <= 20 )); then
            return 0
        fi
        echo "$(date --iso-8601=seconds) waiting_gpu=$gpu free_mb=$free util=$util" 
        sleep 60
    done
}

wait_for_gpu "$gpu_a" &
wait_a=$!
wait_for_gpu "$gpu_b" &
wait_b=$!
wait "$wait_a" "$wait_b"

pids=()
for shard in 0 1; do
    gpu=$gpu_a
    index=$index_a
    [[ "$shard" == 1 ]] && gpu=$gpu_b
    [[ "$shard" == 1 ]] && index=$index_b
    CUDA_VISIBLE_DEVICES=$gpu "$python" tools/extract_rq2_qwen_frame_features.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace "$namespace" --output-index "$index" \
        --feature-root "$feature_root" --model-path Qwen/Qwen3-VL-8B-Instruct \
        --offline --min-pixels $((224 * 224)) --max-pixels $((224 * 224)) \
        --max-batch-frames 256 --loader-workers 8 \
        --shard-count 2 --shard-index "$shard" \
        >"$artifact/logs/full_qwen3_vl_8b_shard${shard}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

touch "$artifact/.qwen3_vl_8b_full_features_complete"
