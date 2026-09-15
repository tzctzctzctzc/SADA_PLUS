#!/usr/bin/env bash
set -euo pipefail

root=${RQ1_ROOT:-/vepfs/vlms/SADA_PLUS}
python=${RQ1_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
sada_python=${SADA_PYTHON:-/vepfs/vlms/envs/sada/bin/python}
artifact=${RQ1_ARTIFACT:-$root/artifacts/rq1_semantic_20260915}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq1/frames}
gpu0=${GPU0:-0}
gpu1=${GPU1:-1}
frame_count=16
namespace=rq1_16f
manifest=$artifact/rq1_val_gt_segments.jsonl
recovery=$artifact/frame_count_16

if [[ "$gpu0" == "$gpu1" ]]; then
    echo "GPU0 and GPU1 must be different" >&2
    exit 2
fi

mkdir -p "$recovery/logs"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "$(date --iso-8601=seconds) prepare_rq1_16f"
"$sada_python" tools/prepare_rq1_frames.py \
    --manifest "$manifest" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace "$namespace" \
    --frame-count "$frame_count" --workers 8

shard_pending() {
    local output=$1 prefix=$2
    shift 2
    local inputs=()
    local path
    for path in "$@"; do
        [[ -s "$path" ]] && inputs+=(--input "$path")
    done
    if [[ ${#inputs[@]} -eq 0 ]]; then
        : >"$output"
        inputs=(--input "$output")
    fi
    "$python" tools/merge_and_shard_rq1.py \
        --manifest "$manifest" "${inputs[@]}" --output "$output" \
        --frame-root "$frame_cache/$namespace" --frame-count "$frame_count" \
        --shard-prefix "$prefix" --num-shards 2
}

run_qwen_pair() {
    local family=$1 model_path=$2 output=$3 tag=$4
    local prefix=$recovery/${tag}_manifest
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/${tag}_output_0.jsonl" \
        "$recovery/${tag}_output_1.jsonl"
    local pids=()
    local index gpu shard shard_output
    for index in 0 1; do
        shard=${prefix}_${index}.jsonl
        [[ ! -s "$shard" ]] && continue
        gpu=$gpu0
        [[ "$index" == 1 ]] && gpu=$gpu1
        shard_output=$recovery/${tag}_output_${index}.jsonl
        CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq1_qwen.py \
            --manifest "$shard" --frame-cache "$frame_cache" \
            --cache-namespace "$namespace" --output "$shard_output" \
            --model-family "$family" --model-path "$model_path" \
            --offline --frame-count "$frame_count" \
            >"$recovery/logs/${tag}_${index}.log" 2>&1 &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/${tag}_output_0.jsonl" \
        "$recovery/${tag}_output_1.jsonl"
}

run_clip_pair() {
    local output=$artifact/clip_vit_b16_16f.jsonl
    local prefix=$recovery/clip_manifest
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/clip_output_0.jsonl" "$recovery/clip_output_1.jsonl"
    local pids=()
    local index gpu shard shard_output
    for index in 0 1; do
        shard=${prefix}_${index}.jsonl
        [[ ! -s "$shard" ]] && continue
        gpu=$gpu0
        [[ "$index" == 1 ]] && gpu=$gpu1
        shard_output=$recovery/clip_output_${index}.jsonl
        CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq1_clip.py \
            --manifest "$shard" --frame-cache "$frame_cache" \
            --cache-namespace "$namespace" --output "$shard_output" \
            --model ViT-B/16 --model-cache /vepfs/vlms/cache/clip \
            --frame-count "$frame_count" --segment-batch-size 32 \
            --image-batch-size 128 >"$recovery/logs/clip_${index}.log" 2>&1 &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/clip_output_0.jsonl" "$recovery/clip_output_1.jsonl"
}

echo "$(date --iso-8601=seconds) rq1_16f_qwen32"
run_qwen_pair qwen3-32b Qwen/Qwen3-VL-32B-Instruct \
    "$artifact/qwen3_vl_32b_16f.jsonl" qwen32
echo "$(date --iso-8601=seconds) rq1_16f_qwen8"
run_qwen_pair qwen3-8b Qwen/Qwen3-VL-8B-Instruct \
    "$artifact/qwen3_vl_8b_16f.jsonl" qwen8
echo "$(date --iso-8601=seconds) rq1_16f_clip"
run_clip_pair

"$python" tools/evaluate_rq1_semantics.py \
    --manifest "$manifest" \
    --prediction "SlowFast-linear=$artifact/slowfast_source_linear.jsonl" \
    --prediction "CLIP-ViT-B16-16f=$artifact/clip_vit_b16_16f.jsonl" \
    --prediction "Qwen3-VL-8B-16f=$artifact/qwen3_vl_8b_16f.jsonl" \
    --prediction "Qwen3-VL-32B-16f=$artifact/qwen3_vl_32b_16f.jsonl" \
    --output-json "$artifact/rq1_metrics_16f.json" \
    --output-csv "$artifact/rq1_metrics_16f.csv"
touch "$artifact/.complete_16f"
echo "$(date --iso-8601=seconds) rq1_16f_complete"
