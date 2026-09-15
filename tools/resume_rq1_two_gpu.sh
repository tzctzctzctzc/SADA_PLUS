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
download_pid=${DOWNLOAD_PID:-}
initial_qwen32_pid=${INITIAL_QWEN32_PID:-}
prefill_clip_pid=${PREFILL_CLIP_PID:-}
prefill_qwen32_pid0=${PREFILL_QWEN32_PID0:-}
prefill_qwen32_pid1=${PREFILL_QWEN32_PID1:-}
skip_data_recovery=${SKIP_DATA_RECOVERY:-0}

if [[ "$gpu0" == "$gpu1" ]]; then
    echo "GPU0 and GPU1 must be different" >&2
    exit 2
fi

manifest=$artifact/rq1_val_gt_segments.jsonl
recovery=$artifact/two_gpu_recovery
mkdir -p "$recovery/logs"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

wait_external() {
    local name=$1 pid=$2
    [[ -z "$pid" ]] && return
    while kill -0 "$pid" 2>/dev/null; do
        echo "$(date --iso-8601=seconds) waiting $name pid=$pid"
        sleep 30
    done
}

video_ids=(
    P01_02 P01_11 P02_10 P02_134 P04_07 P04_10 P04_103 P04_19 P04_22
    P06_02 P07_15 P09_01 P09_105 P10_01 P15_05 P26_24 P27_06 P27_07
    P29_03 P31_07 P31_12 P32_05
)

if [[ "$skip_data_recovery" != 1 ]]; then
    wait_external download "$download_pid"
    wait_external initial_qwen32 "$initial_qwen32_pid"
    wait_external prefill_clip "$prefill_clip_pid"
    wait_external prefill_qwen32_0 "$prefill_qwen32_pid0"
    wait_external prefill_qwen32_1 "$prefill_qwen32_pid1"

    echo "$(date --iso-8601=seconds) verify_and_resume_downloads"
    "$sada_python" tools/download_epic_manifest_videos.py \
        --scenario-root "$artifact/download_scenario" \
        --video-root "$video_root" \
        --epic55-splits /vepfs/vlms/data/EPIC_KITCHENS/epic_55_splits.csv \
        --md5-csv /vepfs/vlms/data/EPIC_KITCHENS/epic_md5.csv \
        --shifts S1 --video-ids "${video_ids[@]}" \
        --workers 8 --connections 8 --retries 5

    echo "$(date --iso-8601=seconds) rebuild_recovered_frames"
    frame_args=()
    for video_id in "${video_ids[@]}"; do
        frame_args+=(--video-id "$video_id")
    done
    "$sada_python" tools/prepare_rq1_frames.py \
        --manifest "$manifest" --video-root "$video_root" \
        --frame-cache "$frame_cache" --cache-namespace rq1_8f \
        --frame-count 8 --workers 8 --force "${frame_args[@]}"
fi

echo "$(date --iso-8601=seconds) verify_all_frames"
"$sada_python" tools/prepare_rq1_frames.py \
    --manifest "$manifest" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace rq1_8f \
    --frame-count 8 --workers 8

shard_pending() {
    local output=$1 prefix=$2
    shift 2
    inputs=()
    for path in "$@"; do
        [[ -s "$path" ]] && inputs+=(--input "$path")
    done
    "$python" tools/merge_and_shard_rq1.py \
        --manifest "$manifest" "${inputs[@]}" --output "$output" \
        --frame-root "$frame_cache/rq1_8f" --frame-count 8 \
        --shard-prefix "$prefix" --num-shards 2
}

run_qwen_pair() {
    local family=$1 model_path=$2 output=$3 tag=$4
    local prefix=$recovery/${tag}_manifest
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/${tag}_output_0.jsonl" \
        "$recovery/${tag}_output_1.jsonl"
    pids=()
    for index in 0 1; do
        shard=${prefix}_${index}.jsonl
        [[ ! -s "$shard" ]] && continue
        gpu=$gpu0
        [[ "$index" == 1 ]] && gpu=$gpu1
        shard_output=$recovery/${tag}_output_${index}.jsonl
        CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq1_qwen.py \
            --manifest "$shard" --frame-cache "$frame_cache" \
            --cache-namespace rq1_8f --output "$shard_output" \
            --model-family "$family" --model-path "$model_path" \
            --offline --frame-count 8 \
            >"$recovery/logs/${tag}_${index}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/${tag}_output_0.jsonl" "$recovery/${tag}_output_1.jsonl"
}

run_clip_pair() {
    local output=$artifact/clip_vit_b16.jsonl
    local prefix=$recovery/clip_manifest
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/clip_prefill.jsonl" \
        "$recovery/clip_output_0.jsonl" "$recovery/clip_output_1.jsonl"
    pids=()
    for index in 0 1; do
        shard=${prefix}_${index}.jsonl
        [[ ! -s "$shard" ]] && continue
        gpu=$gpu0
        [[ "$index" == 1 ]] && gpu=$gpu1
        shard_output=$recovery/clip_output_${index}.jsonl
        CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq1_clip.py \
            --manifest "$shard" --frame-cache "$frame_cache" \
            --cache-namespace rq1_8f --output "$shard_output" \
            --model ViT-B/16 --model-cache /vepfs/vlms/cache/clip \
            --frame-count 8 --segment-batch-size 64 --image-batch-size 128 \
            >"$recovery/logs/clip_${index}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    shard_pending "$output" "$prefix" \
        "$output" "$recovery/clip_prefill.jsonl" \
        "$recovery/clip_output_0.jsonl" "$recovery/clip_output_1.jsonl"
}

echo "$(date --iso-8601=seconds) complete_qwen32"
qwen32_prefill_inputs=(--input "$artifact/qwen3_vl_32b.jsonl")
for path in "$recovery"/qwen32_prefill_output_0.jsonl \
            "$recovery"/qwen32_prefill_output_1.jsonl \
            "$recovery"/qwen32_output_0.jsonl \
            "$recovery"/qwen32_output_1.jsonl; do
    [[ -s "$path" ]] && qwen32_prefill_inputs+=(--input "$path")
done
"$python" tools/merge_and_shard_rq1.py \
    --manifest "$manifest" "${qwen32_prefill_inputs[@]}" \
    --output "$artifact/qwen3_vl_32b.jsonl"
run_qwen_pair qwen3-32b Qwen/Qwen3-VL-32B-Instruct \
    "$artifact/qwen3_vl_32b.jsonl" qwen32
echo "$(date --iso-8601=seconds) complete_qwen8"
run_qwen_pair qwen3-8b Qwen/Qwen3-VL-8B-Instruct \
    "$artifact/qwen3_vl_8b.jsonl" qwen8
echo "$(date --iso-8601=seconds) complete_clip"
run_clip_pair

"$python" - "$manifest" \
    "$artifact/slowfast_source_linear.jsonl" \
    "$artifact/clip_vit_b16.jsonl" \
    "$artifact/qwen3_vl_8b.jsonl" \
    "$artifact/qwen3_vl_32b.jsonl" <<'PY'
import json
import sys

def ids(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line)["candidate_id"] for line in stream if line.strip()]

expected = set(ids(sys.argv[1]))
for index, path in enumerate(sys.argv[2:]):
    current = ids(path)
    assert set(current) == expected, (path, len(current), len(expected))
    if index > 0:
        assert len(current) == len(set(current)), (path, "duplicate candidate_id")
    print(path, len(current), "complete")
PY

"$python" tools/evaluate_rq1_semantics.py \
    --manifest "$manifest" \
    --prediction "SlowFast-linear=$artifact/slowfast_source_linear.jsonl" \
    --prediction "CLIP-ViT-B16=$artifact/clip_vit_b16.jsonl" \
    --prediction "Qwen3-VL-8B=$artifact/qwen3_vl_8b.jsonl" \
    --prediction "Qwen3-VL-32B=$artifact/qwen3_vl_32b.jsonl" \
    --output-json "$artifact/rq1_metrics.json" \
    --output-csv "$artifact/rq1_metrics.csv"
touch "$artifact/.complete"
echo "$(date --iso-8601=seconds) rq1_complete"
