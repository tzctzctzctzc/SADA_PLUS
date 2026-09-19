#!/usr/bin/env bash
set -euo pipefail

root=${RQ2_ROOT:-/vepfs/vlms/SADA_PLUS}
artifact=${RQ2_ARTIFACT:-$root/artifacts/rq2_probe_20260919}
python=${RQ2_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
video_python=${RQ2_VIDEO_PYTHON:-/vepfs/vlms/envs/sada/bin/python}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq2/frames}
gpu_a=${RQ2_GPU_A:-0}
gpu_b=${RQ2_GPU_B:-1}
shift=${RQ2_SHIFT:-S1}
instances_per_class=${RQ2_INSTANCES_PER_CLASS:-20}

if [[ "$gpu_a" == "$gpu_b" ]]; then
    echo "RQ2_GPU_A and RQ2_GPU_B must differ" >&2
    exit 2
fi

manifest=$artifact/rq2_probe_manifest.jsonl
tag=${shift}_n${instances_per_class}
pilot=$artifact/rq2_probe_${tag}_source_train.jsonl
namespace=${RQ2_CACHE_NAMESPACE:-rq2_probe_${shift}_source_train_pilot_2fps}
feature_root=$artifact/features/qwen3_vl_8b_${tag}
index_a=$artifact/qwen3_vl_8b_${tag}_shard0.jsonl
index_b=$artifact/qwen3_vl_8b_${tag}_shard1.jsonl

mkdir -p "$artifact/logs" "$feature_root"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$python" tools/select_rq2_probe_pilot.py \
    --manifest "$manifest" --output "$pilot" --shift "$shift" \
    --role source_train --instances-per-class "$instances_per_class"

"$video_python" tools/prepare_rq2_probe_frames.py \
    --manifest "$pilot" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace "$namespace" --workers 8 \
    --decode-mode seek \
    >"$artifact/logs/${tag}_prepare_frames.log" 2>&1

pids=()
for shard in 0 1; do
    gpu=$gpu_a
    index=$index_a
    [[ "$shard" == 1 ]] && gpu=$gpu_b
    [[ "$shard" == 1 ]] && index=$index_b
    CUDA_VISIBLE_DEVICES=$gpu "$python" tools/extract_rq2_qwen_frame_features.py \
        --manifest "$pilot" --frame-cache "$frame_cache" \
        --cache-namespace "$namespace" --output-index "$index" \
        --feature-root "$feature_root" --model-path Qwen/Qwen3-VL-8B-Instruct \
        --offline --min-pixels $((224 * 224)) --max-pixels $((224 * 224)) \
        --max-batch-frames 256 --loader-workers 8 \
        --shard-count 2 --shard-index "$shard" \
        >"$artifact/logs/${tag}_qwen3_vl_8b_shard${shard}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done

"$python" tools/run_rq2_temporal_probe_pilot.py \
    --manifest "$pilot" --feature-index "$index_a" --feature-index "$index_b" \
    --feature-root "$feature_root" \
    --output "$artifact/qwen3_vl_8b_${tag}_probe.json" \
    --folds 3 --learning-rate 3e-4 --weight-decay 1e-3 \
    --max-epochs 30 --patience 5 --seed 0 --device cpu \
    >"$artifact/logs/${tag}_qwen3_vl_8b_probe.log" 2>&1

touch "$artifact/.qwen3_vl_8b_${tag}_complete"
