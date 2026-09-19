#!/usr/bin/env bash
set -euo pipefail

root=${RQ2_ROOT:-/vepfs/vlms/SADA_PLUS}
artifact=${RQ2_ARTIFACT:-$root/artifacts/rq2_probe_20260919}
python=${RQ2_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
video_python=${RQ2_VIDEO_PYTHON:-/vepfs/vlms/envs/sada/bin/python}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq2/frames}
gpu=${RQ2_GPU:-3}
shift=${RQ2_SHIFT:-S1}
instances_per_class=${RQ2_INSTANCES_PER_CLASS:-5}

manifest=$artifact/rq2_probe_manifest.jsonl
pilot=$artifact/rq2_probe_${shift}_source_train_pilot.jsonl
namespace=rq2_probe_${shift}_source_train_pilot_2fps
feature_root=$artifact/features/qwen3_vl_8b_${shift}_pilot
feature_index=$artifact/qwen3_vl_8b_${shift}_pilot_index.jsonl

mkdir -p "$artifact/logs" "$feature_root"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$manifest" ]]; then
    "$python" tools/build_rq2_probe_manifest.py \
        --repo-root "$root" --output "$manifest" --fps 2 --max-frames 64
fi

"$python" tools/select_rq2_probe_pilot.py \
    --manifest "$manifest" --output "$pilot" --shift "$shift" \
    --role source_train --instances-per-class "$instances_per_class"

"$video_python" tools/prepare_rq2_probe_frames.py \
    --manifest "$pilot" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace "$namespace" --workers 8 \
    --decode-mode seek \
    >"$artifact/logs/${shift}_prepare_frames.log" 2>&1

CUDA_VISIBLE_DEVICES=$gpu "$python" tools/extract_rq2_qwen_frame_features.py \
    --manifest "$pilot" --frame-cache "$frame_cache" \
    --cache-namespace "$namespace" --output-index "$feature_index" \
    --feature-root "$feature_root" --model-path Qwen/Qwen3-VL-8B-Instruct \
    --offline --min-pixels $((224 * 224)) --max-pixels $((224 * 224)) \
    --max-batch-frames 256 --loader-workers 8 \
    >"$artifact/logs/${shift}_qwen3_vl_8b_extract.log" 2>&1

"$python" tools/run_rq2_temporal_probe_pilot.py \
    --manifest "$pilot" --feature-index "$feature_index" \
    --feature-root "$feature_root" \
    --output "$artifact/qwen3_vl_8b_${shift}_probe_pilot.json" \
    --folds 3 --learning-rate 3e-4 --weight-decay 1e-3 \
    --max-epochs 30 --patience 5 --seed 0 --device cpu \
    >"$artifact/logs/${shift}_qwen3_vl_8b_probe.log" 2>&1

touch "$artifact/.qwen3_vl_8b_${shift}_pilot_complete"
