#!/usr/bin/env bash
set -euo pipefail

root=${RQ1_ROOT:-/vepfs/vlms/SADA_PLUS}
python=${RQ1_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
sada_python=${SADA_PYTHON:-/vepfs/vlms/envs/sada/bin/python}
base_artifact=${RQ1_BASE_ARTIFACT:-$root/artifacts/rq1_semantic_20260915}
artifact=${RQ1_PROBE_ARTIFACT:-$base_artifact/source_probe_8f}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq1/frames}
gpu0=${GPU0:-0}
gpu1=${GPU1:-1}

if [[ "$gpu0" == "$gpu1" ]]; then
    echo "GPU0 and GPU1 must be different" >&2
    exit 2
fi

mkdir -p "$artifact/logs" "$artifact/features"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

manifest=$artifact/rq1_source_probe_segments.jsonl
eval_manifest=$base_artifact/rq1_val_gt_segments.jsonl

"$python" tools/build_rq1_source_probe_manifest.py \
    --repo-root "$root" --output "$manifest" --frame-count 8 \
    >"$artifact/logs/build_manifest.log" 2>&1

"$sada_python" tools/download_epic_videos_from_manifest.py \
    --manifest "$manifest" --video-root "$video_root" \
    --epic55-splits /vepfs/vlms/data/EPIC_KITCHENS/epic_55_splits.csv \
    --md5-csv /vepfs/vlms/data/EPIC_KITCHENS/epic_md5.csv \
    --workers 8 --connections 16 --retries 5 --use-http \
    >"$artifact/logs/download_videos.log" 2>&1

"$sada_python" tools/prepare_rq1_frames.py \
    --manifest "$manifest" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace rq1_8f \
    --frame-count 8 --workers 48 \
    >"$artifact/logs/prepare_frames.log" 2>&1

(
    CUDA_VISIBLE_DEVICES=$gpu0 "$python" tools/extract_rq1_clip_features.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace rq1_8f --frame-count 8 \
        --output-prefix "$artifact/features/clip_vit_b16_static" \
        --model ViT-B/16 --model-cache /vepfs/vlms/cache/clip \
        --segment-batch-size 64 --image-batch-size 256

    CUDA_VISIBLE_DEVICES=$gpu0 "$python" tools/extract_rq1_qwen_features.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace rq1_8f --frame-count 8 \
        --output-prefix "$artifact/features/qwen3_vl_8b_static" \
        --model-path Qwen/Qwen3-VL-8B-Instruct --offline \
        --segment-batch-size 16
) >"$artifact/logs/gpu0_clip_then_qwen8.log" 2>&1 &
gpu0_pid=$!

(
    CUDA_VISIBLE_DEVICES=$gpu1 "$python" tools/extract_rq1_qwen_features.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace rq1_8f --frame-count 8 \
        --output-prefix "$artifact/features/qwen3_vl_32b_static" \
        --model-path Qwen/Qwen3-VL-32B-Instruct --offline \
        --segment-batch-size 16
) >"$artifact/logs/gpu1_qwen32.log" 2>&1 &
gpu1_pid=$!

printf '%s\n' "$gpu0_pid" >"$artifact/gpu0.pid"
printf '%s\n' "$gpu1_pid" >"$artifact/gpu1.pid"
wait "$gpu0_pid"
wait "$gpu1_pid"

"$python" tools/run_rq1_frozen_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$artifact/features/clip_vit_b16_static" \
    --model-name CLIP-ViT-B16-source-linear \
    --output "$artifact/clip_vit_b16_source_linear.jsonl" \
    >"$artifact/logs/probe_clip.log" 2>&1

"$python" tools/run_rq1_frozen_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$artifact/features/qwen3_vl_8b_static" \
    --model-name Qwen3-VL-8B-source-linear \
    --output "$artifact/qwen3_vl_8b_source_linear.jsonl" \
    >"$artifact/logs/probe_qwen8.log" 2>&1

"$python" tools/run_rq1_frozen_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$artifact/features/qwen3_vl_32b_static" \
    --model-name Qwen3-VL-32B-source-linear \
    --output "$artifact/qwen3_vl_32b_source_linear.jsonl" \
    >"$artifact/logs/probe_qwen32.log" 2>&1

"$python" tools/evaluate_rq1_semantics.py \
    --manifest "$eval_manifest" \
    --prediction "SlowFast-source-linear=$base_artifact/slowfast_source_linear.jsonl" \
    --prediction "CLIP-ViT-B16-zero-shot=$base_artifact/clip_vit_b16.jsonl" \
    --prediction "CLIP-ViT-B16-source-linear=$artifact/clip_vit_b16_source_linear.jsonl" \
    --prediction "Qwen3-VL-8B-zero-shot=$base_artifact/qwen3_vl_8b.jsonl" \
    --prediction "Qwen3-VL-8B-source-linear=$artifact/qwen3_vl_8b_source_linear.jsonl" \
    --prediction "Qwen3-VL-32B-zero-shot=$base_artifact/qwen3_vl_32b.jsonl" \
    --prediction "Qwen3-VL-32B-source-linear=$artifact/qwen3_vl_32b_source_linear.jsonl" \
    --output-json "$artifact/rq1_metrics_expanded.json" \
    --output-csv "$artifact/rq1_metrics_expanded.csv" \
    >"$artifact/logs/evaluate.log" 2>&1

touch "$artifact/.complete"
echo "$(date --iso-8601=seconds) rq1_source_probes_complete"
