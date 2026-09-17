#!/usr/bin/env bash
set -euo pipefail

root=${RQ1_ROOT:-/vepfs/vlms/SADA_PLUS}
python=${RQ1_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
base_artifact=${RQ1_BASE_ARTIFACT:-$root/artifacts/rq1_semantic_20260915}
probe_artifact=${RQ1_PROBE_ARTIFACT:-$base_artifact/source_probe_8f}
artifact=${RQ1_EGOVIDEO_ARTIFACT:-$base_artifact/egovideo_l_8f}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq1/frames}
egohod_root=${EGOHOD_ROOT:-/vepfs/vlms/external/EgoHOD}
checkpoint=${EGOVIDEO_CHECKPOINT:-/vepfs/vlms/cache/huggingface/models--Jazzcharles--EgoVideo/snapshots/3c1781d01b74e3cbacd3129eda7025b9ee1e3fa9/large_best.pt}
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

manifest=$probe_artifact/rq1_source_probe_segments.jsonl
eval_manifest=$base_artifact/rq1_val_gt_segments.jsonl
feature_prefix=$artifact/features/egovideo_l_joint_8f
text_features=$artifact/features/egovideo_l_text.npy

"$python" tools/extract_rq1_egovideo_features.py \
    --manifest "$manifest" --frame-cache "$frame_cache" \
    --output-prefix "$feature_prefix" --checkpoint "$checkpoint" \
    --egohod-root "$egohod_root" --frame-count 8 --initialize-only \
    >"$artifact/logs/initialize.log" 2>&1

CUDA_VISIBLE_DEVICES=$gpu0 "$python" tools/extract_rq1_egovideo_features.py \
    --manifest "$manifest" --frame-cache "$frame_cache" \
    --cache-namespace rq1_8f --output-prefix "$feature_prefix" \
    --checkpoint "$checkpoint" --egohod-root "$egohod_root" \
    --frame-count 8 --batch-size 16 --shard-count 2 --shard-index 0 \
    --text-features-output "$text_features" \
    >"$artifact/logs/gpu0_shard0.log" 2>&1 &
gpu0_pid=$!

CUDA_VISIBLE_DEVICES=$gpu1 "$python" tools/extract_rq1_egovideo_features.py \
    --manifest "$manifest" --frame-cache "$frame_cache" \
    --cache-namespace rq1_8f --output-prefix "$feature_prefix" \
    --checkpoint "$checkpoint" --egohod-root "$egohod_root" \
    --frame-count 8 --batch-size 16 --shard-count 2 --shard-index 1 \
    >"$artifact/logs/gpu1_shard1.log" 2>&1 &
gpu1_pid=$!

printf '%s\n' "$gpu0_pid" >"$artifact/gpu0.pid"
printf '%s\n' "$gpu1_pid" >"$artifact/gpu1.pid"
wait "$gpu0_pid"
wait "$gpu1_pid"

"$python" tools/run_rq1_embedding_zero_shot.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$feature_prefix" --text-features "$text_features" \
    --model-name EgoVideo-L-zero-shot \
    --output "$artifact/egovideo_l_zero_shot.jsonl" \
    >"$artifact/logs/zero_shot.log" 2>&1

"$python" tools/run_rq1_frozen_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$feature_prefix" \
    --model-name EgoVideo-L-source-linear \
    --output "$artifact/egovideo_l_source_linear.jsonl" \
    >"$artifact/logs/source_linear.log" 2>&1

"$python" tools/evaluate_rq1_semantics.py \
    --manifest "$eval_manifest" \
    --prediction "SlowFast-source-linear=$base_artifact/slowfast_source_linear.jsonl" \
    --prediction "CLIP-ViT-B16-zero-shot=$base_artifact/clip_vit_b16.jsonl" \
    --prediction "CLIP-ViT-B16-source-linear=$probe_artifact/clip_vit_b16_source_linear.jsonl" \
    --prediction "Qwen3-VL-8B-zero-shot=$base_artifact/qwen3_vl_8b.jsonl" \
    --prediction "Qwen3-VL-8B-source-linear=$probe_artifact/qwen3_vl_8b_source_linear.jsonl" \
    --prediction "Qwen3-VL-32B-zero-shot=$base_artifact/qwen3_vl_32b.jsonl" \
    --prediction "Qwen3-VL-32B-source-linear=$probe_artifact/qwen3_vl_32b_source_linear.jsonl" \
    --prediction "EgoVideo-L-zero-shot=$artifact/egovideo_l_zero_shot.jsonl" \
    --prediction "EgoVideo-L-source-linear=$artifact/egovideo_l_source_linear.jsonl" \
    --output-json "$artifact/rq1_metrics_with_egovideo.json" \
    --output-csv "$artifact/rq1_metrics_with_egovideo.csv" \
    >"$artifact/logs/evaluate.log" 2>&1

touch "$artifact/.complete"
echo "$(date --iso-8601=seconds) rq1_egovideo_complete"
