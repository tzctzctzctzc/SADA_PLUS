#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/vepfs/vlms/SADA_PLUS}"
PYTHON_BIN="${PYTHON_BIN:-/vepfs/vlms/envs/rq1-cu121/bin/python}"
VIDEO_ROOT="${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}"
CACHE_ROOT="${CACHE_ROOT:-/vepfs/vlms/cache/rq1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/artifacts/rq1_semantic_20260915}"
FRAME_COUNT="${FRAME_COUNT:-8}"
FRAME_WORKERS="${FRAME_WORKERS:-4}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/vepfs/vlms/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}/logs" "${CACHE_ROOT}/frames" /vepfs/vlms/cache/clip
cd "${REPO_ROOT}"

MANIFEST="${OUTPUT_ROOT}/rq1_val_gt_segments.jsonl"
FRAME_NAMESPACE="rq1_${FRAME_COUNT}f"

"${PYTHON_BIN}" tools/build_rq1_semantic_manifest.py \
  --repo-root "${REPO_ROOT}" \
  --output "${MANIFEST}" \
  --frame-count "${FRAME_COUNT}" \
  > "${OUTPUT_ROOT}/logs/build_manifest.log" 2>&1

"${PYTHON_BIN}" tools/prepare_rq1_frames.py \
  --manifest "${MANIFEST}" \
  --video-root "${VIDEO_ROOT}" \
  --frame-cache "${CACHE_ROOT}/frames" \
  --cache-namespace "${FRAME_NAMESPACE}" \
  --frame-count "${FRAME_COUNT}" \
  --workers "${FRAME_WORKERS}" \
  > "${OUTPUT_ROOT}/logs/prepare_frames.log" 2>&1

"${PYTHON_BIN}" tools/run_rq1_slowfast_probe.py \
  --repo-root "${REPO_ROOT}" \
  --manifest "${MANIFEST}" \
  --output "${OUTPUT_ROOT}/slowfast_source_linear.jsonl" \
  > "${OUTPUT_ROOT}/logs/slowfast_source_linear.log" 2>&1 &
SLOWFAST_PID=$!

(
  CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" tools/run_rq1_qwen.py \
    --manifest "${MANIFEST}" \
    --frame-cache "${CACHE_ROOT}/frames" \
    --cache-namespace "${FRAME_NAMESPACE}" \
    --output "${OUTPUT_ROOT}/qwen3_vl_32b.jsonl" \
    --model-family qwen3-32b \
    --model-path Qwen/Qwen3-VL-32B-Instruct \
    --frame-count "${FRAME_COUNT}"
) > "${OUTPUT_ROOT}/logs/qwen3_vl_32b.log" 2>&1 &
QWEN32_PID=$!

(
  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/run_rq1_clip.py \
    --manifest "${MANIFEST}" \
    --frame-cache "${CACHE_ROOT}/frames" \
    --cache-namespace "${FRAME_NAMESPACE}" \
    --output "${OUTPUT_ROOT}/clip_vit_b16.jsonl" \
    --model ViT-B/16 \
    --model-cache /vepfs/vlms/cache/clip \
    --frame-count "${FRAME_COUNT}"

  CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" tools/run_rq1_qwen.py \
    --manifest "${MANIFEST}" \
    --frame-cache "${CACHE_ROOT}/frames" \
    --cache-namespace "${FRAME_NAMESPACE}" \
    --output "${OUTPUT_ROOT}/qwen3_vl_8b.jsonl" \
    --model-family qwen3-8b \
    --model-path Qwen/Qwen3-VL-8B-Instruct \
    --frame-count "${FRAME_COUNT}"
) > "${OUTPUT_ROOT}/logs/gpu0_clip_then_qwen8.log" 2>&1 &
GPU0_PID=$!

printf '%s\n' "${SLOWFAST_PID}" > "${OUTPUT_ROOT}/slowfast.pid"
printf '%s\n' "${QWEN32_PID}" > "${OUTPUT_ROOT}/qwen32.pid"
printf '%s\n' "${GPU0_PID}" > "${OUTPUT_ROOT}/gpu0.pid"

wait "${SLOWFAST_PID}"
wait "${QWEN32_PID}"
wait "${GPU0_PID}"

"${PYTHON_BIN}" tools/evaluate_rq1_semantics.py \
  --manifest "${MANIFEST}" \
  --prediction "SlowFast-linear=${OUTPUT_ROOT}/slowfast_source_linear.jsonl" \
  --prediction "CLIP-ViT-B16=${OUTPUT_ROOT}/clip_vit_b16.jsonl" \
  --prediction "Qwen3-VL-8B=${OUTPUT_ROOT}/qwen3_vl_8b.jsonl" \
  --prediction "Qwen3-VL-32B=${OUTPUT_ROOT}/qwen3_vl_32b.jsonl" \
  --output-json "${OUTPUT_ROOT}/rq1_metrics.json" \
  --output-csv "${OUTPUT_ROOT}/rq1_metrics.csv" \
  > "${OUTPUT_ROOT}/logs/evaluate.log" 2>&1

touch "${OUTPUT_ROOT}/.complete"
