#!/usr/bin/env bash
set -euo pipefail

# S5 seed-123 single-variable test:
# original from-scratch Trident-SADA + target-only proposal/Qwen feature alignment.
# The frozen Qwen cache contains no target GT. Checkpoints remain selected by
# the original source-validation loss; target validation is evaluation only.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
cache=${SADA_QWEN_RELATION_CACHE:-artifacts/qwen_visual_relation_20260910/target_deepstack0_relation.npz}
artifact_root=${SADA_FEATURE_ALIGN_ARTIFACT_ROOT:-artifacts/s5_qwen_feature_alignment_20260912}
gpu=${GPU:-0}
name=${SADA_RUN_NAME:-s5_trident_qwen8_feature_align_e50_seed123_20260912}
log="$artifact_root/logs/${name}.log"

cd "$root"
mkdir -p "$artifact_root/logs"
ulimit -n 65535

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --vlm-relation-cache "$cache" \
    --vlm-relation-mode feature \
    --vlm-relation-weight "${SADA_FEATURE_ALIGN_WEIGHT:-0.1}" \
    --vlm-relation-temperature "${SADA_FEATURE_ALIGN_TEMPERATURE:-0.07}" \
    --vlm-relation-start-epoch 1 \
    --vlm-relation-ramp-epochs 5 \
    --vlm-relation-levels 0,1,2 >"$log" 2>&1
