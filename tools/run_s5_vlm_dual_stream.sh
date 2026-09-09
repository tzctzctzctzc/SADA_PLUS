#!/usr/bin/env bash
set -euo pipefail

# Confirmatory protocol locked before implementation:
# - Trident-SADA, S5, seed 123, epochs 1--15.
# - The detector/UDA stream receives no gradient or gradient-clip coupling from VLM.
# - A detached long/short semantic stream is source-GT trained and target-Qwen32
#   trained from epoch 1; its zero-initialized residual only changes target class
#   logits and is gated by detached class-agnostic actionness.
# - Advance only if target-val Average mAP exceeds the paired 65.87 reference.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
cache=${SADA_QWEN32_CACHE:-artifacts/s5_strategy_matrix/caches/qwen32_proposal_semantic.json}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
artifact_root=${SADA_DUAL_ARTIFACT_ROOT:-artifacts/s5_vlm_dual_stream_20260909}
gpu=${GPU:-2}
name=s5_trident_qwen32_dual_stream_e15_seed123_20260909
log="$artifact_root/logs/${name}.log"

cd "$root"
mkdir -p "$artifact_root/logs"
ulimit -n 65535

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --proposal-semantic-mode vlm_dual_stream \
    --proposal-semantic-target-cache "$cache" \
    --proposal-semantic-external-selection \
    --proposal-semantic-target-start-epoch 1 \
    --proposal-semantic-levels 0,1,2 \
    --proposal-semantic-max-proposals 256 \
    --proposal-semantic-confidence 0.7 \
    --proposal-semantic-source-weight 1.0 \
    --proposal-semantic-alignment-weight 0.0 \
    --proposal-semantic-direct-weight 0.2 \
    --proposal-semantic-direct-loss positive_bce_rank_teacher_score \
    --proposal-semantic-direct-pooling top1 \
    --proposal-semantic-logit-max-delta 1.0 \
    --proposal-semantic-dual-hidden-dim 128 \
    --proposal-semantic-dual-global-tokens 32 \
    --diagnostic-alignment-levels 0,1,2 \
    --stop-after-epoch 15 >"$log" 2>&1

find "model_results/$name" \
    -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
    -type f -delete
