#!/usr/bin/env bash
set -euo pipefail

# Single-variable S5 test: keep a fixed epoch-10 Trident-SADA parent and train
# only a zero-initialized class-agnostic start/end residual from Qwen visual
# feature changes. Target GT and VLM class logits are never used.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
cache=${SADA_QWEN_RELATION_CACHE:-artifacts/qwen_visual_relation_20260910/target_deepstack0_relation.npz}
parent=${SADA_PARENT_CHECKPOINT:-/home1/gyy/vlms/SADA_CLIP_BENCH_20260904_r1/model_results/tridet_s5_parent_e10_repro_all_seed123_20260909/260909_040935_883/saved_ckpt/best_ckpt/checkpoint-epoch10.pt}
artifact_root=${SADA_RELATION_ARTIFACT_ROOT:-artifacts/s5_qwen_relation_boundary_20260912}
gpu=${GPU:-2}
name=${SADA_RUN_NAME:-s5_trident_qwen8_relation_boundary_seed123_20260912}
log="$artifact_root/logs/$name.log"

cd "$root"
mkdir -p "$artifact_root/logs"
test -f "$cache"
test -f "$cache.meta.json"
test -f "$parent"
ulimit -n 65535

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --vlm-relation-cache "$cache" \
    --vlm-relation-mode boundary \
    --vlm-relation-weight 0.2 \
    --vlm-relation-start-epoch 1 \
    --vlm-relation-ramp-epochs 1 \
    --vlm-relation-levels 0,1,2 \
    --vlm-relation-max-gap 30.0 \
    --vlm-relation-boundary-max-delta 1.0 \
    --vlm-relation-boundary-min-strength 0.5 \
    --vlm-relation-init-from-sada "$parent" \
    --vlm-relation-init-weights ema \
    --vlm-relation-freeze-base \
    --stop-after-epoch 5 >"$log" 2>&1
