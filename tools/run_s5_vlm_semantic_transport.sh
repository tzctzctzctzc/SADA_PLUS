#!/usr/bin/env bash
set -euo pipefail

# Confirmatory protocol locked before implementation:
# - Trident-SADA, S5, seed 123, epochs 1--15, trained from epoch 0.
# - Fixed Qwen3-VL-32B target-train evidence, confidence >= 0.7, FPN L0--L2.
# - Each cached absolute feature index is a sparse semantic point; proposal
#   boundaries are not copied into the student.  Within the architecture-defined
#   temporal radius, feature/temporal affinity selects five reliable neighbours.
# - Propagated VLM classes replace SADA target class routes only on selected
#   points.  All other points retain the original SADA route.
# - Source-GT class prototypes supervise a detached semantic projector.  The
#   propagated target points are transported toward the matching source prototype
#   through a zero-initialized classification-feature adapter.
# - Regression and Trident boundary heads always consume the untouched SADA FPN.
# - Target GT is forbidden in training and selection; it may be read only by a
#   post-hoc mechanism audit after this protocol is fixed.
# - Advance to three seeds only if target-val Average mAP exceeds 65.87.  Treat
#   >=66.37 as the requested large positive signal; otherwise stop or redesign.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
cache=${SADA_QWEN32_CACHE:-artifacts/s5_strategy_matrix/caches/qwen32_proposal_semantic.json}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
artifact_root=${SADA_TRANSPORT_ARTIFACT_ROOT:-artifacts/s5_vlm_semantic_transport_20260909}
gpu=${GPU:-2}
name=s5_trident_qwen32_semantic_transport_e15_seed123_20260909
log="$artifact_root/logs/${name}.log"

cd "$root"
mkdir -p "$artifact_root/logs"
ulimit -n 65535

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --proposal-semantic-mode vlm_semantic_transport \
    --proposal-semantic-target-cache "$cache" \
    --proposal-semantic-external-selection \
    --proposal-semantic-target-start-epoch 1 \
    --proposal-semantic-levels 0,1,2 \
    --proposal-semantic-max-proposals 256 \
    --proposal-semantic-confidence 0.7 \
    --proposal-semantic-source-weight 1.0 \
    --proposal-semantic-alignment-weight 0.1 \
    --proposal-semantic-direct-weight 0.2 \
    --proposal-semantic-transport-hidden-dim 128 \
    --proposal-semantic-transport-topk 5 \
    --proposal-semantic-transport-temperature 0.1 \
    --proposal-semantic-transport-max-residual 0.1 \
    --diagnostic-alignment-levels 0,1,2 \
    --stop-after-epoch 15 >"$log" 2>&1

find "model_results/$name" \
    -path '*/saved_ckpt/latest_ckpt/checkpoint-epoch15.pt' \
    -type f -delete
