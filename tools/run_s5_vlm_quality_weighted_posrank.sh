#!/usr/bin/env bash
set -euo pipefail

# One-seed mechanism gate. Qwen8 intermediate features only reweight the
# frozen Qwen32 positive-rank supervision; they never backpropagate into SADA.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/vepfs/vlms/envs/sada_py39/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
cache=${SADA_QUALITY_CACHE:-artifacts/vlm_quality_weighting_20260912/qwen32_semantic_qwen8_quality_rank.json}
parent=${SADA_PARENT:-artifacts/vlm_quality_weighting_20260912/checkpoint-epoch10-seed123.pt}
artifact_root=${SADA_QUALITY_ARTIFACT_ROOT:-artifacts/vlm_quality_weighting_20260912}
gpu=${GPU:-1}
name=s5_trident_qwen32_qwen8_quality_posrank_seed123_20260912
log="$artifact_root/logs/${name}.log"

cd "$root"
mkdir -p "$artifact_root/logs"
test -f "$cache"
test -f "$parent"
"$python" - "$cache" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
quality = payload.get("quality_weighting", {})
assert quality.get("protocol_version") == "vlm-source-quality-rank-v1"
assert quality.get("target_gt_usage") == "forbidden"
assert quality.get("source_count") == 200
assert quality.get("target_count") == 1570
PY

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --proposal-semantic-continue-from-sada "$parent" \
    --proposal-semantic-init-weights student \
    --proposal-semantic-mode vlm_logit_confident \
    --proposal-semantic-target-cache "$cache" \
    --proposal-semantic-external-selection \
    --proposal-semantic-target-start-epoch 11 \
    --proposal-semantic-levels 0,1,2 \
    --proposal-semantic-max-proposals 256 \
    --proposal-semantic-confidence 0.7 \
    --proposal-semantic-source-weight 0.0 \
    --proposal-semantic-alignment-weight 0.0 \
    --proposal-semantic-direct-weight 0.2 \
    --proposal-semantic-logit-max-delta 1.0 \
    --proposal-semantic-freeze-base \
    --proposal-semantic-direct-loss positive_bce_rank_quality \
    --proposal-semantic-direct-pooling top1 \
    --diagnostic-alignment-levels 0,1,2 \
    --stop-after-epoch 15 >"$log" 2>&1
