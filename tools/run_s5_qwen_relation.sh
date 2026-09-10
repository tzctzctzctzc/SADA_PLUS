#!/usr/bin/env bash
set -euo pipefail

# Locked confirmation protocol:
# - S5 Trident-SADA, seed 123, 50 epochs, initialized from epoch 0.
# - The target cache contains no GT and uses prompt-free frozen Qwen3-VL-8B
#   deepstack-0 visual tokens, averaged within each proposal.
# - For temporally adjacent proposals (gap <= 30 s), match the cosine-distance
#   geometry of target SADA FPN features to Qwen visual features.
# - No VLM class logits, class pseudo-labels, or absolute feature projection.
# - Normalize the loss by the target-only Qwen q90-q10 distance range so its
#   gradient does not vanish merely because high-dimensional cosine distances
#   are numerically small.  Weight 0.1, epoch 5 start, 5-epoch linear ramp.
# - Compare against the paired seed-123 from-scratch Trident-SADA control;
#   checkpoint selection remains the original source-validation loss rule.

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
config=${SADA_CONFIG:-configs/old_vs_new/version_final_ckpt.yaml}
feature_root=${SADA_QWEN_FEATURE_ROOT:-artifacts/qwen_visual_relation_20260910}
manifest=${SADA_TARGET_MANIFEST:-artifacts/matrix_manifests/S5/target_train_manifest.jsonl}
cache=${SADA_QWEN_RELATION_CACHE:-$feature_root/target_deepstack0_relation.npz}
artifact_root=${SADA_RELATION_ARTIFACT_ROOT:-artifacts/s5_qwen_relation_20260910}
gpu=${GPU:-4}
name=s5_trident_qwen8_relation_deepstack0_e50_seed123_v2_20260910
log="$artifact_root/logs/${name}.log"

cd "$root"
mkdir -p "$artifact_root/logs"
ulimit -n 65535

expected=$(wc -l < "$manifest")
observed=$(grep -c '"status": "ok"' "$feature_root/target_index.jsonl" || true)
if [[ "$observed" -ne "$expected" ]]; then
    echo "Qwen target feature cache is incomplete: $observed/$expected" >&2
    exit 1
fi

PYTHONPATH=. "$python" tools/build_vlm_relation_cache.py \
    --manifest "$manifest" \
    --manifest-meta "$manifest.meta.json" \
    --feature-index "$feature_root/target_index.jsonl" \
    --feature-root "$feature_root/target_features" \
    --output "$cache" \
    --feature-layer deepstack_0

CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=. "$python" -u train_da.py \
    -c "$config" -n "$name" -s 123 -i 0 -d --trident-head \
    --vlm-relation-cache "$cache" \
    --vlm-relation-weight 0.1 \
    --vlm-relation-start-epoch 5 \
    --vlm-relation-ramp-epochs 5 \
    --vlm-relation-levels 0,1,2 \
    --vlm-relation-max-gap 30.0 >"$log" 2>&1
