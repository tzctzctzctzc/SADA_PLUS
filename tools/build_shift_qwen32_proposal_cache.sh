#!/usr/bin/env bash
set -euo pipefail

# Build the same temperature-calibrated Qwen32 proposal cache used by S5.
# Predictions must already exist under artifacts/matrix_qwen3_vl_32b/<SHIFT>.

shift=${SHIFT:?set SHIFT to S1, S2, S3, S4, S5, or S6}
root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
prediction_root=${PREDICTION_ROOT:-artifacts/matrix_qwen3_vl_32b}
output_root=${CACHE_ROOT:-artifacts/shift_vlm_best_20260912/caches}

cd "$root"
mkdir -p "$output_root/$shift"
source_predictions="$prediction_root/$shift/source_predictions.jsonl"
target_predictions="$prediction_root/$shift/target_predictions.jsonl"
manifest="artifacts/matrix_manifests/$shift/target_train_manifest.jsonl"
temperature="$output_root/$shift/qwen32_source_temperature.json"
cache="$output_root/$shift/qwen32_proposal_semantic.json"

for path in "$source_predictions" "$target_predictions" "$manifest"; do
    test -s "$path" || { echo "missing input: $path" >&2; exit 2; }
done

PYTHONPATH=. "$python" tools/calibrate_vlm_temperature.py \
    --source-predictions "$source_predictions" \
    --output "$temperature"

PYTHONPATH=. "$python" tools/build_proposal_semantic_cache.py \
    --manifest "$manifest" \
    --predictions "$target_predictions" \
    --temperature-calibration-report "$temperature" \
    --output "$cache"

sha256sum "$cache" >"$cache.sha256"
echo "$cache"
