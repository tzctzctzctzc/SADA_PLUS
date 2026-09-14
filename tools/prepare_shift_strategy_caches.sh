#!/usr/bin/env bash
set -euo pipefail

root=${SADA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
python=${SADA_PYTHON:-/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python}
artifact_root=${BENCHMARK_ARTIFACT_ROOT:-artifacts/fromzero_sada_strategy_matrix_20260914}
IFS=',' read -r -a shifts <<<"${SHIFTS:-S3,S4,S5}"
IFS=',' read -r -a models <<<"${MODELS:-clip,qwen8,qwen32}"

prediction_root() {
    case "$1" in
        clip) echo artifacts/matrix_clip ;;
        qwen8) echo artifacts/matrix_qwen3_vl_8b ;;
        qwen32) echo artifacts/matrix_qwen3_vl_32b ;;
        *) echo "unknown VLM: $1" >&2; return 2 ;;
    esac
}

cd "$root"
mkdir -p "$artifact_root/caches"

for shift in "${shifts[@]}"; do
    manifest_root="artifacts/matrix_manifests/$shift"
    test -s "$manifest_root/source_val_manifest.jsonl"
    test -s "$manifest_root/target_train_manifest.jsonl"
    mkdir -p "$artifact_root/caches/$shift"

    for model in "${models[@]}"; do
        predictions="$(prediction_root "$model")/$shift"
        test -s "$predictions/source_predictions.jsonl" || {
            echo "missing source predictions: $predictions/source_predictions.jsonl" >&2
            exit 3
        }
        test -s "$predictions/target_predictions.jsonl" || {
            echo "missing target predictions: $predictions/target_predictions.jsonl" >&2
            exit 3
        }

        gate="$predictions/source_gate.json"
        if [[ "$model" == clip ]]; then
            gate="$artifact_root/caches/$shift/clip_source_gate.json"
            PYTHONPATH=. "$python" tools/score_vlm_target_adapter.py \
                --manifest "$manifest_root/source_val_manifest.jsonl" \
                --predictions "$predictions/source_predictions.jsonl" \
                --output "$gate"
        fi
        test -s "$gate"

        temperature="$artifact_root/caches/$shift/${model}_source_temperature.json"
        PYTHONPATH=. "$python" tools/calibrate_vlm_temperature.py \
            --source-predictions "$predictions/source_predictions.jsonl" \
            --output "$temperature"

        gate_override=()
        if ! "$python" -c \
            "import json; assert json.load(open('$gate'))['adapter_pass']"; then
            gate_override=(--allow-gate-failure)
        fi

        PYTHONPATH=. "$python" tools/build_vlm_target_adapter_cache.py \
            --manifest "$manifest_root/target_train_manifest.jsonl" \
            --predictions "$predictions/target_predictions.jsonl" \
            --source-gate-report "$gate" \
            --temperature-calibration-report "$temperature" \
            --output "$artifact_root/caches/$shift/${model}_target_adapter.json" \
            "${gate_override[@]}"

        PYTHONPATH=. "$python" tools/build_proposal_semantic_cache.py \
            --manifest "$manifest_root/target_train_manifest.jsonl" \
            --predictions "$predictions/target_predictions.jsonl" \
            --temperature-calibration-report "$temperature" \
            --output "$artifact_root/caches/$shift/${model}_proposal_semantic.json"
    done
done
