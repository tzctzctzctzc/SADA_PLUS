#!/usr/bin/env bash
set -euo pipefail

root=/home1/gyy/vlms/SADA_CLIP_BENCH_20260904_r1
python=/home1/gyy/vlms/SADA_STAGE2_20260727/env/sada/bin/python
manifest=artifacts/matrix_manifests/S5/target_train_manifest.jsonl
output_root=artifacts/s5_strategy_matrix/caches

cd "$root"
mkdir -p "$output_root"

for model in clip qwen8 qwen32; do
    case "$model" in
        clip)
            prediction_root=artifacts/matrix_clip/S5
            gate="$output_root/clip_source_gate.json"
            PYTHONPATH=. "$python" tools/score_vlm_target_adapter.py \
                --manifest artifacts/matrix_manifests/S5/source_val_manifest.jsonl \
                --predictions "$prediction_root/source_predictions.jsonl" \
                --output "$gate"
            ;;
        qwen8)
            prediction_root=artifacts/matrix_qwen3_vl_8b/S5
            gate="$prediction_root/source_gate.json"
            ;;
        qwen32)
            prediction_root=artifacts/matrix_qwen3_vl_32b/S5
            gate="$prediction_root/source_gate.json"
            ;;
    esac

    temperature="$output_root/${model}_source_temperature.json"
    PYTHONPATH=. "$python" tools/calibrate_vlm_temperature.py \
        --source-predictions "$prediction_root/source_predictions.jsonl" \
        --output "$temperature"

    gate_override=()
    if ! "$python" -c \
        "import json; assert json.load(open('$gate'))['adapter_pass']"; then
        gate_override=(--allow-gate-failure)
    fi
    PYTHONPATH=. "$python" tools/build_vlm_target_adapter_cache.py \
        --manifest "$manifest" \
        --predictions "$prediction_root/target_predictions.jsonl" \
        --source-gate-report "$gate" \
        --temperature-calibration-report "$temperature" \
        --output "$output_root/${model}_target_adapter.json" \
        "${gate_override[@]}"

    PYTHONPATH=. "$python" tools/build_proposal_semantic_cache.py \
        --manifest "$manifest" \
        --predictions "$prediction_root/target_predictions.jsonl" \
        --temperature-calibration-report "$temperature" \
        --output "$output_root/${model}_proposal_semantic.json"
done

