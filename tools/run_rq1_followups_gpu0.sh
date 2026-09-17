#!/usr/bin/env bash
set -euo pipefail

root=${RQ1_ROOT:-/vepfs/vlms/SADA_PLUS}
python=${RQ1_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
base_artifact=${RQ1_BASE_ARTIFACT:-$root/artifacts/rq1_semantic_20260915}
probe_artifact=${RQ1_PROBE_ARTIFACT:-$base_artifact/source_probe_8f}
egovideo_artifact=${RQ1_EGOVIDEO_ARTIFACT:-$base_artifact/egovideo_l_8f}
artifact=${RQ1_FOLLOWUP_ARTIFACT:-$base_artifact/rq1_followups_8f}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq1/frames}
egohod_root=${EGOHOD_ROOT:-/vepfs/vlms/external/EgoHOD}
checkpoint=${EGOVIDEO_CHECKPOINT:-/vepfs/vlms/cache/huggingface/models--Jazzcharles--EgoVideo/snapshots/3c1781d01b74e3cbacd3129eda7025b9ee1e3fa9/large_best.pt}
gpu0=${GPU0:-0}

mkdir -p "$artifact/logs" "$artifact/features"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export TOKENIZERS_PARALLELISM=false

manifest=$probe_artifact/rq1_source_probe_segments.jsonl
eval_manifest=$base_artifact/rq1_val_gt_segments.jsonl
egovideo_prefix=$egovideo_artifact/features/egovideo_l_joint_8f

# This CPU process uses existing features and never reads target labels for selection.
"$python" tools/run_rq1_tuned_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$egovideo_prefix" \
    --model-name EgoVideo-HOD-source-linear-tuned \
    --output "$artifact/egovideo_source_linear_tuned.jsonl" \
    --alphas 1e-6,1e-5,1e-4,1e-3,1e-2 --seeds 0,1,2,3,4 \
    --write-seed-predictions \
    >"$artifact/logs/egovideo_tuned_probe.log" 2>&1 &
probe_pid=$!

# GPU0 only: encode two official-style text banks, then the matched CLIP-L control.
CUDA_VISIBLE_DEVICES=$gpu0 "$python" tools/extract_rq1_egovideo_text_features.py \
    --checkpoint "$checkpoint" --egohod-root "$egohod_root" \
    --prompt-output "bare_label=$artifact/features/egovideo_bare_label.npy" \
    --prompt-output "bare_phrase=$artifact/features/egovideo_bare_phrase.npy" \
    >"$artifact/logs/egovideo_text_features.log" 2>&1

for preset in bare_label bare_phrase; do
    "$python" tools/run_rq1_embedding_zero_shot.py \
        --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
        --feature-prefix "$egovideo_prefix" \
        --text-features "$artifact/features/egovideo_${preset}.npy" \
        --model-name "EgoVideo-HOD-zero-shot-${preset}" \
        --output "$artifact/egovideo_zero_shot_${preset}.jsonl" \
        >"$artifact/logs/egovideo_zero_shot_${preset}.log" 2>&1
done

clip_prefix=$artifact/features/clip_vit_l14_336_static
CUDA_VISIBLE_DEVICES=$gpu0 "$python" tools/extract_rq1_clip_features.py \
    --manifest "$manifest" --frame-cache "$frame_cache" \
    --cache-namespace rq1_8f --frame-count 8 \
    --output-prefix "$clip_prefix" --model 'ViT-L/14@336px' \
    --model-cache /vepfs/vlms/cache/clip --pooling static \
    --segment-batch-size 32 --image-batch-size 256 --workers 8 \
    --text-features-output "$artifact/features/clip_vit_l14_336_text.npy" \
    >"$artifact/logs/clip_l14_336_features.log" 2>&1

"$python" tools/run_rq1_embedding_zero_shot.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$clip_prefix" \
    --text-features "$artifact/features/clip_vit_l14_336_text.npy" \
    --model-name CLIP-ViT-L14-336-zero-shot \
    --output "$artifact/clip_vit_l14_336_zero_shot.jsonl" \
    >"$artifact/logs/clip_l14_336_zero_shot.log" 2>&1

"$python" tools/run_rq1_tuned_feature_probe.py \
    --feature-manifest "$manifest" --eval-manifest "$eval_manifest" \
    --feature-prefix "$clip_prefix" \
    --model-name CLIP-ViT-L14-336-source-linear-tuned \
    --output "$artifact/clip_vit_l14_336_source_linear_tuned.jsonl" \
    --alphas 1e-6,1e-5,1e-4,1e-3,1e-2 --seeds 0,1,2,3,4 \
    --write-seed-predictions \
    >"$artifact/logs/clip_l14_336_tuned_probe.log" 2>&1

wait "$probe_pid"

"$python" tools/evaluate_rq1_semantics.py \
    --manifest "$eval_manifest" \
    --prediction "EgoVideo-shared-prompt=$egovideo_artifact/egovideo_l_zero_shot.jsonl" \
    --prediction "EgoVideo-bare-label=$artifact/egovideo_zero_shot_bare_label.jsonl" \
    --prediction "EgoVideo-bare-phrase=$artifact/egovideo_zero_shot_bare_phrase.jsonl" \
    --prediction "EgoVideo-source-linear-tuned=$artifact/egovideo_source_linear_tuned.jsonl" \
    --prediction "CLIP-L14-336-zero-shot=$artifact/clip_vit_l14_336_zero_shot.jsonl" \
    --prediction "CLIP-L14-336-source-linear-tuned=$artifact/clip_vit_l14_336_source_linear_tuned.jsonl" \
    --output-json "$artifact/rq1_followup_metrics.json" \
    --output-csv "$artifact/rq1_followup_metrics.csv" \
    >"$artifact/logs/evaluate.log" 2>&1

touch "$artifact/.complete"
echo "$(date --iso-8601=seconds) rq1_followups_complete"
