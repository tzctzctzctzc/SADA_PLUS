#!/usr/bin/env bash
set -euo pipefail

root=${RQ2_ROOT:-/vepfs/vlms/SADA_PLUS}
python=${RQ2_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
sada_python=${SADA_PYTHON:-/vepfs/vlms/envs/sada/bin/python}
artifact=${RQ2_ARTIFACT:-$root/artifacts/rq2_boundary_20260915}
rq1_artifact=${RQ1_ARTIFACT:-$root/artifacts/rq1_semantic_20260915}
video_root=${VIDEO_ROOT:-/vepfs/vlms/data/EPIC_KITCHENS/videos}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq2/frames}
gpu0=${GPU0:-0}
gpu1=${GPU1:-1}
manifest=$artifact/rq2_boundary_manifest.jsonl
namespace=rq2_boundary_16f

mkdir -p "$artifact/logs"
cd "$root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

while [[ ! -f "$rq1_artifact/.complete_16f" ]]; do
    echo "$(date --iso-8601=seconds) waiting_rq1_16f"
    sleep 60
done

echo "$(date --iso-8601=seconds) compare_rq1_8f_16f"
"$python" tools/compare_rq1_frame_counts.py \
    --manifest "$rq1_artifact/rq1_val_gt_segments.jsonl" \
    --pair "CLIP-ViT-B16=$rq1_artifact/clip_vit_b16.jsonl=$rq1_artifact/clip_vit_b16_16f.jsonl" \
    --pair "Qwen3-VL-8B=$rq1_artifact/qwen3_vl_8b.jsonl=$rq1_artifact/qwen3_vl_8b_16f.jsonl" \
    --pair "Qwen3-VL-32B=$rq1_artifact/qwen3_vl_32b.jsonl=$rq1_artifact/qwen3_vl_32b_16f.jsonl" \
    --output-json "$rq1_artifact/rq1_frame_density_8f_vs_16f.json" \
    --output-csv "$rq1_artifact/rq1_frame_density_8f_vs_16f.csv"

echo "$(date --iso-8601=seconds) build_rq2_manifest"
"$python" tools/build_rq2_boundary_manifest.py \
    --repo-root "$root" --output "$manifest" --frame-count 16
echo "$(date --iso-8601=seconds) prepare_rq2_frames"
"$sada_python" tools/prepare_rq1_frames.py \
    --manifest "$manifest" --video-root "$video_root" \
    --frame-cache "$frame_cache" --cache-namespace "$namespace" \
    --frame-count 16 --workers 8

echo "$(date --iso-8601=seconds) rq2_slowfast"
"$python" tools/run_rq2_slowfast.py \
    --repo-root "$root" --manifest "$manifest" \
    --output "$artifact/slowfast_framewise.jsonl" \
    >"$artifact/logs/slowfast.log" 2>&1

run_qwen_pair() {
    local family=$1 model_path=$2 tag=$3
    local pids=()
    local index gpu
    for index in 0 1; do
        gpu=$gpu0
        [[ "$index" == 1 ]] && gpu=$gpu1
        CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq2_qwen.py \
            --manifest "$manifest" --frame-cache "$frame_cache" \
            --cache-namespace "$namespace" \
            --output "$artifact/${tag}_shard_${index}.jsonl" \
            --model-family "$family" --model-path "$model_path" --offline \
            --frame-count 16 --shard-count 2 --shard-index "$index" \
            >"$artifact/logs/${tag}_${index}.log" 2>&1 &
        pids+=("$!")
    done
    local pid
    for pid in "${pids[@]}"; do wait "$pid"; done
    "$python" tools/merge_rq2_predictions.py --manifest "$manifest" \
        --input "$artifact/${tag}_shard_0.jsonl" \
        --input "$artifact/${tag}_shard_1.jsonl" \
        --output "$artifact/${tag}.jsonl"
}

echo "$(date --iso-8601=seconds) rq2_qwen32"
run_qwen_pair qwen3-32b Qwen/Qwen3-VL-32B-Instruct qwen3_vl_32b
echo "$(date --iso-8601=seconds) rq2_qwen8"
run_qwen_pair qwen3-8b Qwen/Qwen3-VL-8B-Instruct qwen3_vl_8b

echo "$(date --iso-8601=seconds) rq2_clip"
pids=()
for index in 0 1; do
    gpu=$gpu0
    [[ "$index" == 1 ]] && gpu=$gpu1
    CUDA_VISIBLE_DEVICES=$gpu "$python" tools/run_rq2_clip.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace "$namespace" \
        --output "$artifact/clip_shard_${index}.jsonl" \
        --model ViT-B/16 --model-cache /vepfs/vlms/cache/clip \
        --frame-count 16 --segment-batch-size 16 --image-batch-size 128 \
        --shard-count 2 --shard-index "$index" \
        >"$artifact/logs/clip_${index}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$python" tools/merge_rq2_predictions.py --manifest "$manifest" \
    --input "$artifact/clip_shard_0.jsonl" \
    --input "$artifact/clip_shard_1.jsonl" \
    --output "$artifact/clip.jsonl"

"$python" - "$manifest" "$artifact/qwen3_vl_32b.jsonl" \
    "$artifact/qwen3_vl_8b.jsonl" "$artifact/clip.jsonl" <<'PY'
import json
import sys

def ids(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line)["candidate_id"] for line in stream if line.strip()]

expected = ids(sys.argv[1])
for path in sys.argv[2:]:
    current = ids(path)
    assert len(current) == len(expected), (path, len(current), len(expected))
    assert set(current) == set(expected), (path, "candidate mismatch")
    assert len(current) == len(set(current)), (path, "duplicate candidate")
    print(path, len(current), "complete")
PY

"$python" tools/evaluate_rq2_boundaries.py \
    --manifest "$manifest" \
    --prediction "SlowFast-framewise=$artifact/slowfast_framewise.jsonl" \
    --prediction "CLIP-framewise=$artifact/clip.jsonl" \
    --prediction "Qwen3-VL-8B=$artifact/qwen3_vl_8b.jsonl" \
    --prediction "Qwen3-VL-32B=$artifact/qwen3_vl_32b.jsonl" \
    --output-json "$artifact/rq2_metrics.json" \
    --output-csv "$artifact/rq2_metrics.csv"
touch "$artifact/.complete"
echo "$(date --iso-8601=seconds) rq2_complete"
