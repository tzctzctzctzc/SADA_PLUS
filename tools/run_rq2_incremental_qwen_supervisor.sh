#!/usr/bin/env bash
set -euo pipefail

root=${RQ2_ROOT:-/vepfs/vlms/SADA_PLUS}
artifact=${RQ2_ARTIFACT:-$root/artifacts/rq2_probe_20260919}
python=${RQ2_PYTHON:-/vepfs/vlms/envs/rq1-cu121/bin/python}
frame_cache=${FRAME_CACHE:-/vepfs/vlms/cache/rq2/frames}
namespace=rq2_probe_counterfactual_2fps_dedup_v1
manifest=$artifact/rq2_probe_manifest.jsonl
feature_root=$artifact/features/qwen3_vl_8b_full
minimum_free_mb=${RQ2_MINIMUM_FREE_MB:-30000}

cd "$root"
mkdir -p "$artifact/logs" "$feature_root"
export PYTHONPATH=$root
export HF_HOME=${HF_HOME:-/vepfs/vlms/cache/huggingface}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

prep_pid=""
find_prep() {
    prep_pid=$(pgrep -f "prepare_rq2_probe_frames.py.*--cache-namespace $namespace" | head -n 1 || true)
}

gpu_ready() {
    local gpu=$1 values used free util
    values=$(nvidia-smi -i "$gpu" --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
    IFS=',' read -r used free util <<<"$values"
    free=${free// /}
    util=${util// /}
    (( free >= minimum_free_mb && util <= 20 ))
}

launch_shard() {
    local gpu=$1 shard=$2 index=$artifact/qwen3_vl_8b_full_shard${shard}.jsonl
    local log=$artifact/logs/full_qwen3_vl_8b_shard${shard}_incremental.log
    if pgrep -af "extract_rq2_qwen_frame_features.py.*--shard-index $shard.*--available-only" >/dev/null; then
        return 0
    fi
    if ! gpu_ready "$gpu"; then
        return 0
    fi
    echo "$(date --iso-8601=seconds) launch gpu=$gpu shard=$shard"
    nohup env CUDA_VISIBLE_DEVICES="$gpu" "$python" tools/extract_rq2_qwen_frame_features.py \
        --manifest "$manifest" --frame-cache "$frame_cache" \
        --cache-namespace "$namespace" --output-index "$index" \
        --feature-root "$feature_root" --model-path Qwen/Qwen3-VL-8B-Instruct \
        --offline --min-pixels $((224 * 224)) --max-pixels $((224 * 224)) \
        --max-batch-frames 256 --loader-workers 8 \
        --shard-count 2 --shard-index "$shard" --available-only \
        >>"$log" 2>&1 < /dev/null &
}

find_prep
while [[ -n "$prep_pid" ]] && kill -0 "$prep_pid" 2>/dev/null; do
    launch_shard 0 0
    launch_shard 3 1
    sleep 60
    find_prep
done

# The main full-feature launcher takes over after frame preparation.
echo "$(date --iso-8601=seconds) frame preparation ended; incremental supervisor exiting"
