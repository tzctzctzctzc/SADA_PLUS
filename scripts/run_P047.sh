#!/usr/bin/env bash
set -euo pipefail

cd /vepfs/vlms/1
PYTHON=/vepfs/vlms/1/envs/qwen3/bin/python
DATA_PYTHON=/vepfs/vlms/1/envs/main/bin/python
GPU_UUID=GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f

"$PYTHON" scripts/prepare_P047.py
"$PYTHON" scripts/audit_P047_implementation.py
"$DATA_PYTHON" scripts/extract_P047_images.py
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" scripts/audit_P047_corruptions.py
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" src/run_P047.py --audit-only
for arm in ordinary_retrieval paraphrase_invariance image_role_swap text_role_swap; do
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" src/run_P047.py --arm "$arm"
done
"$PYTHON" scripts/analyze_P047.py
