#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
PYTHON=/vepfs/vlms/1/envs/qwen3/bin/python
DATA_PYTHON=/vepfs/vlms/1/envs/main/bin/python
GPU_UUID=GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f

"$PYTHON" scripts/prepare_P049.py
"$PYTHON" scripts/audit_P049_implementation.py
"$DATA_PYTHON" scripts/extract_P049_images.py
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" src/run_P049.py --audit-only
CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" src/run_P049.py --prepare-base
for arm in ordinary_verifier positive_counterfactual dod_counterfactual; do
  CUDA_VISIBLE_DEVICES="$GPU_UUID" "$PYTHON" src/run_P049.py --arm "$arm"
done
"$PYTHON" scripts/analyze_P049.py
