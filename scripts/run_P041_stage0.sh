#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/vepfs/vlms/1
cd "$PROJECT_ROOT"
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
exec envs/main/bin/python scripts/prepare_P041.py --config configs/P041_selective_dependence_pilot.json --stage all

