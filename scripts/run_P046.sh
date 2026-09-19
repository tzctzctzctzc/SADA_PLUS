#!/usr/bin/env bash
set -euo pipefail

cd /vepfs/vlms/1
export CUDA_VISIBLE_DEVICES=GPU-6fc32588-9464-5f67-98bc-d436aec84f4a
export PYTHONUNBUFFERED=1

python scripts/audit_P046_implementation.py \
  2>&1 | tee logs/P046_static_audit.log

envs/qwen3/bin/python src/run_P046.py --audit-only --arm role_corruption \
  2>&1 | tee logs/P046_runtime_audit.log

envs/qwen3/bin/python scripts/audit_P046_corruptions.py \
  2>&1 | tee logs/P046_corruption_audit.log

for arm in ordinary_retrieval generic_augmentation role_corruption; do
  envs/qwen3/bin/python src/run_P046.py --arm "$arm" \
    2>&1 | tee "logs/P046_${arm}.log"
done
