#!/usr/bin/env bash
set -euo pipefail
cd /vepfs/vlms/1
source env.sh
trap 'rc=$?; printf "%s\n" "$rc" > runs/qwen3_env_setup.exitcode' EXIT
test ! -e envs/qwen3
envs/main/bin/python -m venv envs/qwen3
envs/qwen3/bin/python -m pip install --disable-pip-version-check torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
envs/qwen3/bin/python -m pip install --disable-pip-version-check transformers==4.57.1 qwen-vl-utils==0.0.14 accelerate==1.6.0 scipy==1.15.3 pillow==11.3.0
envs/qwen3/bin/python -m pip check
envs/qwen3/bin/python -m pip freeze > runs/qwen3_environment_freeze.txt
envs/qwen3/bin/python -c 'import torch, transformers; from transformers import Qwen3VLForConditionalGeneration; print(torch.__version__, transformers.__version__)'
