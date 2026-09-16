#!/usr/bin/env bash
set -euo pipefail
source /vepfs/vlms/1/env.sh
envs/main/bin/uv pip install --python envs/main/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
envs/main/bin/uv pip install --python envs/main/bin/python --index-url https://pypi.org/simple transformers==4.51.3 accelerate==1.6.0 qwen-vl-utils==0.0.11 datasets==3.5.0 huggingface-hub==0.30.2 pillow scipy scikit-learn pandas pypdf einops sentencepiece protobuf
printf 'setup complete\n'
