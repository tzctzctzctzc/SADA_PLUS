#!/usr/bin/env python3
"""Load the official Qwen3-VL-Embedding-8B path on the mandated physical GPU1."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import torch


ROOT = Path("/vepfs/vlms/1")
EXPECTED_GPU = "GPU-277ce254-c418-8feb-d0eb-e3929d6a6f9f"
assert Path.cwd().resolve() == ROOT
assert os.environ.get("CUDA_VISIBLE_DEVICES") == EXPECTED_GPU
assert torch.cuda.device_count() == 1
assert "A100" in torch.cuda.get_device_name(0)

model_path = ROOT / "models/Qwen3-VL-Embedding-8B"
source_path = model_path / "scripts/qwen3_vl_embedding.py"
config = json.loads((ROOT / "configs/P041_selective_dependence_pilot.json").read_text())
assert model_path.exists() and source_path.exists()
assert config["model"]["path"] == str(model_path.relative_to(ROOT))

spec = importlib.util.spec_from_file_location("qwen3_vl_embedding_official", source_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

torch.manual_seed(20260917)
torch.backends.cuda.matmul.allow_tf32 = False
embedder = module.Qwen3VLEmbedder(
    str(model_path),
    max_length=512,
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    local_files_only=True,
)
embeddings = embedder.process(
    [
        {"text": "a red circular object", "instruction": "Represent the candidate image."},
        {"text": "a blue square object", "instruction": "Represent the candidate image."},
    ]
)
norms = embeddings.float().norm(dim=-1)
parameter_device = str(next(embedder.model.parameters()).device)
assert parameter_device == "cuda:0"
assert embeddings.device.type == "cuda"
assert embeddings.shape[0] == 2
assert embeddings.shape[1] == 4096
assert torch.isfinite(embeddings).all()
assert torch.allclose(norms, torch.ones_like(norms), atol=1e-2, rtol=0)

summary = {
    "status": "pass",
    "gpu_uuid": EXPECTED_GPU,
    "gpu_name": torch.cuda.get_device_name(0),
    "model_revision": config["model"]["revision"],
    "official_source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "parameter_device": parameter_device,
    "embedding_device": str(embeddings.device),
    "embedding_shape": list(embeddings.shape),
    "embedding_norms": norms.tolist(),
    "pair_cosine": float((embeddings[0].float() * embeddings[1].float()).sum()),
    "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
    "torch": torch.__version__,
}
out = ROOT / "runs/P041_selective_dependence/model_smoke.json"
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))

