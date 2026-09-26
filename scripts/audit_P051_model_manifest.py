#!/usr/bin/env python3
"""核验主编码器全部权重和运行配置的固定 revision 与上游内容哈希。"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    config = json.loads(Path("configs/P051_tracebind.json").read_text())
    folder = Path(config["model"]["path"])
    revision = config["model"]["revision"]
    weights = set(json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"].values())
    files = weights | {p.name for p in folder.iterdir() if p.is_file() and p.name != "source_manifest.json" and p.suffix in [".json", ".jinja", ".model", ".tiktoken"]}
    files |= {"merges.txt", "scripts/qwen3_vl_embedding.py"}
    def check(name):
        path = folder / name
        metadata = (folder / ".cache/huggingface/download" / (name + ".metadata")).read_text().splitlines()
        assert metadata[0] == revision, "revision 不一致: " + name
        expected = metadata[1]
        assert len(expected) in [40, 64]
        digest = hashlib.sha256() if len(expected) == 64 else hashlib.sha1()
        if len(expected) == 40:
            digest.update(f"blob {path.stat().st_size}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1048576), b""):
                digest.update(chunk)
        assert digest.hexdigest() == expected, "上游内容哈希不一致: " + name
        return {"name": name, "bytes": path.stat().st_size, "upstream_hash": expected,
                "hash_type": "sha256_lfs" if len(expected) == 64 else "sha1_git_blob"}
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(check, sorted(files)))
    report = {"status": "pass", "revision": revision, "weight_files_verified": len(weights),
              "runtime_files_verified": len(results), "verified_files": results,
              "scope": "main embedding encoder weights, tokenizer/processor configs and official embedding source; not model efficacy"}
    Path("runs/P051_tracebind/embedding_model_integrity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
