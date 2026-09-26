#!/usr/bin/env python3
"""获取独立冻结的新标注模型；权重 LFS SHA 必须核验。"""
import hashlib
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ENDPOINT = "https://hf-mirror.com"
MODELS = [
    ("Qwen/Qwen3-VL-32B-Instruct", "0cfaf48183f594c314753d30a4c4974bc75f3ccb"),
    ("OpenGVLab/InternVL3_5-8B-HF", "741a7d03020411e666c6109218ab71e08151ef86"),
]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-verified", action="store_true", help="保留已核验且大小未变的文件，只补缺失的模板等文件")
    args = parser.parse_args()
    jobs, manifests = [], {}
    for repo, revision in MODELS:
        assert len(revision) == 40
        folder = Path("models") / repo.split("/")[1]
        folder.mkdir(exist_ok=True)
        response = requests.get(f"{ENDPOINT}/api/models/{repo}/revision/{revision}", params={"blobs": "true"}, timeout=45)
        response.raise_for_status(); info = response.json()
        assert info["sha"] == revision
        entries = [entry for entry in info["siblings"] if entry["rfilename"].endswith((".json", ".safetensors", ".model", ".jinja", ".tiktoken")) or entry["rfilename"] == "merges.txt"]
        assert any(entry["rfilename"].endswith(".safetensors") for entry in entries)
        previous = json.loads((folder / "source_manifest.json").read_text()) if args.resume_verified and (folder / "source_manifest.json").exists() else None
        verified = []
        if previous:
            assert previous["repo"] == repo and previous["revision"] == revision
            by_name = {e["rfilename"]: e for e in entries}
            for entry in previous["verified_files"]:
                expected = by_name[entry["name"]].get("lfs", {}).get("sha256")
                assert not expected or expected == entry["sha256"]
                if (folder / entry["name"]).exists() and (folder / entry["name"]).stat().st_size == entry["bytes"]:
                    verified.append(entry)
        manifest = {"repo": repo, "revision": revision, "complete": False, "files": entries, "verified_files": verified}
        manifests[repo] = (folder, manifest)
        (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verified_names = {e["name"] for e in verified}
        jobs.extend((repo, revision, folder, entry) for entry in entries if entry["rfilename"] not in verified_names)
        print(json.dumps({"repo": repo, "revision": revision, "files": len(entries), "weight_bytes": sum(e.get("lfs", {}).get("size", 0) for e in entries)}), flush=True)
    def fetch(job):
        repo, revision, folder, entry = job
        name = entry["rfilename"]
        path = hf_hub_download(repo, name, revision=revision, endpoint=ENDPOINT, local_dir=str(folder), cache_dir="models/_hf_cache", token=False)
        actual = sha(path)
        expected = entry.get("lfs", {}).get("sha256")
        if name.endswith(".safetensors"):
            assert expected, "权重必须有上游 SHA"
        assert not expected or actual == expected
        assert not entry.get("size") or Path(path).stat().st_size == entry["size"]
        return repo, {"name": name, "sha256": actual, "bytes": Path(path).stat().st_size}
    # 先获取较小标注器，让其有界核验与大模型下载并行。
    jobs.sort(key=lambda job: (job[3]["rfilename"].endswith(".safetensors"), job[0] != MODELS[1][0]))
    with ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(fetch, job) for job in jobs]):
            repo, entry = future.result()
            folder, manifest = manifests[repo]
            manifest["verified_files"].append(entry)
            (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"repo": repo, "verified": entry["name"], "bytes": entry["bytes"], "complete_files": len(manifest["verified_files"])}), flush=True)
    for folder, manifest in manifests.values():
        assert len(manifest["verified_files"]) == len(manifest["files"])
        manifest["complete"] = True
        (folder / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
