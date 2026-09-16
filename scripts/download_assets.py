import concurrent.futures
import json
import time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root = Path("/vepfs/vlms/1")
jobs = [
    ("model", "Qwen/Qwen2.5-VL-7B-Instruct", root / "models/Qwen2.5-VL-7B-Instruct"),
    ("dataset", "TIGER-Lab/Mantis-Eval", root / "data/Mantis-Eval"),
]
def fetch(job):
    kind, repo, dest = job
    api = HfApi()
    info = api.repo_info(repo, repo_type=kind, files_metadata=True)
    manifest = {"repository": repo, "revision": info.sha, "type": kind,
                "files": [{"path": x.rfilename, "size": x.size,
                           "lfs": x.lfs} for x in info.siblings]}
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "source_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"START {repo} revision={info.sha}", flush=True)
    for attempt in range(5):
        try:
            snapshot_download(repo_id=repo, repo_type=kind, revision=info.sha,
                              local_dir=dest, max_workers=4,
                              allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.parquet", "*.jpg", "*.png"],
                              token=False)
            break
        except Exception as error:
            print(f"RETRY {repo} attempt={attempt + 1}: {type(error).__name__}", flush=True)
            if attempt == 4:
                raise
            time.sleep(10)
    print(f"DONE {repo}", flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(fetch, jobs))
(root / "runs/download_assets.done").write_text(str(time.time()))
