"""Smoke-test and launch balanced RQ2 dense VLM extraction workers."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tools.rq2_dense_protocol import (
    MODEL_SPECS,
    read_protocol,
    validate_feature_file,
)


MODEL_QUEUE = ("qwen3_vl_8b", "timelens2_8b", "timeple_8b")
MODEL_DEFAULTS = {
    "qwen3_vl_8b": {
        "cache": "models--Qwen--Qwen3-VL-8B-Instruct",
        "repository": "/vepfs/vlms/repos/Qwen3-VL",
        "workers": 3,
        "batch_tokens": 1,
    },
    "timelens2_8b": {
        "cache": "models--MCG-NJU--TimeLens2-8B",
        "repository": "/vepfs/vlms/repos/TimeLens2",
        "workers": 3,
        "batch_tokens": 1,
    },
    "timeple_8b": {
        "cache": "models--KlingTeam--TimePLE",
        "repository": "/vepfs/vlms/repos/TimePLE",
        "timeple_root": "/vepfs/vlms/repos/TimePLE",
        "workers": 2,
        "batch_tokens": 1,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("auto",) + MODEL_QUEUE, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--python", default="/vepfs/vlms/envs/rq1-cu121/bin/python")
    parser.add_argument("--repository-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--video-root", default="/vepfs/vlms/data/EPIC_KITCHENS/videos")
    parser.add_argument(
        "--frame-recovery-root",
        default="/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery",
    )
    parser.add_argument(
        "--reference-feature-root",
        default="/vepfs/vlms/SADA_PLUS/data/ek100_slowfast/features",
    )
    parser.add_argument(
        "--shift-registry",
        default="/vepfs/vlms/SADA_PLUS/configs/rq2/shift_registry.yaml",
    )
    parser.add_argument(
        "--hf-cache-root", default="/vepfs/vlms/cache/huggingface/hub"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-tokens", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke-only", action="store_true")
    mode.add_argument("--launch", action="store_true")
    return parser.parse_args()


def resolve_snapshot(cache_root, cache_name):
    root = Path(cache_root) / cache_name
    reference = root / "refs" / "main"
    if not reference.is_file():
        raise FileNotFoundError("model cache is incomplete: {}".format(root))
    snapshot = root / "snapshots" / reference.read_text(encoding="ascii").strip()
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError("model snapshot is incomplete: {}".format(snapshot))
    return snapshot


def model_arguments(args, model_name):
    defaults = MODEL_DEFAULTS[model_name]
    snapshot = resolve_snapshot(args.hf_cache_root, defaults["cache"])
    repository = Path(defaults["repository"])
    if not (repository / ".git").is_dir():
        raise FileNotFoundError("official model repository is missing: {}".format(repository))
    result = [
        "--model", model_name,
        "--model-path", str(snapshot),
        "--model-repository", str(repository),
    ]
    if defaults.get("timeple_root"):
        result += ["--timeple-root", defaults["timeple_root"]]
    return result


def common_command(args, model_name, output_root, batch_tokens):
    return [
        args.python,
        "-m",
        "tools.extract_rq2_dense_vlm_features",
        *model_arguments(args, model_name),
        "--video-root", args.video_root,
        "--frame-recovery-root", args.frame_recovery_root,
        "--reference-feature-root", args.reference_feature_root,
        "--output-root", str(output_root),
        "--shift-registry", args.shift_registry,
        "--device", "cuda:0",
        "--batch-tokens", str(batch_tokens),
        "--offline",
        "--device-map", "none",
        "--attn-implementation", "sdpa",
        "--dtype", "bfloat16",
    ]


def gpu_is_available(gpu):
    output = subprocess.check_output(
        [
            "nvidia-smi", "-i", str(gpu),
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    free_mb, utilization = [int(value.strip()) for value in output.split(",")]
    return free_mb >= 30000 and utilization <= 20, free_mb, utilization


def extraction_environment(gpu):
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    })
    return environment


def run_smoke(args, model_name, batch_tokens):
    available, free_mb, utilization = gpu_is_available(args.gpu)
    if not available:
        raise RuntimeError(
            "GPU {} is busy: free_mb={} utilization={}".format(
                args.gpu, free_mb, utilization
            )
        )
    smoke_root = Path(args.artifact_root) / "smoke" / model_name
    command = common_command(args, model_name, smoke_root, batch_tokens)
    command += ["--video-id", "P02_05"]
    subprocess.run(
        command,
        cwd=args.repository_root,
        env=extraction_environment(args.gpu),
        check=True,
    )
    protocol = read_protocol(smoke_root / "protocol.json")
    with np.load(Path(args.reference_feature_root) / "P02_05.npz") as payload:
        expected_length = int(payload["feats"].shape[0])
    if not validate_feature_file(
        smoke_root / "P02_05.npz",
        model_name,
        expected_length,
        protocol["protocol_sha256"],
    ):
        raise RuntimeError("smoke feature validation failed")
    print(
        "smoke passed model={} shape=({}, {}) protocol={}".format(
            model_name,
            expected_length,
            MODEL_SPECS[model_name]["output_dim"],
            protocol["protocol_sha256"],
        ),
        flush=True,
    )


def balanced_assignments(video_ids, reference_root, workers):
    weighted = []
    for video_id in video_ids:
        with np.load(reference_root / (video_id + ".npz")) as payload:
            weighted.append((int(payload["feats"].shape[0]), video_id))
    groups = [[] for _ in range(workers)]
    totals = [0] * workers
    for tokens, video_id in sorted(weighted, reverse=True):
        worker = min(range(workers), key=totals.__getitem__)
        groups[worker].append(video_id)
        totals[worker] += tokens
    return groups, totals


def select_auto_model(args):
    for model_name in MODEL_QUEUE:
        output_root = Path(args.artifact_root) / model_name
        if output_appears_complete(output_root) or output_has_live_workers(output_root):
            continue
        try:
            model_arguments(args, model_name)
        except FileNotFoundError:
            continue
        if MODEL_SPECS[model_name]["formal_eligible"]:
            return model_name
    raise RuntimeError("no prepared formal-eligible model remains in the queue")


def output_appears_complete(output_root):
    video_index = output_root / "video_ids.txt"
    if not video_index.is_file():
        return False
    video_ids = video_index.read_text(encoding="ascii").splitlines()
    if not video_ids or video_ids != sorted(set(video_ids)):
        return False
    completed = {path.stem for path in output_root.glob("*.npz")}
    return completed == set(video_ids)


def output_has_live_workers(output_root):
    state_path = output_root / "launch_state.json"
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        pids = [int(worker["pid"]) for worker in state["workers"]]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    for pid in pids:
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            continue
        return True
    return False


def launch_workers(args, model_name, workers, batch_tokens):
    if not MODEL_SPECS[model_name]["formal_eligible"]:
        raise RuntimeError(
            "{} has not passed the formal interface gate".format(model_name)
        )
    available, free_mb, utilization = gpu_is_available(args.gpu)
    if not available:
        raise RuntimeError(
            "GPU {} is busy: free_mb={} utilization={}".format(
                args.gpu, free_mb, utilization
            )
        )
    output_root = Path(args.artifact_root) / model_name
    if output_has_live_workers(output_root):
        raise RuntimeError("{} already has live extraction workers".format(model_name))
    protocol_command = common_command(args, model_name, output_root, batch_tokens)
    subprocess.run(
        protocol_command + ["--protocol-only"],
        cwd=args.repository_root,
        env=extraction_environment(args.gpu),
        check=True,
    )
    video_ids = (output_root / "video_ids.txt").read_text(
        encoding="ascii"
    ).splitlines()
    groups, totals = balanced_assignments(
        video_ids, Path(args.reference_feature_root), workers
    )
    assignment_root = output_root / "assignments"
    log_root = output_root / "logs"
    assignment_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, group in enumerate(groups):
        assignment = assignment_root / "worker_{}.txt".format(index)
        assignment.write_text(
            "".join(video_id + "\n" for video_id in group), encoding="ascii"
        )
        command = common_command(args, model_name, output_root, batch_tokens)
        command += ["--assignment-file", str(assignment)]
        log_path = log_root / "worker_{}.log".format(index)
        log_stream = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.repository_root,
            env=extraction_environment(args.gpu),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append({
            "worker": index,
            "pid": process.pid,
            "tokens": totals[index],
            "assignment": str(assignment),
            "log": str(log_path),
        })
    state = {
        "schema": "rq2-dense-vlm-launch-v1",
        "model": model_name,
        "gpu": args.gpu,
        "workers": processes,
        "protocol": str(output_root / "protocol.json"),
    }
    state_path = output_root / "launch_state.json"
    temporary = output_root / ".launch_state.{}.tmp".format(os.getpid())
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, state_path)
    print(json.dumps(state, sort_keys=True), flush=True)


def main():
    args = parse_args()
    model_name = select_auto_model(args) if args.model == "auto" else args.model
    defaults = MODEL_DEFAULTS[model_name]
    workers = args.workers or defaults["workers"]
    batch_tokens = args.batch_tokens or defaults["batch_tokens"]
    if workers <= 0 or batch_tokens <= 0:
        raise ValueError("workers and batch tokens must be positive")
    if args.smoke_only:
        run_smoke(args, model_name, batch_tokens)
    else:
        launch_workers(args, model_name, workers, batch_tokens)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
