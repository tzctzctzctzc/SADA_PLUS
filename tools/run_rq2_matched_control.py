"""One-card, fresh SO -> original SADA-UDA matched control (one shift/seed)."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import yaml

from tools.audit_rq2_vlm_benchmark import audit_so_uda_pair
from tools.rq2_dense_protocol import MODEL_SPECS, sha256_file, validate_training_protocol
from tools.rq2_evaluate import load_checkpoint, selected_checkpoint_state
from tools.rq2_formal import (
    RNG_PROTOCOL, SOURCE_OPTIMIZATION_PROTOCOL, TRAINING_BUDGET_PROTOCOL,
    check_registry_config, validate_formal_runtime, validate_formal_resume,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
MATCHED_MODELS = ("slowfast", "egovideo_hod", "internvideo2_clip_1b", "qwen3_vl_8b", "timelens2_8b")


def matched_feature_args(model, feature_root):
    """Only the frozen representation and its input dimension may change."""
    if model == "slowfast":
        if feature_root is not None:
            raise ValueError("SlowFast uses the frozen reference feature root")
        return []
    if feature_root is None:
        raise ValueError("VLM matched control requires --feature-root")
    root = Path(feature_root).resolve()
    protocol_path = root / "protocol.json"
    dim = MODEL_SPECS[model]["output_dim"]
    protocol = validate_training_protocol(protocol_path, root, dim)
    if protocol["model_name"] != model:
        raise ValueError("feature protocol belongs to another model")
    return ["--feature_folder", str(root), "--input_dim", str(dim),
            "--feature-protocol", str(protocol_path)]


def check_code(expected_head):
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE_ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=CODE_ROOT, text=True)
    if head != expected_head or status.strip():
        raise ValueError("matched control requires the audited clean code commit")


def idle_gpu_uuid(gpu):
    line = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip().split(",")
    uuid, memory, utilization = (part.strip() for part in line)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader",
    ], text=True)
    if int(memory) >= 2048 or int(utilization) > 5 or uuid in processes:
        raise RuntimeError("selected physical GPU is occupied; external jobs are not interrupted")
    return uuid


def check_new_protocol(cfg, allow_accelerated=False):
    formal = cfg["rq2_formal"]
    if (formal.get("training_budget") != TRAINING_BUDGET_PROTOCOL
            or formal.get("source_optimization") != SOURCE_OPTIMIZATION_PROTOCOL
            or formal.get("rng_protocol") != RNG_PROTOCOL
            or not formal.get("data_provenance")):
        raise ValueError("legacy SO/UDA results cannot enter the new matched control")
    from tools.rq3_accelerated import valid_batch
    batch_ok = valid_batch(cfg) if allow_accelerated else int(cfg['loader']['batch_size']) == 2
    if int(cfg["opt"]["epochs"]) != 50 or not batch_ok:
        raise ValueError("matched control requires 50 epochs and batch2")
    validate_formal_runtime(cfg)


def checked_train_result(run_root, train_log, expected_cfg, allow_accelerated=False):
    configs = list(run_root.glob("*/config.json"))
    best = list(run_root.glob("*/saved_ckpt/best_ckpt/*.pt"))
    latest = list(run_root.glob("*/saved_ckpt/latest_ckpt/checkpoint-epoch50.pt"))
    if len(configs) != 1 or len(best) != 1 or len(latest) != 1:
        raise ValueError("training did not finish with unique config/best/epoch50 checkpoints")
    cfg = json.loads(configs[0].read_text())
    check_new_protocol(cfg, allow_accelerated=allow_accelerated)
    signature = expected_cfg["rq2_formal"]["run_signature"]
    for path in (best[0], latest[0]):
        checkpoint = load_checkpoint(path)
        validate_formal_resume(checkpoint, signature)
        state = checkpoint.get("rq2_training_state", {})
        if state.get("schema") != "rq2-training-control-v2":
            raise ValueError("checkpoint lacks full model/scheduler control state")
        if path == best[0]:
            selected_checkpoint_state(checkpoint, cfg, require_provenance=True)
        elif checkpoint.get("epoch") != 50:
            raise ValueError("latest checkpoint is not epoch50")
        del checkpoint
    content = train_log.read_text(errors="replace")
    if re.search(r"Traceback|out of memory|\bnan\b|\b\w*loss\w*\s*:\s*[-+]?inf\b",
                 content, re.IGNORECASE):
        raise ValueError("training log contains an exception or nonfinite value")
    for key in ("t_target_gt_cls_loss", "t_target_gt_reg_loss", "t_target_gt_downstream_loss"):
        values = [float(value) for value in re.findall(r"\b" + key + r":\s*([-+\d.eE]+)", content)]
        if not values or any(value != 0 for value in values):
            raise ValueError("target GT loss must be present and exactly zero: " + key)
    return configs[0], best[0]


def checked_evaluation(root, cfg, checkpoint):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("evaluation checkpoint hash differs")
    if manifest["shift"] != cfg["rq2_formal"]["shift"] or manifest["seed"] != cfg["seed"]:
        raise ValueError("evaluation shift/seed differs")
    if manifest["selection_provenance"]["split"] != "source_validation":
        raise ValueError("evaluation checkpoint was not source-selected")
    policy = {key: cfg["rq2_formal"][key] for key in (
        "training_budget", "source_optimization", "epoch_budget", "rng_protocol", "data_provenance",
    )}
    if manifest["training_protocol"] != policy:
        raise ValueError("evaluation belongs to another training/data protocol")
    for name in ("source_val.json", "target_val.json"):
        path = root / name
        if manifest["outputs"][name] != sha256_file(path):
            raise ValueError("evaluation output hash differs: " + name)
        payload = json.loads(path.read_text())
        if payload["training_protocol"] != policy or not math.isfinite(payload["map_0.1_to_0.5"]):
            raise ValueError("evaluation protocol or mAP is invalid")
    return manifest


def run_command(command, log, data_root, gpu_uuid=None, gpu_index=None):
    env = dict(os.environ, PYTHONPATH=str(CODE_ROOT), PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES=gpu_uuid or "",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8")
    env.pop("RQ3_GPU_INDEX", None)
    env.pop("RQ3_GPU_UUID", None)
    if gpu_index is not None:
        env.update(RQ3_GPU_INDEX=str(gpu_index), RQ3_GPU_UUID=gpu_uuid)
    with log.open("x", encoding="utf-8") as stream:
        subprocess.run(command, cwd=data_root, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MATCHED_MODELS, default="slowfast")
    parser.add_argument("--feature-root", type=Path,
                        help="complete frozen VLM features with protocol.json and video_ids.txt")
    parser.add_argument("--shift", choices=["S" + str(i) for i in range(1, 7)], default="S1")
    parser.add_argument("--seed", type=int, choices=(6, 7, 8), default=6)
    parser.add_argument("--gpu", type=int, choices=range(4), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/vepfs/vlms/SADA_PLUS"))
    parser.add_argument("--manifest-root", type=Path,
                        default=Path("/vepfs/vlms/rq2_uda_target_manifests_0924"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    check_code(args.expected_head)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite or silently reuse old results.
    (root / "logs").mkdir()
    registry = args.data_root / "configs/rq2/shift_registry.yaml"
    if sha256_file(registry) != sha256_file(CODE_ROOT / "configs/rq2/shift_registry.yaml"):
        raise ValueError("code/data shift registries differ")
    entry = yaml.safe_load(registry.read_text())["shifts"][args.shift]
    config = CODE_ROOT / entry["config"]
    check_registry_config(config, registry, args.shift)
    reference = args.data_root / "data/ek100_slowfast/features"
    common = ["-c", str(config), "-s", str(args.seed), "-i", "0",
              "--rq2-shift", args.shift, "--rq2-registry", str(registry),
              "--rq2-reference-root", str(reference)]
    common += matched_feature_args(args.model, args.feature_root)
    train_commands, audited = {}, {}
    for regime, supervised in (("so", "source_only"), ("uda", "off")):
        name = "rq2_matched_{}_{}_seed{}_{}_{}".format(
            args.model, args.shift, args.seed, regime, root.name)
        if (args.data_root / "model_results" / name).exists():
            raise FileExistsError("new matched run name already exists: " + name)
        command = [sys.executable, "-B", str(CODE_ROOT / "train_da.py"), *common,
                   "-n", name, "--debug", "--benchmark-audit-log",
                   "--rq2-regime", regime, "--supervised-upper-bound", supervised]
        if regime == "uda":
            command += ["--rq2-target-manifest", str(args.manifest_root / (args.shift + ".json"))]
        audit_path = root / (regime + "_preflight.json")
        run_command(command + ["--rq2-audit-output", str(audit_path)],
                    root / "logs" / (regime + "_preflight.log"), args.data_root)
        audited[regime] = json.loads(audit_path.read_text())
        # Relative annotation/feature paths are rooted in the data checkout.
        previous = Path.cwd()
        os.chdir(args.data_root)
        try:
            check_new_protocol(audited[regime])
        finally:
            os.chdir(previous)
        train_commands[regime] = (name, command)
    pairing = audit_so_uda_pair(root / "so_preflight.json", root / "uda_preflight.json")
    (root / "preflight_pair.json").write_text(json.dumps(pairing, sort_keys=True))
    print("CPU preflight and SO/UDA pairing passed", flush=True)
    if args.audit_only:
        return

    # A local lock prevents our launchers from sharing this physical card.
    import fcntl
    with (args.data_root / (".rq2_gpu" + str(args.gpu) + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.chdir(args.data_root)
        actual_configs = {}
        for regime in ("so", "uda"):
            check_code(args.expected_head)
            check_new_protocol(audited[regime])
            check_registry_config(config, registry, args.shift, audited[regime])
            if regime == "uda":
                # Check the completed NEW SO against the exact UDA runtime BEFORE GPU work.
                audit_so_uda_pair(actual_configs["so"], root / "uda_preflight.json")
            uuid = idle_gpu_uuid(args.gpu)
            name, command = train_commands[regime]
            print("start {} on physical GPU{} ({})".format(name, args.gpu, uuid), flush=True)
            train_log = root / "logs" / (regime + ".train.log")
            run_command(command, train_log, args.data_root, uuid)
            config_path, checkpoint = checked_train_result(
                args.data_root / "model_results" / name, train_log, audited[regime])
            actual_configs[regime] = config_path
            if regime == "uda":
                pair = audit_so_uda_pair(actual_configs["so"], config_path)
                (root / "pair_audit.json").write_text(json.dumps(pair, sort_keys=True))
            check_code(args.expected_head)
            uuid = idle_gpu_uuid(args.gpu)
            output = root / regime / (args.shift + "_seed" + str(args.seed))
            evaluation = [sys.executable, "-B", str(CODE_ROOT / "evaluate_da.py"), *common,
                          "-n", name + "_eval", "-r", str(checkpoint),
                          "--rq2-formal", "--rq2-output-dir", str(output)]
            run_command(evaluation, root / "logs" / (regime + ".eval.log"), args.data_root, uuid)
            checked_evaluation(output, audited[regime], checkpoint)
            print(regime + " 50 epochs, checkpoint and source/target evaluation passed", flush=True)
        (root / "complete.json").write_text(json.dumps({
            "model": args.model, "shift": args.shift, "seed": args.seed,
            "gpu": args.gpu, "code_commit": args.expected_head,
            "training_budget": TRAINING_BUDGET_PROTOCOL, "rng_protocol": RNG_PROTOCOL,
            "feature_protocol_sha256": audited["so"].get("rq2_feature_protocol", {}).get("protocol_sha256"),
            "so": str(actual_configs["so"]), "uda": str(actual_configs["uda"]),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
