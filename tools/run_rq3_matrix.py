"""RQ3：三个VLM教师×三个策略×六shift×seed6。默认只打印计划，不训练。"""

import argparse
import json
import os
import math
from pathlib import Path
import sys

import yaml

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_gpu import gpu_argument, gpu_slot
from tools.rq3_protocol import (MODELS, STRATEGIES, audit_baseline, load_cache, members_for,
                                audit_cache_predictions, validate_interface_audit, validate_reference_lengths)

CODE_ROOT = Path(__file__).resolve().parents[1]


def matrix():
    return [(model, strategy, "S" + str(shift))
            for model in MODELS for strategy in STRATEGIES for shift in range(1, 7)]


def baseline_root(parent, shift):
    return parent / ("rq2_matched_slowfast_{}_seed6_7067306_0927".format(shift))


def check_cache_snapshot(cache_root, snapshot):
    for identity, digest in snapshot.items():
        if sha256_file(cache_root / identity) != digest:
            raise ValueError("RQ3 semantic cache changed between strategies: " + identity)


def supervision_audit(path):
    """只检查辅助监督实际生效，不按target增益筛选；零覆盖结果也保留评估。"""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    totals = {}
    for key in ("eligible", "selected", "loss", "gradient_norm"):
        values = [row.get("t_rq3_" + key, 0) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("RQ3 auxiliary metrics are nonfinite")
        totals[key + "_epoch_mean_sum"] = sum(values)
    totals["effective"] = all(value > 0 for value in totals.values())
    return totals


def completed_baseline(root, shift):
    from tools.run_rq2_matched_control import check_new_protocol, checked_evaluation, checked_train_result
    complete = json.loads((root / "complete.json").read_text())
    if complete["model"] != "slowfast" or complete["shift"] != shift or complete["seed"] != 6:
        raise ValueError("RQ3 baseline root belongs to another model/shift/seed")
    configs, checkpoints = {}, {}
    for regime in ("so", "uda"):
        path = Path(complete[regime])
        cfg = json.loads(path.read_text())
        check_new_protocol(cfg)
        run_root = path.parent.parent
        checked_path, checkpoint = checked_train_result(run_root, root / "logs" / (regime + ".train.log"), cfg)
        if checked_path != path:
            raise ValueError("completed baseline checkpoint/config identity differs")
        checked_evaluation(root / regime / (shift + "_seed6"), cfg, checkpoint)
        configs[regime], checkpoints[regime] = cfg, checkpoint
    from tools.audit_rq2_vlm_benchmark import audit_so_uda_pair
    audit_so_uda_pair(Path(complete["so"]), Path(complete["uda"]))
    return configs["uda"], checkpoints["so"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=gpu_argument, default="auto")
    parser.add_argument("--wait-gpu", action="store_true")
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="<root>/<teacher>/<shift>.json；三策略复用同一缓存")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-parent", type=Path, default=Path("/vepfs/vlms"))
    parser.add_argument("--data-root", type=Path, default=Path("/vepfs/vlms/SADA_PLUS"))
    parser.add_argument("--manifest-root", type=Path, default=Path("/vepfs/vlms/rq2_uda_target_manifests_0924"))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--audit-only", action="store_true", help="真实train_da CPU预检，不创建GPU任务")
    parser.add_argument("--execute", action="store_true", help="显式执行54项；同时只用一张空闲卡")
    args = parser.parse_args()
    if args.audit_only and args.execute:
        parser.error("--audit-only and --execute are exclusive")
    if not args.audit_only and not args.execute:
        for model, strategy, shift in matrix():
            print("{} {} {} seed6 gpu={} cache={}".format(model, strategy, shift, args.gpu,
                                                        args.cache_root / model / (shift + ".json")))
        return
    from tools.run_rq2_matched_control import (
        check_code, check_new_protocol, checked_evaluation, checked_train_result,
        run_command,
    )
    # 后面会切换到数据工作树；相对路径只在调用目录解释一次。
    for key in ("cache_root", "output_root", "baseline_parent", "data_root", "manifest_root"):
        setattr(args, key, getattr(args, key).resolve())
    check_code(args.expected_head)
    args.output_root.mkdir(parents=True, exist_ok=False)
    registry = args.data_root / "configs/rq2/shift_registry.yaml"
    if sha256_file(registry) != sha256_file(CODE_ROOT / "configs/rq2/shift_registry.yaml"):
        raise ValueError("RQ3 code/data registries differ")
    shifts = yaml.safe_load(registry.read_text())["shifts"]
    previous = Path.cwd()
    os.chdir(args.data_root)
    try:
        baselines = {shift: completed_baseline(baseline_root(args.baseline_parent, shift), shift)
                     for shift in ("S" + str(index) for index in range(1, 7))}
        # 三模型每shift必须实际使用同一候选、SO teacher和帧；不按增益筛选。
        shared, snapshot = {}, {}
        for model in MODELS:
            for shift, (cfg, teacher_checkpoint) in baselines.items():
                path = args.cache_root / model / (shift + ".json")
                payload, _ = load_cache(path, model, shift, members_for(cfg, "tgt"))
                interface = validate_interface_audit(payload["input_protocol"], model)
                audit_cache_predictions(payload, interface)
                validate_reference_lengths(payload, cfg)
                snapshot[str(path.relative_to(args.cache_root))] = sha256_file(path)
                if payload["teacher_checkpoint_sha256"] != sha256_file(teacher_checkpoint):
                    raise ValueError("RQ3 cache does not use the completed source-selected SO teacher")
                identity = (payload["candidate_sha256"], payload["frames_sha256"], payload["teacher_checkpoint_sha256"])
                if shift in shared and shared[shift] != identity:
                    raise ValueError("RQ3 VLMs use different candidate/frame/teacher inputs")
                shared[shift] = identity
        (args.output_root / "cache_snapshot.json").write_text(json.dumps(snapshot, sort_keys=True))
        # 防止重复矩阵；每个GPU阶段另外取得跨RQ3入口的单卡锁。
        import fcntl
        with (args.data_root / ".rq3_matrix.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for model, strategy, shift in matrix():
                check_cache_snapshot(args.cache_root, snapshot)
                root = args.output_root / model / strategy / (shift + "_seed6")
                (root / "logs").mkdir(parents=True, exist_ok=False)
                name = "rq3_{}_{}_{}_seed6_{}".format(model, strategy, shift, args.output_root.name)
                if (args.data_root / "model_results" / name).exists():
                    raise FileExistsError("RQ3 result exists; never overwrite partial or completed runs")
                config = CODE_ROOT / shifts[shift]["config"]
                common = ["-c", str(config), "-s", "6", "-i", "0", "--rq2-shift", shift,
                          "--rq2-registry", str(registry), "--rq2-reference-root",
                          str(args.data_root / "data/ek100_slowfast/features")]
                command = [sys.executable, "-B", str(CODE_ROOT / "train_da.py"), *common, "-n", name,
                           "--debug", "--benchmark-audit-log", "--rq2-regime", "uda",
                           "--supervised-upper-bound", "off", "--rq2-target-manifest",
                           str(args.manifest_root / (shift + ".json")), "--rq3-strategy", strategy,
                           "--rq3-teacher", model, "--rq3-cache", str(args.cache_root / model / (shift + ".json"))]
                check_code(args.expected_head)
                preflight = root / "preflight.json"
                run_command(command + ["--rq2-audit-output", str(preflight)], root / "logs/preflight.log", args.data_root)
                cfg = json.loads(preflight.read_text())
                check_new_protocol(cfg)
                pairing = audit_baseline(baselines[shift][0], cfg)
                identity = str(Path(model) / (shift + ".json"))
                if cfg["rq3_auxiliary"]["cache_sha256"] != snapshot[identity]:
                    raise ValueError("RQ3 preflight used a different semantic cache")
                (root / "baseline_pair.json").write_text(json.dumps(pairing, sort_keys=True))
                if args.audit_only:
                    continue
                check_code(args.expected_head)
                check_cache_snapshot(args.cache_root, snapshot)
                with gpu_slot(args.gpu, args.data_root, args.wait_gpu) as training_gpu:
                    check_code(args.expected_head)
                    check_cache_snapshot(args.cache_root, snapshot)
                    run_command(command, root / "logs/uda.train.log", args.data_root, training_gpu["uuid"])
                config_path, checkpoint = checked_train_result(args.data_root / "model_results" / name,
                                                              root / "logs/uda.train.log", cfg)
                audit_baseline(baselines[shift][0], json.loads(config_path.read_text()))
                check_code(args.expected_head)
                check_cache_snapshot(args.cache_root, snapshot)
                evaluation = [sys.executable, "-B", str(CODE_ROOT / "evaluate_da.py"), *common,
                              "-n", name + "_eval", "-r", str(checkpoint), "--rq2-formal",
                              "--rq2-output-dir", str(root / "evaluation")]
                with gpu_slot(args.gpu, args.data_root, args.wait_gpu) as evaluation_gpu:
                    check_code(args.expected_head)
                    check_cache_snapshot(args.cache_root, snapshot)
                    run_command(evaluation, root / "logs/uda.eval.log", args.data_root, evaluation_gpu["uuid"])
                manifest = checked_evaluation(root / "evaluation", cfg, checkpoint)
                if manifest.get("rq3_auxiliary") != cfg["rq3_auxiliary"]:
                    raise ValueError("RQ3 evaluation lost strategy/cache provenance")
                check_cache_snapshot(args.cache_root, snapshot)
                coverage = supervision_audit(config_path.parent / "benchmark_epoch_metrics.jsonl")
                (root / "supervision_audit.json").write_text(json.dumps(coverage, sort_keys=True))
                if not coverage["effective"]:
                    raise ValueError("RQ3 auxiliary supervision never activated; outputs retained, queue stopped")
                (root / "complete.json").write_text(json.dumps({
                    "teacher": model, "strategy": strategy, "shift": shift, "seed": 6,
                    "single_seed_stage_result": True, "code_commit": args.expected_head,
                    "training_gpu": training_gpu, "evaluation_gpu": evaluation_gpu,
                    "baseline_root": str(baseline_root(args.baseline_parent, shift)),
                    "cache_sha256": cfg["rq3_auxiliary"]["cache_sha256"],
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest_sha256": sha256_file(root / "evaluation/manifest.json"),
                }, sort_keys=True))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
