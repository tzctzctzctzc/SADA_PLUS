"""RQ3：三个VLM教师×三个策略×六shift×seed6。默认只打印计划，不训练。"""

import argparse
import json
import os
import math
from pathlib import Path
import sys

import yaml

from tools.rq2_dense_protocol import sha256_file
from tools.rq3_gpu import GpuLease, gpu_argument, gpu_slot
from tools.rq3_protocol import (MODELS, LEGACY_MODELS, STRATEGIES, audit_baseline, load_cache, members_for,
                                audit_cache_predictions, validate_interface_audit, validate_reference_lengths)

CODE_ROOT = Path(__file__).resolve().parents[1]


def matrix(shard_index=0, shard_count=1):
    if shard_count not in (1, 2) or not 0 <= shard_index < shard_count:
        raise ValueError("RQ3 matrix supports one or two non-overlapping shards")
    entries = [(model, strategy, "S" + str(shift))
               for model in LEGACY_MODELS for strategy in STRATEGIES for shift in range(1, 7)]
    return entries[shard_index::shard_count]


def matrix_lock_name(shard_index, shard_count):
    slot = os.environ.get('RQ3_QUEUE_SLOT')
    if slot is None:
        return '.rq3_matrix_%d_of_%d.lock' % (shard_index, shard_count)
    if slot not in ('0', '1') or not os.environ.get('RQ3_MONITOR_PID'):
        raise ValueError('invalid managed RQ3 queue slot')
    return '.rq3_matrix_managed_slot%s.lock' % slot


def baseline_root(parent, shift):
    return parent / ("rq2_matched_slowfast_{}_seed6_7067306_0927".format(shift))


def check_cache_snapshot(cache_root, snapshot):
    for identity, digest in snapshot.items():
        if sha256_file(cache_root / identity) != digest:
            raise ValueError("RQ3 semantic cache changed between strategies: " + identity)


def revised_inputs(args, model, strategy, shift):
    local = strategy == 'verify'
    cache = (args.verify_cache_root if local else args.cache_root) / model / (shift + '.json')
    manifest = (args.verify_candidate_root if local else args.candidate_root) / shift / 'manifest.json'
    calibration = args.calibration_root / strategy / model / (shift + '.json')
    return cache, manifest, calibration


def completed_entries(roots):
    """仅complete.json代表完成；旧部分目录保留，未完成项使用全新输出根。"""
    completed = {}
    allowed = {(m, s, 'S' + str(i)) for m in MODELS for s in STRATEGIES for i in range(1, 7)}
    for root in roots:
        for path in sorted(root.rglob("complete.json")):
            record = json.loads(path.read_text())
            key = (record["teacher"], record["strategy"], record["shift"])
            if key not in allowed or record["seed"] != 6 or key in completed:
                raise ValueError("RQ3 completed roots contain unknown or duplicate experiment: " + str(path))
            completed[key] = path
    return completed


def checked_completed_entry(path, args, baselines, snapshot):
    from tools.run_rq2_matched_control import check_new_protocol, checked_evaluation, checked_train_result
    record = json.loads(path.read_text())
    root = path.parent
    model, strategy, shift = record["teacher"], record["strategy"], record["shift"]
    cfg = json.loads((root / "preflight.json").read_text())
    if record.get('early_stopping') != cfg.get('rq3_auxiliary', {}).get('early_stopping'):
        raise ValueError('completed result lost stopping-policy provenance')
    accelerated = bool(cfg.get('rq3_auxiliary', {}).get('acceleration'))
    if accelerated and (record.get('batch_size') != cfg['loader']['batch_size']
                        or record.get('batch_matched_baseline') is not False):
        raise ValueError('completed accelerated result lost its batch provenance')
    check_new_protocol(cfg, allow_accelerated=accelerated)
    if record["baseline_root"] != str(baseline_root(args.baseline_parent, shift)):
        raise ValueError("completed RQ3 result uses another baseline")
    identity = str(Path(model) / (shift + ".json"))
    if record["cache_sha256"] != snapshot[identity] or cfg["rq3_auxiliary"]["cache_sha256"] != snapshot[identity]:
        raise ValueError("completed RQ3 result uses another semantic cache")
    if (cfg["rq3_auxiliary"]["strategy"] != strategy
            or cfg["rq3_auxiliary"]["teacher_model"] != model
            or cfg["rq2_formal"]["shift"] != shift or cfg["seed"] != 6):
        raise ValueError("completed RQ3 strategy/teacher differs from preflight")
    name = "rq3_{}_{}_{}_seed6_{}".format(model, strategy, shift, path.parents[3].name)
    config, checkpoint = checked_train_result(args.data_root / "model_results" / name,
                                             root / "logs/uda.train.log", cfg, allow_accelerated=accelerated)
    audit_baseline(baselines[shift][0], json.loads(config.read_text()))
    checked_evaluation(root / "evaluation", cfg, checkpoint)
    if (sha256_file(checkpoint) != record["checkpoint_sha256"]
            or sha256_file(root / "evaluation/manifest.json") != record["manifest_sha256"]):
        raise ValueError("completed RQ3 checkpoint or manifest changed")
    coverage = supervision_audit(config.parent / "benchmark_epoch_metrics.jsonl", cfg)
    if coverage != json.loads((root / "supervision_audit.json").read_text()) or not coverage["effective"]:
        raise ValueError("completed RQ3 auxiliary supervision was ineffective")


def supervision_audit(path, cfg=None):
    """只检查辅助监督实际生效，不按target增益筛选；零覆盖结果也保留评估。"""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    totals = {}
    from tools.rq3_protocol import REVISION
    auxiliary = (cfg or {}).get('rq3_auxiliary', {})
    revised = auxiliary.get('schema') == REVISION
    keys = ('eligible', 'selected', 'loss', 'gradient_norm')
    if revised and auxiliary['strategy'] == 'verify':
        keys = ('verify_eligible', 'verify_accepted', 'verify_rejected', 'verify_ambiguous',
                'verify_observed', 'verify_uncertain')
    elif revised:
        keys += ('source_loss', 'alignment_loss', 'target_gradient_norm', 'aligned_classes',
                 'candidate_count', 'fully_visible_count')
    for key in keys:
        values = [row.get("t_rq3_" + key, 0) for row in rows]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("RQ3 auxiliary metrics are nonfinite")
        totals[key + "_epoch_mean_sum"] = sum(values)
    totals["effective"] = all(value > 0 for value in totals.values())
    if revised and auxiliary['strategy'] == 'verify':
        # Verify没有新增辅助loss；全体一致时是有效核验但零干预，不强迫拒绝。
        # 有局部证据但全部不可信时保持原SADA，也是合法的零干预；不为通过门禁降阈值。
        totals['effective'] = totals['verify_observed_epoch_mean_sum'] > 0
        totals['intervened'] = totals['verify_rejected_epoch_mean_sum'] > 0
    elif revised:
        totals['intervened'] = totals['selected_epoch_mean_sum'] > 0
        visible_source = all(totals[key + '_epoch_mean_sum'] > 0
                             for key in ('source_loss', 'gradient_norm', 'fully_visible_count'))
        target_keys = ('alignment_loss', 'target_gradient_norm', 'aligned_classes')
        target_ok = (all(totals[key + '_epoch_mean_sum'] > 0 for key in target_keys)
                     if totals['intervened'] else all(totals[key + '_epoch_mean_sum'] == 0 for key in target_keys))
        # 可信门槛没有放行样本时允许诚实记录零VLM干预，不能为通过门禁降低阈值。
        totals['effective'] = visible_source and target_ok
    if revised and auxiliary.get('participation_fraction') == .2:
        # 新协议必须实际使用已选证据；Verify全体同意可以不拒绝。
        if auxiliary['strategy'] == 'verify':
            totals['effective'] = totals['verify_eligible_epoch_mean_sum'] > 0
        else:
            totals['effective'] = totals['effective'] and totals['intervened']
    return totals


def load_entry_plan(path):
    plan = json.loads(Path(path).read_text())
    expanded = plan.get('schema') == 'rq3-expanded-fresh-matrix-v1'
    fresh_matrix = plan.get('schema') in ('rq3-revised-fresh-matrix-v1', 'rq3-expanded-fresh-matrix-v1')
    if plan.get('schema') not in ('rq3-revised-completed-only-v1', 'rq3-revised-fresh-matrix-v1', 'rq3-expanded-fresh-matrix-v1'):
        raise ValueError('unknown RQ3 rerun plan')
    if fresh_matrix and (plan.get('seed') != 6 or plan.get('fresh') is not True):
        raise ValueError('fresh RQ3 matrix requires seed6 and fresh=true')
    entries = []
    for row in plan['entries']:
        entry = (row['teacher'], row['strategy'], row['shift'])
        allowed = ((entry[0] in MODELS and entry[1] in STRATEGIES and entry[2] in ['S'+str(i) for i in range(1,7)])
                   if expanded else entry in matrix() and entry[1] in ('verify', 'prototype'))
        if not allowed or entry in entries:
            raise ValueError('invalid/duplicate entry for the explicit RQ3 plan schema')
        if not fresh_matrix:
            original = Path(row['original_complete'])
            if sha256_file(original) != row['original_complete_sha256']:
                raise ValueError('original RQ3 result identity changed')
            for artifact, digest in row.get('artifact_sha256', {}).items():
                if sha256_file(artifact) != digest:
                    raise ValueError('original RQ3 artifact changed: ' + artifact)
            record = json.loads(original.read_text())
            if (record['teacher'], record['strategy'], record['shift']) != entry or record['seed'] != 6:
                raise ValueError('rerun entry differs from original complete result')
        entries.append(entry)
    if not entries:
        raise ValueError('empty RQ3 rerun plan')
    return entries


def audit_shifts(entries, fresh_matrix):
    return sorted({row[2] for row in entries}) if fresh_matrix else ['S' + str(i) for i in range(1, 7)]


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
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="<root>/<teacher>/<shift>.json；三策略复用同一缓存")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-parent", type=Path, default=Path("/vepfs/vlms"))
    parser.add_argument("--data-root", type=Path, default=Path("/vepfs/vlms/SADA_PLUS"))
    parser.add_argument("--manifest-root", type=Path, default=Path("/vepfs/vlms/rq2_uda_target_manifests_0924"))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument('--entry-plan', type=Path)
    parser.add_argument('--batch-size', type=int, choices=(2, 4, 8), default=2)
    parser.add_argument('--early-stop-patience', type=int, choices=(0, 15), default=0)
    parser.add_argument('--benchmark-only', action='store_true', help='真实短程吞吐测试，不生成complete')
    parser.add_argument('--candidate-root', type=Path)
    parser.add_argument('--verify-cache-root', type=Path, help='重新打分的局部窗口缓存，不能复用整段概率')
    parser.add_argument('--verify-candidate-root', type=Path)
    parser.add_argument('--calibration-root', type=Path, help='<strategy>/<teacher>/<shift>.json，源域校准')
    parser.add_argument('--revision', choices=('legacy', 'revised'), default='legacy')
    parser.add_argument("--completed-root", type=Path, action="append", default=[],
                        help="只读审计并跳过已完成实验；可重复传入原分片根")
    parser.add_argument("--gpu-idle-minutes", type=float, default=0,
                        help="首次占卡前连续空闲时长；取得卡后本分片复用同卡")
    parser.add_argument('--immediate-gpu', action='store_true',
                        help='用户明确授权本次直接启动；仍须锁内实际空闲核验')
    parser.add_argument("--audit-only", action="store_true", help="真实train_da CPU预检，不创建GPU任务")
    parser.add_argument("--execute", action="store_true", help="执行本分片；两个分片可用两张空闲卡并行")
    args = parser.parse_args()
    entries = matrix(args.shard_index, args.shard_count)
    if args.revision == 'revised':
        if not all((args.entry_plan, args.candidate_root, args.verify_cache_root,
                    args.verify_candidate_root, args.calibration_root)) or args.completed_root:
            parser.error('revised reruns require explicit plan, proposal/local inputs and source calibrations; no completed-root')
        if args.execute and args.gpu_idle_minutes < 60 and not args.immediate_gpu:
            parser.error('revised GPU runs must observe at least 60 continuous idle minutes')
        if args.immediate_gpu and args.gpu_idle_minutes:
            parser.error('immediate-gpu and continuous idle timer are mutually exclusive')
        args.entry_plan = args.entry_plan.resolve()
        args.candidate_root = args.candidate_root.resolve()
        args.verify_cache_root = args.verify_cache_root.resolve()
        args.verify_candidate_root = args.verify_candidate_root.resolve()
        args.calibration_root = args.calibration_root.resolve()
        entries = load_entry_plan(args.entry_plan)[args.shard_index::args.shard_count]
    elif any((args.entry_plan, args.candidate_root, args.verify_cache_root, args.verify_candidate_root, args.calibration_root)):
        parser.error('entry-plan/candidate-root require revised protocol')
    completed = completed_entries([root.resolve() for root in args.completed_root])
    if args.gpu_idle_minutes < 0:
        parser.error("--gpu-idle-minutes cannot be negative")
    if args.audit_only and args.execute:
        parser.error("--audit-only and --execute are exclusive")
    if not args.audit_only and not args.execute:
        for model, strategy, shift in entries:
            if (model, strategy, shift) in completed:
                continue
            cache = (revised_inputs(args, model, strategy, shift)[0] if args.revision == 'revised'
                     else args.cache_root / model / (shift + '.json'))
            print("{} {} {} seed6 gpu={} cache={}".format(model, strategy, shift, args.gpu, cache))
        return
    from tools.run_rq2_matched_control import (
        check_code, check_new_protocol, checked_evaluation, checked_train_result,
        run_command,
    )
    # 后面会切换到数据工作树；相对路径只在调用目录解释一次。
    for key in ("cache_root", "output_root", "baseline_parent", "data_root", "manifest_root"):
        setattr(args, key, getattr(args, key).resolve())
    check_code(args.expected_head)
    fresh_matrix = bool(args.entry_plan and json.loads(args.entry_plan.read_text())['schema'] in ('rq3-revised-fresh-matrix-v1', 'rq3-expanded-fresh-matrix-v1'))
    args.output_root.mkdir(parents=True, exist_ok=False)
    if args.entry_plan:
        (args.output_root / 'entry_plan.json').write_bytes(args.entry_plan.read_bytes())
    registry = args.data_root / "configs/rq2/shift_registry.yaml"
    if sha256_file(registry) != sha256_file(CODE_ROOT / "configs/rq2/shift_registry.yaml"):
        raise ValueError("RQ3 code/data registries differ")
    shifts = yaml.safe_load(registry.read_text())["shifts"]
    previous = Path.cwd()
    lease = (GpuLease(args.gpu, args.data_root, args.gpu_idle_minutes * 60)
             if args.gpu_idle_minutes and args.execute else None)
    os.chdir(args.data_root)
    try:
        baselines = {shift: completed_baseline(baseline_root(args.baseline_parent, shift), shift)
                     for shift in audit_shifts(entries, fresh_matrix)}
        executing_models = {row[0] for row in entries}
        # 三模型每shift必须实际使用同一候选、SO teacher和帧；不按增益筛选。
        shared, snapshot = {}, {}
        for model in sorted(set(LEGACY_MODELS) | executing_models):
            for shift, (cfg, teacher_checkpoint) in baselines.items():
                path = args.cache_root / model / (shift + ".json")
                payload, _ = load_cache(path, model, shift, members_for(cfg, "tgt"))
                # 非本次执行教师仍比较候选/像素/预测身份；大权重只重验实际执行教师。
                interface = validate_interface_audit(payload["input_protocol"], model,
                                                     verify_files=not fresh_matrix or model in executing_models)
                audit_cache_predictions(payload, interface)
                if not fresh_matrix or model in executing_models:
                    validate_reference_lengths(payload, cfg)
                snapshot[str(path.relative_to(args.cache_root))] = sha256_file(path)
                if payload["teacher_checkpoint_sha256"] != sha256_file(teacher_checkpoint):
                    raise ValueError("RQ3 cache does not use the completed source-selected SO teacher")
                identity = (payload["candidate_sha256"], payload["frames_sha256"], payload["teacher_checkpoint_sha256"])
                if shift in shared and shared[shift] != identity:
                    raise ValueError("RQ3 VLMs use different candidate/frame/teacher inputs")
                shared[shift] = identity
        (args.output_root / "cache_snapshot.json").write_text(json.dumps(snapshot, sort_keys=True))
        trusted_snapshot, scoped_shared = {}, {}
        if args.revision == 'revised':
            from tools.rq3_protocol import proposal_evidence
            from tools.rq3_confidence import load_calibration
            for model, strategy, shift in entries:
                if strategy == 'hard':
                    # Hard保持旧全候选硬标签辅助；上面的缓存/帧/权重审计同样适用。
                    if json.loads(args.entry_plan.read_text())['schema'] != 'rq3-expanded-fresh-matrix-v1':
                        raise ValueError('Hard requires explicit expanded plan')
                    continue
                cache, manifest, calibration = revised_inputs(args, model, strategy, shift)
                cfg, so_checkpoint = baselines[shift]
                payload, _ = load_cache(cache, model, shift, members_for(cfg, 'tgt'))
                interface = validate_interface_audit(payload['input_protocol'], model)
                audit_cache_predictions(payload, interface)
                validate_reference_lengths(payload, cfg)
                proposal_evidence(payload, manifest)
                report = load_calibration(calibration, cfg, payload, json.loads(manifest.read_text()), strategy)
                old, _ = load_cache(args.cache_root / model / (shift + '.json'))
                coordinates = lambda value: [(r['video_id'], r['absolute_feature_index'], r['source_fpn_levels'])
                                              for r in value['entries']]
                if coordinates(payload) != coordinates(old) or payload['teacher_checkpoint_sha256'] != sha256_file(so_checkpoint):
                    raise ValueError('trusted rerun changed frozen candidate identities or SO teacher')
                identity = (payload['candidate_sha256'], payload['frames_sha256'])
                key = (strategy, shift)
                if key in scoped_shared and scoped_shared[key] != identity:
                    raise ValueError('teachers scored different evidence within revised strategy/shift')
                scoped_shared[key] = identity
                for file in (cache, manifest, calibration, Path(report['source_cache_path']), Path(report['source_manifest_path'])):
                    trusted_snapshot[str(file.resolve())] = sha256_file(file)
            (args.output_root / 'trusted_input_snapshot.json').write_text(json.dumps(trusted_snapshot, sort_keys=True))
        if args.entry_plan and not fresh_matrix:
            for row in json.loads(args.entry_plan.read_text())['entries']:
                if (row['teacher'], row['strategy'], row['shift']) in entries:
                    checked_completed_entry(Path(row['original_complete']), args, baselines, snapshot)
            print('RQ3_ORIGINAL_RESULTS_REAUDITED', len(entries), flush=True)
        # 防止重复分片；两个分片成员互斥，每个GPU阶段再取得共享容量与卡锁。
        import fcntl
        with (args.data_root / matrix_lock_name(args.shard_index, args.shard_count)).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for model, strategy, shift in entries:
                check_cache_snapshot(args.cache_root, snapshot)
                check_cache_snapshot(Path('.'), trusted_snapshot)
                if (model, strategy, shift) in completed:
                    path = completed[(model, strategy, shift)]
                    checked_completed_entry(path, args, baselines, snapshot)
                    print("RQ3_COMPLETED_SKIPPED " + str(path), flush=True)
                    continue
                print("RQ3_PENDING_ENTRY {} {} {} seed6".format(model, strategy, shift), flush=True)
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
                           "--rq3-teacher", model, "--rq3-cache", str(args.cache_root / model / (shift + ".json")),
                           '--rq3-batch-size', str(args.batch_size)]
                command += ['--rq3-early-stop-patience', str(args.early_stop_patience)]
                if args.revision == 'revised' and strategy != 'hard':
                    cache, manifest, calibration = revised_inputs(args, model, strategy, shift)
                    command[command.index('--rq3-cache') + 1] = str(cache)
                    command += ['--rq3-revision', 'revised', '--rq3-candidate-manifest',
                                str(manifest), '--rq3-confidence-calibration', str(calibration)]
                check_code(args.expected_head)
                preflight = root / "preflight.json"
                run_command(command + ["--rq2-audit-output", str(preflight)], root / "logs/preflight.log", args.data_root)
                cfg = json.loads(preflight.read_text())
                check_new_protocol(cfg, allow_accelerated=args.batch_size != 2)
                if args.revision == 'revised' and strategy != 'hard':
                    from tools.rq3_protocol import REVISION
                    if cfg['rq3_auxiliary']['schema'] != REVISION:
                        raise ValueError('revised queue reached legacy strategy entry')
                pairing = audit_baseline(baselines[shift][0], cfg)
                identity = str(Path(model) / (shift + ".json"))
                expected_cache_hash = (sha256_file(cache) if args.revision == 'revised' and strategy != 'hard' else snapshot[identity])
                if cfg["rq3_auxiliary"]["cache_sha256"] != expected_cache_hash:
                    raise ValueError("RQ3 preflight used a different semantic cache")
                (root / "baseline_pair.json").write_text(json.dumps(pairing, sort_keys=True))
                if args.audit_only:
                    continue
                if args.benchmark_only:
                    if len(entries) != 1 or args.batch_size not in (4, 8):
                        raise ValueError('benchmark needs one accelerated entry')
                    report = args.output_root / 'batch_benchmark.json'
                    benchmark_command = [sys.executable, '-B', str(CODE_ROOT / 'tools/benchmark_rq3_batch.py'),
                                         str(report), *command[2:]]
                    with gpu_slot(args.gpu, args.data_root, args.wait_gpu) as selected:
                        run_command(benchmark_command, root / 'logs/batch_benchmark.log', args.data_root,
                                    selected['uuid'], gpu_index=selected['index'])
                    continue
                check_code(args.expected_head)
                check_cache_snapshot(args.cache_root, snapshot)
                check_cache_snapshot(Path('.'), trusted_snapshot)
                with (lease.stage() if lease else gpu_slot(args.gpu, args.data_root, args.wait_gpu)) as training_gpu:
                    check_code(args.expected_head)
                    check_cache_snapshot(args.cache_root, snapshot)
                    check_cache_snapshot(Path('.'), trusted_snapshot)
                    run_command(command, root / "logs/uda.train.log", args.data_root,
                                training_gpu["uuid"], gpu_index=training_gpu["index"])
                config_path, checkpoint = checked_train_result(args.data_root / "model_results" / name,
                                                              root / "logs/uda.train.log", cfg, allow_accelerated=args.batch_size != 2)
                audit_baseline(baselines[shift][0], json.loads(config_path.read_text()))
                check_code(args.expected_head)
                check_cache_snapshot(args.cache_root, snapshot)
                evaluation = [sys.executable, "-B", str(CODE_ROOT / "evaluate_da.py"), *common,
                              "-n", name + "_eval", "-r", str(checkpoint), "--rq2-formal",
                              "--rq2-output-dir", str(root / "evaluation")]
                with (lease.stage() if lease else gpu_slot(args.gpu, args.data_root, args.wait_gpu)) as evaluation_gpu:
                    check_code(args.expected_head)
                    check_cache_snapshot(args.cache_root, snapshot)
                    check_cache_snapshot(Path('.'), trusted_snapshot)
                    run_command(evaluation, root / "logs/uda.eval.log", args.data_root,
                                evaluation_gpu["uuid"], gpu_index=evaluation_gpu["index"])
                manifest = checked_evaluation(root / "evaluation", cfg, checkpoint)
                if manifest.get("rq3_auxiliary") != cfg["rq3_auxiliary"]:
                    raise ValueError("RQ3 evaluation lost strategy/cache provenance")
                check_cache_snapshot(args.cache_root, snapshot)
                check_cache_snapshot(Path('.'), trusted_snapshot)
                coverage = supervision_audit(config_path.parent / "benchmark_epoch_metrics.jsonl", cfg)
                (root / "supervision_audit.json").write_text(json.dumps(coverage, sort_keys=True))
                if not coverage["effective"]:
                    raise ValueError("RQ3 auxiliary supervision never activated; outputs retained, queue stopped")
                (root / "complete.json").write_text(json.dumps({
                    "teacher": model, "strategy": strategy, "shift": shift, "seed": 6,
                    "single_seed_stage_result": True, "code_commit": args.expected_head,
                    "batch_size": args.batch_size, "batch_matched_baseline": args.batch_size == 2,
                    "early_stopping": cfg['rq3_auxiliary'].get('early_stopping'),
                    "stopping_budget_matched_baseline": not bool(args.early_stop_patience),
                    "training_gpu": training_gpu, "evaluation_gpu": evaluation_gpu,
                    "baseline_root": str(baseline_root(args.baseline_parent, shift)),
                    "cache_sha256": cfg["rq3_auxiliary"]["cache_sha256"],
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest_sha256": sha256_file(root / "evaluation/manifest.json"),
                }, sort_keys=True))
    finally:
        if lease is not None:
            lease.close()
        os.chdir(previous)


if __name__ == "__main__":
    main()
