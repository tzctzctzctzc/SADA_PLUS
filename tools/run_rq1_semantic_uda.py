"""RQ1 已知边界分类的匹配 SO/UDA 入口；preflight/benchmark 不计完成实验。"""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
import yaml

from tools.build_rq1_semantic_manifest import SHIFT_CONFIGS
from tools.rq1_semantic_uda import (Recipe, SemanticModel, CyclingSampler, fit,
    make_optimizer, optimization_step, predict, metrics, synchronize, update_ema)


MODELS = {
    'clip_b16': 'source_probe_8f/features/clip_vit_b16_static',
    'clip_l14': 'rq1_followups_8f/features/clip_vit_l14_336_static',
    'ego': 'egovideo_l_8f/features/egovideo_l_joint_8f',
    'intern': 'protocol_v2_internvideo2_full/internvideo2_clip_1b',
    'qwen': 'protocol_v2_qwen8_full/features/qwen3_vl_8b_multimodal_full',
    'videochat3': 'protocol_v2_frontier_models/protocol_v2_videochat3_full/features/videochat3_4b_multimodal_full',
    'videochat_flash': 'protocol_v2_frontier_models/protocol_v2_videochat_flash_full/features/videochat_flash_qwen25_7b_multimodal_full',
    'moss': 'protocol_v2_frontier_models/protocol_v2_moss_vl_full/features/moss_vl_instruct_0708_multimodal_full',
}
ROLES = dict(source_train='src_train_split', target_train='tgt_train_split',
             source_val='src_val_split', target_val='tgt_val_split')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def partitions(rows, config):
    result = {role: np.asarray([i for i, row in enumerate(rows)
                               if row['subset'].lower() in {s.lower() for s in config[key]}], dtype=np.int64)
              for role, key in ROLES.items()}
    groups = {role: {rows[i]['video_id'] for i in indices} for role, indices in result.items()}
    for index, role in enumerate(ROLES):
        if not len(result[role]):
            raise ValueError('empty partition: ' + role)
        for other in list(ROLES)[index + 1:]:
            if groups[role] & groups[other]:
                raise ValueError('video leakage: ' + role + '/' + other)
    return result


def recipe_from_config(config, seed=6):
    m, o, t = config['model'], config['opt'], config['train_cfg']
    return Recipe(seed=seed, lr=o['learning_rate'], disc_lr=o['disc_learning_rate'],
        min_lr=o['eta_min'], weight_decay=o['weight_decay'], warmup=o['warmup_epochs'],
        adv_start=t['warmup_adv'], beta=m['beta'], threshold=m['pl_threshold'],
        source_domain_weight=m['weight_source_cond_adv'], target_domain_weight=m['weight_target_cond_adv'],
        adv_weight=m['loss_weights']['weight_adv'], task_weight=m['loss_weights']['weight_downstream'],
        disc_layers=m['num_conditional_adv_layers'], disc_dropout=m['conditional_adv_drop_rate'],
        disc_bias=m['conditional_adv_bias'], ema_decay=t['ema_decay'], val_every=t['valid_freq'])


def load_store(prefix, manifest_sha, count):
    meta_path = Path(str(prefix) + '.meta.json')
    meta = json.loads(meta_path.read_text())
    features_path, status_path = Path(str(prefix) + '.features.npy'), Path(str(prefix) + '.status.npy')
    features = np.load(features_path, mmap_mode='r')
    status = np.load(status_path)
    if (meta['schema'] != 'rq1-frozen-segment-features-v1' or meta['manifest_sha256'] != manifest_sha
        or features.shape != (count, meta['dimensions']) or meta['rows'] != count
        or meta['frame_count'] != 8 or status.shape != (count,) or not (status == 1).all()):
        raise ValueError('incomplete/mismatched feature store: ' + str(prefix))
    for start in range(0, count, 2048):
        block = np.asarray(features[start:start + 2048], dtype=np.float32)
        if not np.isfinite(block).all() or (np.linalg.norm(block, axis=1) <= 1e-12).any():
            raise ValueError('nonfinite/zero feature vector')
    return features, {'meta': meta, 'features_sha256': sha(features_path),
                      'status_sha256': sha(status_path), 'meta_sha256': sha(meta_path)}


def tensor_features(features, indices, device):
    array = np.array(features[indices], dtype=np.float32, copy=True)
    array /= np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)
    return torch.from_numpy(array).to(device)


def training_inputs(features, rows, split, device):
    # 不读取 target_train 的 gt_label_id，不把 target_val 特征交给训练器。
    xs = tensor_features(features, split['source_train'], device)
    ys = torch.tensor([rows[i]['gt_label_id'] for i in split['source_train']], device=device)
    xt = tensor_features(features, split['target_train'], device)
    xv = tensor_features(features, split['source_val'], device)
    yv = torch.tensor([rows[i]['gt_label_id'] for i in split['source_val']], device=device)
    return xs, ys, xt, xv, yv


def benchmark(inputs, recipe, count):
    from copy import deepcopy
    xs, ys, xt, xv, yv = inputs
    results = {}
    for regime, active in [('so', False), ('so', True), ('uda', True)]:
        torch.manual_seed(recipe.seed)
        model = SemanticModel(xs.shape[1], recipe).to(xs.device).train()
        ema, optimizer = deepcopy(model).eval(), make_optimizer(model)
        source, target = CyclingSampler(len(xs), recipe.batch_size, 107), CyclingSampler(len(xt), recipe.batch_size, 208)
        # CUDA首次初始化/缓存预热不纳入稳定吞吐；计时包含取样、反传和EMA。
        for step in range(count + 5):
            if step == 5:
                synchronize(xs.device)
                start = time.perf_counter()
            si, ti = source.next().to(xs.device), target.next().to(xs.device)
            optimization_step(model, optimizer, xs[si], ys[si], xt[ti], active, regime == 'uda')
            update_ema(ema, model)
        synchronize(xs.device)
        seconds = (time.perf_counter() - start) / count
        start = time.perf_counter()
        predict(ema, xv)
        synchronize(xs.device)
        results[regime + ('_adaptation' if active else '_warmup')] = {
            'seconds_per_step': seconds, 'source_validation_seconds': time.perf_counter() - start}
    return results


def run_item(features, rows, split, recipe, regime, output, provenance, device):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', {**provenance, 'regime': regime, 'recipe': asdict(recipe),
        'oracle_action_boundaries': True, 'target_train_labels_used': False,
        'target_val_used_for_training_or_selection': False, 'frozen_vlm': True,
        'classifier': 'trainable_512_adapter_softmax', 'background_alignment': False,
        'protocol': 'rq1-known-boundary-sada-semantic-v1',
        'source_selection': 'EMA source validation CE', 'same_source_updates_so_uda': True,
        'so_uda_max_budget_matched': True, 'actual_stop_may_differ': True,
        'phase_optimizer': 'separate_warmup_and_alignment_AdamW_states',
        'rq3_differences': ['segment_batch128_not_video_batch2', 'single_scale_foreground_only',
            'softmax_CE_not_TAL_focal_regression', 'trainable_adapter_on_frozen_VLM',
            'no_Verify_or_Prototype_teacher', 'classification_LR_schedule_shared_SO_UDA']})
    model, history, termination = fit(*training_inputs(features, rows, split, device),
                                      recipe, regime, output)
    scores = {}
    for role in ('source_val', 'target_val'):
        labels = np.asarray([rows[i]['gt_label_id'] for i in split[role]], dtype=np.int64)
        p = predict(model, tensor_features(features, split[role], device)).cpu().numpy()
        scores[role] = metrics(p, labels)
        np.savez_compressed(output / (role + '_predictions.npz'), probabilities=p,
                            labels=labels, indices=split[role],
                            candidate_ids=np.asarray([rows[i]['candidate_id'] for i in split[role]]))
    write_json(output / 'metrics.json', scores)
    alignment_active = any(row['target_embedding_gradient'] > 0 for row in history)
    write_json(output / 'audit.json', {'termination': termination, 'target_gt_loss': 0.,
        'target_alignment_effective': alignment_active,
        'note': 'Zero accepted target support means no demonstrated target alignment; never lower threshold based on target results.'})
    files = ['best.pt', 'latest.pt', 'history.jsonl', 'manifest.json', 'metrics.json', 'audit.json',
             'source_val_predictions.npz', 'target_val_predictions.npz']
    write_json(output / 'complete.json', {'protocol': 'rq1-known-boundary-sada-semantic-v1',
        'target_alignment_effective': alignment_active, 'sha256': {name: sha(output / name) for name in files}})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--artifact-root', type=Path, default=Path('/vepfs/vlms/SADA_PLUS/artifacts/rq1_semantic_20260915'))
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--mode', choices=['preflight', 'benchmark', 'train'], default='preflight')
    p.add_argument('--models', default=','.join(MODELS))
    p.add_argument('--shifts', default=','.join(SHIFT_CONFIGS))
    p.add_argument('--regimes', default='so,uda')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--gpu', type=int)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--benchmark-steps', type=int, default=30)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    args = p.parse_args()
    if args.cpu == (args.gpu is not None):
        p.error('select exactly one of --cpu or --gpu PHYSICAL_INDEX')
    if args.shards < 1 or not 0 <= args.shard_index < args.shards or args.benchmark_steps < 1:
        p.error('invalid shard or benchmark length')
    model_names, shifts, regimes = args.models.split(','), args.shifts.split(','), args.regimes.split(',')
    if not set(model_names) <= MODELS.keys() or not set(shifts) <= SHIFT_CONFIGS.keys() or not set(regimes) <= {'so', 'uda'}:
        p.error('unknown model, shift or regime')
    repo = Path(__file__).resolve().parents[1]
    code_sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('only clean committed code may run')
    args.output_root.mkdir(parents=True, exist_ok=False)
    manifest = args.artifact_root / 'source_probe_8f/rq1_source_probe_segments.jsonl'
    rows = [json.loads(line) for line in manifest.open()]
    manifest_sha = sha(manifest)
    configs = {s: yaml.safe_load((repo / SHIFT_CONFIGS[s]).read_text()) for s in shifts}
    splits = {s: partitions(rows, configs[s]) for s in shifts}
    stores, audit = {}, {}
    for name in model_names:
        stores[name], audit[name] = load_store(args.artifact_root / MODELS[name], manifest_sha, len(rows))
    write_json(args.output_root / 'preflight.json', {'code_sha': code_sha, 'features': audit,
        'manifest_sha256': manifest_sha, 'counts': {s: {k: len(v) for k, v in split.items()} for s, split in splits.items()},
        'oracle_action_boundaries': True, 'target_train_labels_hidden': True})
    if args.mode == 'preflight':
        print('RQ1_PREFLIGHT_PASS', flush=True)
        return
    torch.set_num_threads(args.threads)
    from contextlib import nullcontext
    from tools.rq3_gpu import gpu_slot, idle_gpu_uuid
    # 复用物理卡与两容量锁，绝不以NVML PID不可见为由绕过隔离。
    with nullcontext(None) if args.cpu else gpu_slot(str(args.gpu), wait=False) as slot:
        if slot is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = slot['uuid']
            if idle_gpu_uuid(slot['index']) != slot['uuid']:
                raise RuntimeError('GPU changed before CUDA creation')
        device = 'cpu' if slot is None else 'cuda:0'
        timing = {}
        for index, (name, shift) in enumerate((m, s) for m in model_names for s in shifts):
            if index % args.shards != args.shard_index:
                continue
            recipe = recipe_from_config(configs[shift], args.seed)
            if args.mode == 'benchmark':
                timing[name + '_' + shift] = benchmark(training_inputs(stores[name], rows, splits[shift], device), recipe, args.benchmark_steps)
                write_json(args.output_root / 'benchmark.json', {'measurements': timing, 'device': device,
                    'gpu': slot, 'seed': args.seed, 'not_an_experiment': True, 'recipe': asdict(recipe)})
            else:
                for regime in regimes:
                    provenance = {'model': name, 'shift': shift, 'code_sha': code_sha,
                        'feature_store': audit[name], 'manifest_sha256': manifest_sha,
                        'split_config_sha256': sha(repo / SHIFT_CONFIGS[shift]),
                        'counts': {k: len(v) for k, v in splits[shift].items()}}
                    run_item(stores[name], rows, splits[shift], recipe, regime,
                             args.output_root / (name + '_' + shift + '_' + regime), provenance, device)
    print('RQ1_' + args.mode.upper() + '_FINISHED', flush=True)


if __name__ == '__main__':
    main()
