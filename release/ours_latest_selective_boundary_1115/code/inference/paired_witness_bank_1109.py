"""冻结源域 EMA，以真实 GT 及其等宽错位框建立配对参照，并独立核验源 VAL。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from paired_witness_core_1109 import describe, make_pairs, witness_scores


def split_names(split):
    return {s.lower() for s in ([split] if isinstance(split, str) else split)}


def source_diagnosis(bank, held):
    if set(bank['ids']) & set(held['ids']):
        raise ValueError('源留出视频与建库视频重合')
    real_pos, swap_pos, covered = witness_scores(held['pos'], held['labels'], held['levels'], bank)
    real_neg, swap_neg, neg_covered = witness_scores(held['neg'], held['labels'], held['levels'], bank)
    np.testing.assert_array_equal(covered, neg_covered)
    rates = {}
    for name, positive, negative in [('real', real_pos, real_neg), ('shuffled', swap_pos, swap_neg)]:
        diff = positive - negative
        win = (diff > 0).astype(float) + .5 * (diff == 0)
        video_win = [float(win[held['ids'] == vid].mean()) for vid in np.unique(held['ids'])]
        rates[name] = dict(video_equal_win=float(np.mean(video_win)), pair_equal_win=float(win.mean()),
                           ties=int((diff == 0).sum()), wins=int((diff > 0).sum()),
                           covered_win=float(win[covered].mean()) if np.any(covered) else None)
    return dict(videos=len(np.unique(held['ids'])), pairs=len(covered), covered=int(covered.sum()), **rates)


@torch.inference_mode()
def source_pairs(model, cfg, split, manifest, cache):
    from libs.evaluation.evaluator_anet import load_gt_seg_from_json
    from tools import baseline_engine as base

    branch = model.adapt_route
    dims = tuple(branch.descriptor_dims) if hasattr(branch, 'descriptor_dims') else (16, 64, 272)
    # 新描述、模型参数与代码各自绑定，禁止复用旧352维参照。
    fingerprint = branch.fingerprint() if hasattr(branch, 'fingerprint') else 'legacy352'
    code_sha = hashlib.sha256(Path(__file__).with_name('paired_witness_core_1109.py').read_bytes()).hexdigest()
    cache = cache / (fingerprint + '_' + code_sha[:12])
    cache.mkdir(parents=True, exist_ok=True)
    ds = base.dataset(cfg, manifest, split, False, unlabeled=True)
    truth = load_gt_seg_from_json(manifest, split, action_subsampling='10_majority_classes')
    groups = {str(name): rows for name, rows in truth.groupby('video-id')}
    outputs = []
    for index in sorted(range(len(ds.data_list)), key=lambda i: ds.data_list[i]['id']):
        row = ds.data_list[index]
        name = str(row['id'])
        if name not in groups:
            continue
        rows = groups[name]
        boxes = rows[['t-start', 't-end']].to_numpy(float)
        labels = rows.label.to_numpy(int)
        valid = np.isfinite(boxes).all(1) & (boxes[:, 1] > boxes[:, 0])
        good, bad, labels = make_pairs(boxes[valid], labels[valid], float(row['duration']))
        if not len(labels):
            continue
        arrays_path = cache / (hashlib.sha256(name.encode()).hexdigest() + '.npz')
        if not arrays_path.exists():
            video = ds[index]
            assert video['segments'].numel() == video['labels'].numel() == 0
            data, masks = model.preprocessing([video], mode='test')
            maps, masks, _ = model.backbone_source(data, masks)
            maps, masks = model.neck_source(maps, masks)
            logits, _ = model.cls_head_pl(maps, masks)
            pos, levels = describe(model, maps, masks, logits, video['metadata'], good, labels)
            neg, bad_levels = describe(model, maps, masks, logits, video['metadata'], bad, labels, force_levels=levels)
            np.testing.assert_array_equal(levels, bad_levels)
            np.testing.assert_allclose(np.diff(good, axis=1), np.diff(bad, axis=1), rtol=1e-5, atol=1e-5)
            temp = arrays_path.with_suffix('.tmp')
            with temp.open('wb') as f:
                np.savez(f, pos=pos, neg=neg, labels=labels, levels=levels, ids=np.asarray([name]*len(labels)))
            temp.replace(arrays_path)
        with np.load(arrays_path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f['labels'], labels)
            if f['pos'].shape != (len(labels), sum(dims)) or f['neg'].shape != f['pos'].shape:
                raise ValueError('源配对缓存维度不匹配')
            outputs.append({k: f[k] for k in ('pos', 'neg', 'labels', 'levels', 'ids')})
    if not outputs:
        raise ValueError(f'源视频没有符合条件的配对 {split}')
    bank = {k: np.concatenate([output[k] for output in outputs]) for k in outputs[0]}
    bank['descriptor_dims'] = dims
    bank['descriptor_fingerprint'] = fingerprint
    return bank


def build_bank(spec, shift, model, cfg, checkpoint_sha=None, cache_dir=None):
    """模型与引用的 EMA SHA 由上层装载核验；本层按脚本、标注和权重 SHA 隔离缓存。"""
    from tools.baseline_protocol import read, sha
    import paired_witness_core_1109 as core

    audit = read(Path(spec['training_root']) / shift / 'state_audit.json')
    if audit['status'] != 'passed' or (checkpoint_sha and checkpoint_sha != audit['best_sha256']):
        raise ValueError('原生检查点审计或 SHA 不匹配')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('源参照需使用冻结的 eval 模型')
    train_manifest = Path(cfg['dataset']['json_file'])
    val_manifest = Path(cfg['baseline']['annotation_for_offline_evaluation'])
    train = cfg['src_train_split']
    val = cfg['src_val_split']
    other_splits = split_names(cfg['tgt_train_split']) | split_names(cfg['tgt_val_split']) | split_names(val)
    if split_names(train) & other_splits or train_manifest == Path(cfg['baseline']['target_manifest']):
        raise ValueError('源TRAIN参照使用了错误的数据划分')
    binding = dict(shift=shift, checkpoint=audit['best_sha256'], train=sorted(split_names(train)),
                   val=sorted(split_names(val)), train_manifest=sha(train_manifest), val_manifest=sha(val_manifest),
                   core=sha(Path(core.__file__)), extractor=sha(Path(__file__)))
    digest = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    parent = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / 'banks'
    path = parent / shift / digest
    bank = source_pairs(model, cfg, train, train_manifest, path / 'train')
    held = source_pairs(model, cfg, val, val_manifest, path / 'val')
    summary = dict(binding=binding, train_videos=len(np.unique(bank['ids'])), train_pairs=len(bank['labels']),
                   train_class_levels=len(set(zip(bank['labels'], bank['levels']))), source_val=source_diagnosis(bank, held))
    return bank, summary
