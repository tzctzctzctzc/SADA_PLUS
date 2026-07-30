"""Audit a source-calibrated semantic residual refiner for Stage 2 offsets."""

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def boundary_features(probabilities, offsets, point_grid, level):
    grid, stride = point_grid[:, 0], point_grid[:, 3]
    confidence, predicted_class = probabilities.max(dim=-1)
    starts = grid[None, :] - offsets[:, 0] * stride[None, :]
    ends = grid[None, :] + offsets[:, 1] * stride[None, :]
    left = torch.searchsorted(
        grid.contiguous(), (starts - stride[None, :]).contiguous()
    ).clamp(0, grid.numel() - 1)
    right = torch.searchsorted(
        grid.contiguous(), (ends + stride[None, :]).contiguous()
    ).clamp(0, grid.numel() - 1)
    classes = predicted_class.unsqueeze(-1)
    left_prob = probabilities.gather(
        1, left.unsqueeze(-1).expand_as(probabilities)
    ).gather(-1, classes).squeeze(-1)
    right_prob = probabilities.gather(
        1, right.unsqueeze(-1).expand_as(probabilities)
    ).gather(-1, classes).squeeze(-1)
    contrast = (confidence - 0.5 * (left_prob + right_prob)).clamp_min(0.0)
    log_offsets = torch.log1p(offsets.permute(0, 2, 1))
    common = torch.stack((
        confidence, left_prob, right_prob, contrast,
        left_prob - right_prob,
    ), dim=-1)
    features = torch.cat((
        log_offsets,
        common,
        log_offsets * contrast.unsqueeze(-1),
        log_offsets * confidence.unsqueeze(-1),
        torch.full_like(confidence.unsqueeze(-1), float(level)),
    ), dim=-1)
    return confidence, predicted_class, features


def model_outputs(model, inputs, masks, domain):
    if domain == 'source':
        backbone, neck, cls_head = (
            model.backbone_source, model.neck_source, model.cls_head,
        )
    else:
        backbone, neck, cls_head = (
            model.backbone_target, model.neck_target, model.cls_head_target,
        )
    features, feature_masks, _ = backbone(inputs, masks)
    fpn_features, fpn_masks = neck(features, feature_masks)
    logits, _ = cls_head(fpn_features, fpn_masks)
    teacher_features, teacher_masks, _ = model.backbone_pl(inputs, masks)
    teacher_fpn, teacher_fpn_masks = model.neck_pl(
        teacher_features, teacher_masks,
    )
    offsets, _ = model.reg_head_pl(teacher_fpn, teacher_fpn_masks)
    return logits, offsets, teacher_fpn, teacher_fpn_masks


def collect_records(model, dataset, annotations, cfg, domain, split, levels, device):
    records = {key: [] for key in ('features', 'offsets', 'gt_offsets')}
    alpha = float(cfg['model'].get('pl_threshold', 0.7))
    batch_size = int(cfg.get('loader', {}).get('batch_size', 1))
    mode = 'train' if split == 'train' else 'validation'
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            raw_batch = [
                dataset[index]
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            batch = [move_video(video, device) for video in raw_batch]
            inputs, masks = model.preprocessing(batch, mode=mode)
            logits, offsets, teacher_fpn, teacher_masks = model_outputs(
                model, inputs, masks, domain,
            )
            points = model.point_generator(teacher_fpn)
            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_batch, batch):
                segments, labels = crop_targets(
                    annotations[raw_video['metadata']['video_id']],
                    video['metadata'], video['feats'].shape[-1], cfg, device,
                )
                gt_segments.append(segments)
                gt_labels.append(labels)
            gt_cls, gt_offsets = model.label_points(points, gt_segments, gt_labels)
            gt_cls, gt_offsets = torch.stack(gt_cls), torch.stack(gt_offsets)
            offset = 0
            for level, point in enumerate(points):
                next_offset = offset + point.shape[0]
                if level not in levels:
                    offset = next_offset
                    continue
                probabilities = logits[level].permute(0, 2, 1).sigmoid()
                confidence, predicted_class, features = boundary_features(
                    probabilities, offsets[level], point, level,
                )
                selected = (
                    teacher_masks[level].squeeze(1) & (confidence >= alpha)
                )
                if domain == 'source':
                    labels = gt_cls[:, offset:next_offset]
                    selected &= labels.sum(dim=-1) > 0
                    selected &= labels.gather(
                        -1, predicted_class.unsqueeze(-1),
                    ).squeeze(-1) > 0
                if selected.any():
                    records['features'].append(features[selected].cpu())
                    records['offsets'].append(
                        offsets[level].permute(0, 2, 1)[selected].cpu()
                    )
                    records['gt_offsets'].append(
                        gt_offsets[:, offset:next_offset][selected].cpu()
                    )
                offset = next_offset
    if not records['offsets']:
        raise RuntimeError('No selected anchors')
    return {key: torch.cat(values) for key, values in records.items()}


def fit_ridge(features, targets, l2, device):
    features = features.to(device, dtype=torch.float64)
    targets = targets.to(device, dtype=torch.float64)
    mean = features.mean(dim=0)
    std = features.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (features - mean) / std
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    identity = torch.eye(augmented.shape[1], device=device, dtype=torch.float64)
    identity[-1, -1] = 0.0
    weight = torch.linalg.solve(
        augmented.T.matmul(augmented) + l2 * identity,
        augmented.T.matmul(targets),
    )
    return {'mean': mean.cpu(), 'std': std.cpu(), 'weight': weight.cpu()}


def corrected_offsets(features, offsets, calibration, gamma, device):
    features = features.to(device, dtype=torch.float64)
    offsets = offsets.to(device, dtype=torch.float64)
    normalized = (
        features - calibration['mean'].to(device)
    ) / calibration['std'].to(device)
    augmented = torch.cat((normalized, torch.ones_like(normalized[:, :1])), dim=1)
    residual = augmented.matmul(calibration['weight'].to(device))
    return torch.expm1(
        torch.log1p(offsets) + gamma * residual
    ).clamp_min(0.0).float()


def metrics(offsets, targets):
    return {
        'anchors': int(offsets.shape[0]),
        'offset_mse': float((offsets - targets).square().mean()),
    }


def load_model(args):
    cfg = read_yaml(args.config)
    cfg['stage1'] = {'enabled': False, 'perturbation': 'shift'}
    _update_config(cfg)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = make_meta_arch(cfg['model_name'], cfg, device=args.device).to(args.device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    annotations = json.loads(
        Path(cfg['dataset']['json_file']).read_text(encoding='utf-8')
    )['database']
    return cfg, checkpoint, model, annotations


def source_select(args):
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    levels = [int(value) for value in args.levels.split(',')]
    train = make_dataset(
        cfg['dataset_name'], True, cfg['src_train_split'], **cfg['dataset']
    )
    val = make_dataset(
        cfg['dataset_name'], False, cfg['src_val_split'], **cfg['dataset']
    )
    train_records = collect_records(
        model, train, annotations, cfg, 'source', 'train', levels, args.device,
    )
    val_records = collect_records(
        model, val, annotations, cfg, 'source', 'val', levels, args.device,
    )
    target_residual = (
        torch.log1p(train_records['gt_offsets'])
        - torch.log1p(train_records['offsets'])
    )
    candidates, best = [], None
    for l2 in parse_numbers(args.ridge_l2):
        calibration = fit_ridge(
            train_records['features'], target_residual, l2, args.device,
        )
        for gamma in parse_numbers(args.correction_scales):
            corrected = corrected_offsets(
                val_records['features'], val_records['offsets'], calibration,
                gamma, args.device,
            ).cpu()
            candidate = {
                'l2': l2, 'gamma': gamma,
                'metrics': metrics(corrected, val_records['gt_offsets']),
            }
            candidates.append(candidate)
            if best is None or (
                candidate['metrics']['offset_mse'] < best['metrics']['offset_mse']
            ):
                best, best_calibration = candidate, calibration
    artifact = {
        'calibration': best_calibration, 'l2': best['l2'],
        'gamma': best['gamma'], 'levels': levels, 'config': args.config,
        'checkpoint': args.checkpoint,
    }
    torch.save(artifact, args.calibration_output)
    return {
        'audit': 'source_only_semantic_boundary_log_offset_refiner_selection',
        'checkpoint_epoch': checkpoint.get('epoch'),
        'source_train_anchors': int(train_records['offsets'].shape[0]),
        'source_val_baseline': metrics(
            val_records['offsets'], val_records['gt_offsets'],
        ),
        'selected': best, 'candidates': candidates,
        'selection_metric': 'source_val_offset_mse',
        'calibration_output': args.calibration_output,
        'target_gt_usage': 'none',
    }


def target_eval(args):
    artifact = torch.load(args.calibration, map_location='cpu')
    seed_everything(args.seed)
    cfg, checkpoint, model, annotations = load_model(args)
    val = make_dataset(
        cfg['dataset_name'], False, cfg['tgt_val_split'], **cfg['dataset']
    )
    records = collect_records(
        model, val, annotations, cfg, 'target', 'val', artifact['levels'],
        args.device,
    )
    corrected = corrected_offsets(
        records['features'], records['offsets'], artifact['calibration'],
        artifact['gamma'], args.device,
    ).cpu()
    return {
        'audit': 'fixed_source_only_semantic_boundary_log_offset_refiner_target_evaluation',
        'checkpoint_epoch': checkpoint.get('epoch'),
        'fixed_source_selection': {
            key: artifact[key] for key in ('l2', 'gamma', 'levels', 'config', 'checkpoint')
        },
        'target_baseline': metrics(records['offsets'], records['gt_offsets']),
        'target_refined': metrics(corrected, records['gt_offsets']),
        'target_gt_usage': 'offline evaluation only',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('source_select', 'target_eval'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--calibration-output')
    parser.add_argument('--calibration')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--ridge-l2', default='0.1,1,10,100')
    parser.add_argument('--correction-scales', default='0.25,0.5,0.75,1')
    parser.add_argument('--seed', type=int, default=20260731)
    args = parser.parse_args()
    if args.mode == 'source_select' and not args.calibration_output:
        parser.error('--calibration-output is required for source_select')
    if args.mode == 'target_eval' and not args.calibration:
        parser.error('--calibration is required for target_eval')
    result = source_select(args) if args.mode == 'source_select' else target_eval(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
