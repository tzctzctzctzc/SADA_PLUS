"""Target-GT audit for independent-teacher boundary consensus.

The script never feeds target annotations into a teacher.  It first obtains
the dense predictions from independently trained EMA teachers, then uses GT
only to measure whether their boundary consensus is a better Stage 2 target.
"""

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libs.datasets import make_dataset
from libs.models.SADA import make_meta_arch
from libs.models.SADA.stage2_utils import (
    aligned_boundary_iou_from_offsets,
    top_fraction_per_sample_mask,
)
from libs.utils.utils import _update_config, read_yaml
from tools.audit_original_sada_pseudolabels import crop_targets, move_video


def parse_numbers(value, cast):
    return [cast(item.strip()) for item in value.split(',') if item.strip()]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def rank(values):
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = torch.arange(
        values.numel(), device=values.device, dtype=torch.float64
    )
    return ranks


def pearson(first, second):
    if first.numel() < 2:
        return None
    first = first.double() - first.double().mean()
    second = second.double() - second.double().mean()
    denominator = first.square().mean().sqrt() * second.square().mean().sqrt()
    if denominator == 0:
        return None
    return float((first * second).mean() / denominator)


def summarize(records, indices, baseline_foreground):
    chosen = [records[index] for index in indices]
    foreground = [record for record in chosen if record['foreground']]
    good = [
        record for record in chosen
        if record['foreground'] and record['class_correct']
        and record['consensus_iou'] >= 0.5
    ]
    result = {
        'n': len(chosen),
        'foreground_precision': ratio(len(foreground), len(chosen)),
        'class_correct_precision': ratio(
            sum(record['class_correct'] for record in chosen), len(chosen)
        ),
        'mean_primary_iou_all': ratio(
            sum(record['primary_iou'] for record in chosen), len(chosen)
        ),
        'mean_consensus_iou_all': ratio(
            sum(record['consensus_iou'] for record in chosen), len(chosen)
        ),
        'mean_iou_delta_all': ratio(
            sum(record['consensus_iou'] - record['primary_iou'] for record in chosen),
            len(chosen),
        ),
        'mean_primary_iou_foreground': ratio(
            sum(record['primary_iou'] for record in foreground), len(foreground)
        ),
        'mean_consensus_iou_foreground': ratio(
            sum(record['consensus_iou'] for record in foreground), len(foreground)
        ),
        'mean_iou_delta_foreground': ratio(
            sum(record['consensus_iou'] - record['primary_iou'] for record in foreground),
            len(foreground),
        ),
        'good_anchor_precision': ratio(len(good), len(chosen)),
        'good_anchor_coverage': ratio(len(good), baseline_foreground),
        'agreement_mean': ratio(
            sum(record['agreement'] for record in chosen), len(chosen)
        ),
    }
    for threshold in (0.3, 0.5, 0.7):
        result['primary_iou_ge_{:.1f}'.format(threshold)] = ratio(
            sum(record['primary_iou'] >= threshold for record in foreground),
            len(foreground),
        )
        result['consensus_iou_ge_{:.1f}'.format(threshold)] = ratio(
            sum(record['consensus_iou'] >= threshold for record in foreground),
            len(foreground),
        )
    return result


def forward_teacher(model, inputs, masks):
    backbone_features, backbone_masks, _ = model.backbone_target(inputs, masks)
    fpn_features, fpn_masks = model.neck_target(
        backbone_features, backbone_masks
    )
    logits, _ = model.cls_head_target(fpn_features, fpn_masks)
    offsets, _ = model.reg_head_target(fpn_features, fpn_masks)
    return (
        torch.cat([value.permute(0, 2, 1) for value in logits], dim=1),
        torch.cat([value.permute(0, 2, 1) for value in offsets], dim=1),
        torch.cat([value.squeeze(1) for value in fpn_masks], dim=1),
        model.point_generator(fpn_features),
    )


def load_teacher(cfg, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'ema_state_dict' not in checkpoint:
        raise ValueError('{} has no ema_state_dict'.format(checkpoint_path))
    model = make_meta_arch(cfg['model_name'], cfg, device=device).to(device)
    model_state = model.state_dict()
    compatible_state = {
        key: value for key, value in checkpoint['ema_state_dict'].items()
        if key in model_state and model_state[key].shape == value.shape
    }
    result = model.load_state_dict(compatible_state, strict=False)
    invalid_missing = [
        key for key in result.missing_keys
        if not key.startswith(('disc_dom_adv.', 'disc_dom_conditional_adv.'))
    ]
    if invalid_missing:
        raise RuntimeError(
            '{} is incompatible outside training-only discriminators: {}'.format(
                checkpoint_path, invalid_missing
            )
        )
    model.eval()
    return model, checkpoint.get('epoch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--levels', default='0,1,2')
    parser.add_argument('--retain-fractions', default='1.0,0.75,0.5,0.25')
    parser.add_argument('--verify-crop-targets', action='store_true')
    args = parser.parse_args()

    if len(args.checkpoints) < 2:
        raise ValueError('independent consensus requires at least two teachers')
    levels = parse_numbers(args.levels, int)
    fractions = parse_numbers(args.retain_fractions, float)
    if (
        not levels or len(set(levels)) != len(levels)
        or any(level < 0 or level > 5 for level in levels)
    ):
        raise ValueError('levels must be unique FPN levels from 0 to 5')
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError('retain fractions must be in (0, 1]')

    cfg = read_yaml(args.config)
    cfg['stage1'] = {'enabled': False, 'perturbation': 'shift'}
    cfg['stage2'] = {'enabled': False, 'routing': 'confidence'}
    _update_config(cfg)
    teachers, epochs = zip(*[
        load_teacher(cfg, path, args.device) for path in args.checkpoints
    ])
    dataset = make_dataset(
        cfg['dataset_name'], True, cfg['tgt_train_split'], **cfg['dataset']
    )
    annotations = json.loads(
        Path(cfg['dataset']['json_file']).read_text(encoding='utf-8')
    )['database']
    batch_size = args.batch_size or int(cfg.get('loader', {}).get('batch_size', 1))
    alpha = float(cfg['model'].get('pl_threshold', 0.7))
    records = []
    videos_checked = 0

    with torch.no_grad():
        for batch_start in range(0, len(dataset), batch_size):
            raw_videos = [
                dataset[index]
                for index in range(
                    batch_start, min(batch_start + batch_size, len(dataset))
                )
            ]
            videos = [move_video(video, args.device) for video in raw_videos]
            inputs, masks = teachers[0].preprocessing(videos, mode='eval')

            # Target GT is intentionally read only after all teacher forwards.
            outputs = [forward_teacher(model, inputs, masks) for model in teachers]
            logits = torch.stack([output[0] for output in outputs])
            offsets = torch.stack([output[1] for output in outputs])
            valid_mask = outputs[0][2]
            points = outputs[0][3]

            gt_segments, gt_labels = [], []
            for raw_video, video in zip(raw_videos, videos):
                video_id = raw_video['metadata']['video_id']
                segments, labels = crop_targets(
                    annotations[video_id], video['metadata'],
                    video['feats'].shape[-1], cfg, args.device,
                )
                if args.verify_crop_targets and (
                    segments.shape != raw_video['segments'].shape
                    or not torch.equal(
                        segments, raw_video['segments'].to(args.device)
                    )
                    or not torch.equal(
                        labels, raw_video['labels'].to(args.device)
                    )
                ):
                    raise RuntimeError('reconstructed target GT differs for {}'.format(video_id))
                gt_segments.append(segments)
                gt_labels.append(labels)
                videos_checked += 1
            gt_cls, gt_offsets = teachers[0].label_points(
                points, gt_segments, gt_labels
            )
            gt_cls = torch.stack(gt_cls)
            gt_offsets = torch.stack(gt_offsets)

            probabilities = logits.sigmoid()
            scores, classes = probabilities.max(dim=-1)
            primary_scores = scores[0]
            primary_classes = classes[0]
            same_class = (classes == primary_classes.unsqueeze(0)).all(dim=0)
            consensus_offsets = offsets.median(dim=0).values
            agreement = torch.stack([
                aligned_boundary_iou_from_offsets(offsets[first], offsets[second])
                for first, second in itertools.combinations(range(len(teachers)), 2)
            ]).mean(dim=0)
            primary_iou = aligned_boundary_iou_from_offsets(
                offsets[0], gt_offsets
            )
            consensus_iou = aligned_boundary_iou_from_offsets(
                consensus_offsets, gt_offsets
            )
            action = gt_cls.sum(dim=-1) > 0
            correct_class = gt_cls.gather(
                -1, primary_classes.unsqueeze(-1)
            ).squeeze(-1) > 0

            level_ids = torch.empty(
                valid_mask.shape[1], dtype=torch.long, device=args.device
            )
            start = 0
            for level, point_grid in enumerate(points):
                end = start + point_grid.shape[0]
                level_ids[start:end] = level
                start = end
            eligible = (
                valid_mask
                & (primary_scores >= alpha)
                & torch.isin(level_ids, torch.tensor(levels, device=args.device))
            )
            for sample_index, raw_video in enumerate(raw_videos):
                selected = torch.where(eligible[sample_index])[0]
                for anchor_index in selected.tolist():
                    records.append({
                        'video_id': raw_video['metadata']['video_id'],
                        'crop_start': int(
                            raw_video['metadata'].get('crop_start', 0)
                        ),
                        'level': int(level_ids[anchor_index]),
                        'foreground': bool(action[sample_index, anchor_index]),
                        'class_correct': bool(correct_class[sample_index, anchor_index]),
                        'class_agreement': bool(same_class[sample_index, anchor_index]),
                        'primary_iou': float(primary_iou[sample_index, anchor_index]),
                        'consensus_iou': float(consensus_iou[sample_index, anchor_index]),
                        'agreement': float(agreement[sample_index, anchor_index]),
                    })

    if not records:
        raise RuntimeError('the primary teacher selected no target anchors')
    baseline_foreground = sum(record['foreground'] for record in records)
    output = {
        'audit': 'independent_teacher_same_anchor_boundary_consensus_v1',
        'checkpoints': args.checkpoints,
        'checkpoint_epochs': list(epochs),
        'videos': videos_checked,
        'method': {
            'primary_candidate_rule': 'teacher0 valid and score >= alpha',
            'consensus_offset': 'componentwise median of teacher offsets',
            'agreement': 'mean pairwise same-anchor offset IoU',
            'target_gt_usage': 'post-prediction offline metrics only',
            'levels': levels,
            'pl_threshold': alpha,
        },
        'baseline': summarize(records, list(range(len(records))), baseline_foreground),
        'class_agreement': summarize(
            records,
            [index for index, record in enumerate(records) if record['class_agreement']],
            baseline_foreground,
        ),
        'top_agreement_per_crop': {},
        'agreement_correlation': {
            'pearson_all': pearson(
                torch.tensor([record['agreement'] for record in records]),
                torch.tensor([record['consensus_iou'] for record in records]),
            ),
            'spearman_all': pearson(
                rank(torch.tensor([record['agreement'] for record in records])),
                rank(torch.tensor([record['consensus_iou'] for record in records])),
            ),
        },
    }
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[(record['video_id'], record['crop_start'])].append(index)
    for fraction in fractions:
        chosen = []
        for indices in grouped.values():
            scores = torch.tensor([records[index]['agreement'] for index in indices])
            mask = torch.ones_like(scores, dtype=torch.bool).unsqueeze(0)
            local = top_fraction_per_sample_mask(
                scores.unsqueeze(0), mask, fraction
            ).squeeze(0)
            chosen.extend(indices[index] for index in torch.where(local)[0].tolist())
        output['top_agreement_per_crop']['{:.2f}'.format(fraction)] = summarize(
            records, chosen, baseline_foreground
        )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output['baseline'], indent=2))
    print(json.dumps(output['top_agreement_per_crop'], indent=2))
    print('wrote', args.output)


if __name__ == '__main__':
    main()
