"""固定教师实例的源侧准入；目标缓存只核身份，不读目标GT。"""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import math
import torch

from tools import baseline_protocol as p

SCHEMA = 'prototype-direct-instance-task-v1'
SOURCE_POLICY = dict(levels=[0, 1, 2], raw_threshold='reference_model_pl_threshold',
    duplicate_coordinate='highest_score_one_level', selection_cap=None,
    nms='reference_multiclass_soft_nms', after_nms_threshold='reference_model_pl_threshold',
    visibility='complete_interval_within_actual_feature_support', matching='score_greedy_same_class_one_to_one',
    ground_truth_input_to_teacher=False, teacher='reference_source_selected_best_EMA')
SOURCE_GATE = dict(precision05=.80, precision07=.50, coverage05=.10,
    aggregate='equal_weight_six_source_directions', target_effect_proof=False)


def interval_from_offsets(coordinate, stride, offsets, metadata):
    scale = float(metadata['feat_stride']); offset = .5*float(metadata['feat_num_frames'])
    fps = float(metadata['fps']); duration = float(metadata['duration'])
    return [min(duration, max(0., ((coordinate-float(offsets[0])*stride)*scale+offset)/fps)),
            min(duration, max(0., ((coordinate+float(offsets[1])*stride)*scale+offset)/fps))]


def rows_from_dense(points, masks, logits, offsets, metadata, threshold):
    # 与c82256c旧raw_pool政策一致：先阈值、只0/1/2层，再同绝对坐标择最高分。
    grouped = defaultdict(list)
    for level in SOURCE_POLICY['levels']:
        grid = points[level].detach().cpu()
        valid = masks[level][0].reshape(-1).detach().cpu().bool()
        score, classes = logits[level][0].detach().T.sigmoid().max(-1)
        score, classes = score.cpu(), classes.cpu()
        distance = offsets[level][0].detach().cpu().T
        for index in torch.nonzero(valid & (score >= threshold)).flatten().tolist():
            coordinate = float(grid[index, 0]); absolute = int(round(coordinate))
            if abs(coordinate-absolute) > 1e-6: raise ValueError('不是原FPN整数坐标')
            segment = interval_from_offsets(coordinate, float(grid[index, 3]), distance[index], metadata)
            if segment[1] <= segment[0]: continue
            grouped[absolute].append(dict(candidate_id=f"{metadata['video_id']}-p{absolute:07d}",
                video_id=metadata['video_id'], absolute_feature_index=absolute,
                interval=segment, score=float(score[index]), class_id=int(classes[index]), level=level))
    return [max(grouped[k], key=lambda x:x['score']) for k in sorted(grouped)]


@torch.no_grad()
def raw_video(model, video):
    inputs, masks = model.preprocessing([video], mode='validation')
    features, masks, _ = model.backbone_target(inputs, masks)
    features, masks = model.neck_target(features, masks)
    logits, _ = model.cls_head_target(features, masks)
    offsets, _ = model.reg_head_target(features, masks)
    if model.use_trident_head: raise ValueError('只允许原普通左右回归头')
    return rows_from_dense(model.point_generator(features), masks, logits, offsets,
        video['metadata'], model.cfg['model']['pl_threshold'])


def instances(rows, cfg):
    """秒区间直接NMS；multiclass=True时官方实现不做边界投票。"""
    if not rows: return []
    from libs.utils import batched_nms
    options = cfg['test_cfg']
    if not options['multiclass_nms'] or options['nms_method'] != 'soft':
        raise ValueError('预注册配方仅接受原multiclass soft-NMS')
    segments, scores, labels = batched_nms(torch.tensor([r['interval'] for r in rows]),
        torch.tensor([r['score'] for r in rows]), torch.tensor([r['class_id'] for r in rows]),
        options['iou_threshold'], options['min_score'], options['max_seg_num'],
        use_soft_nms=True, multiclass=True, sigma=options['nms_sigma'], voting_thresh=options['voting_thresh'])
    return [dict(video_id=rows[0]['video_id'], interval=segment, score=score, class_id=label)
        for segment, score, label in zip(segments.tolist(), scores.tolist(), labels.tolist())
        if score >= cfg['model']['pl_threshold'] and segment[1] > segment[0]]


def fully_visible(rows, metadata, feature_length):
    """与训练crop一样只用完整可见区间；不裁短后沿用原标签。"""
    start = float(metadata.get('crop_start', 0))
    end = start+float(metadata.get('crop_feat_len',feature_length))
    fps, stride = float(metadata['fps']), float(metadata['feat_stride'])
    offset = .5*float(metadata['feat_num_frames'])
    result = []
    for row in rows:
        left, right = [(float(t)*fps-offset)/stride for t in row['interval']]
        if start <= left < right <= end:
            result.append(dict(row, feature_interval=[left-start,right-start]))
    return result


def overlap(a, b):
    intersection = max(0., min(a[1], b[1])-max(a[0], b[0]))
    return intersection/max(1e-12, a[1]-a[0]+b[1]-b[0]-intersection)


def source_metrics(predictions, truth):
    """按分数的同类一对一precision，并单独统计GT被正确实例覆盖的比例。"""
    result = dict(predictions=len(predictions), ground_truth=sum(len(v) for v in truth.values()))
    for threshold in (.5, .7):
        used = set(); true_positive = 0
        for row in sorted(predictions, key=lambda x:-x['score']):
            candidates = [(overlap(row['interval'], gt['interval']), index)
                for index, gt in enumerate(truth[row['video_id']])
                if row['class_id'] == gt['class_id'] and (row['video_id'], index) not in used]
            if candidates:
                quality, index = max(candidates)
                if quality >= threshold:
                    used.add((row['video_id'], index)); true_positive += 1
        result['precision'+str(int(threshold*10)).zfill(2)] = true_positive/max(1, len(predictions))
        if threshold == .5: result['coverage05'] = true_positive/max(1,result['ground_truth'])
    return result


def source_decision(results):
    if sorted(r['shift'] for r in results) != ['S'+str(i) for i in range(1, 7)]:
        raise ValueError('必须全六方向，不按局部源结果放行')
    mean = {key:math.fsum(row['metrics'][key] for row in results)/6 for key in ('precision05','precision07','coverage05')}
    return dict(status='passed' if all(mean[key]>=SOURCE_GATE[key] for key in mean) else 'rejected',
        mean=mean, frozen_gate=SOURCE_GATE, count=6, target_data_read=False)


def worker(args):
    from libs.models.SADA import make_meta_arch
    from tools.baseline_engine import dataset, tensor_digest
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import prepare_source_video
    from libs.utils import fix_random_seed
    import os
    entry = args.output_root/'source'/args.child; entry.mkdir(parents=True, exist_ok=False)
    identity = p.code_identity(); proof = p.read(args.output_root/'inputs.json')
    if proof['code'] != identity or proof['source_policy'] != SOURCE_POLICY: raise ValueError('源准入身份漂移')
    ref = args.control/'formal/control'/args.child; cfg = p.read(ref/'config.json')
    for name, digest in proof['references'][args.child]['reference_artifacts'].items():
        if p.sha(ref/name) != digest: raise ValueError('源参考改变: '+name)
    saved = torch.load(ref/'best.pt', map_location='cpu')
    if saved['config'] != cfg or saved['epoch'] != p.termination(p.read(ref/'termination.json')['history'])['best_epoch']:
        raise ValueError('不是源选best EMA')
    os.chdir(args.data_root)
    guard = first_cuda_guard(); fix_random_seed(cfg['seed'], include_cuda=True); torch.cuda.manual_seed(cfg['seed'])
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'], strict=True); del saved
    model.eval().requires_grad_(False); before = tensor_digest(model.state_dict())
    source = dataset(cfg, cfg['dataset']['json_file'], cfg['src_val_split'], False)
    predictions = []; truths = {}; members = []; raw_count = 0
    source_manifest = Path(cfg['dataset']['json_file'])
    if p.sha(source_manifest) != cfg['baseline']['source_manifest_sha256']: raise ValueError('源manifest漂移')
    for i, raw in enumerate(source):
        video, _ = prepare_source_video(raw, 'cuda:0'); p.assert_no_target_gt([video])
        vid = video['metadata']['video_id']; meta = video['metadata']; members.append(vid)
        raw_rows = raw_video(model, video); raw_count += len(raw_rows)
        predictions.extend(fully_visible(instances(raw_rows,cfg),meta,video['feats'].shape[-1]))
        seconds = (raw['segments']*float(meta['feat_stride'])+.5*float(meta['feat_num_frames']))/float(meta['fps'])
        truths[vid] = [dict(interval=segment, class_id=int(label))
            for segment, label in zip(seconds.tolist(), raw['labels'].tolist())]
        print('SOURCE_INSTANCE_GATE', args.child, i+1, len(source), len(raw_rows), flush=True)
    if before != tensor_digest(model.state_dict()): raise ValueError('源准入改动冻结教师')
    p.save(entry/'predictions.json', predictions); p.save(entry/'membership.json', members)
    p.save(entry/'result.json', dict(status='completed_source_instance_quality_check', shift=args.child,
        source_policy=SOURCE_POLICY, metrics=source_metrics(predictions, truths), raw_candidates=raw_count,
        members=members, ground_truth_input_to_teacher=False, target_data_read=False,
        source_manifest_sha256=p.sha(source_manifest), reference=str(ref), reference_artifacts=proof['references'][args.child]['reference_artifacts'],
        model_sha256_before=before, model_sha256_after=before, code=identity, gpu_guard=guard, gpu_guard_sha256=p.sha(guard),
        artifacts={str(entry/name):p.sha(entry/name) for name in ['predictions.json','membership.json']},
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
