"""冻结原模型的源侧语义纠错诊断；不训练student，不读取目标数据。"""
from copy import deepcopy
from pathlib import Path
import math
import numpy as np
import torch
from tools import baseline_protocol as p

SCHEMA = 'prototype-detector-complementarity-source-v1'
SPEC = dict(population='original_levels012_threshold_deduplicated_raw_fully_visible_no_cap',
    detector='original_proposal_generating_point_and_level', prototype='existing_source_CE_classifier',
    eligibility='any_source_GT_tIoU_at_least_0.5_accept_all_overlapping_classes',
    calibration='source_train_only_five_fixed_margin_bins_Laplace_correctness',
    margin_bins=[0., .2, .4, .6, .8, 1.], probability_advantage=.10,
    gate='different_top1_and_calibrated_prototype_correctness_minus_detector_at_least_0.10',
    uncertainty_claim='empirical_branch_correctness_not_calibrated_target_uncertainty',
    seed=123, bootstrap_replicates=2000, source_val_previously_used_for_checkpoint_selection=True)
GATE = dict(minimum_mean_net_correction_pp=1., minimum_mean_eligible_coverage=.05,
    minimum_mean_selected_precision=.70, minimum_clustered_bootstrap_lower_net_pp=0., target_effect_proof=False)


def foreground_distribution(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('类别分数必须是有限二维矩阵')
    return scores / np.maximum(scores.sum(1, keepdims=True), 1e-12)


def prediction_margin(probabilities):
    probabilities = np.asarray(probabilities)
    ordered = np.sort(probabilities, axis=1)
    return probabilities.argmax(1), ordered[:, -1] - ordered[:, -2]


def matching_labels(segments, labels, interval):
    """GT只在两路推理完成后使用，不按模型预测类别决定匹配GT。"""
    if len(segments) == 0: return [], 0.
    segments = np.asarray(segments, dtype=np.float64)
    left, right = map(float, interval)
    intersection = np.maximum(0., np.minimum(segments[:, 1], right)-np.maximum(segments[:, 0], left))
    quality = intersection / np.maximum(1e-12, segments[:, 1]-segments[:, 0]+right-left-intersection)
    return sorted(set(int(x) for x in np.asarray(labels)[quality >= .5])), float(quality.max())


def bin_index(margin):
    return int(np.clip(np.searchsorted(SPEC['margin_bins'], margin, side='right')-1, 0, 4))


def correctness(row, branch):
    return int(row[branch+'_class'] in row['acceptable_classes'])


def calibrate(rows):
    result = {}
    for branch in ('detector', 'prototype'):
        counts = [[0, 0] for _ in range(5)]
        for row in rows:
            # 未匹配候选也计错，避免把条件于GT前景的正确率当可部署可靠性。
            cell = counts[bin_index(row[branch+'_margin'])]
            cell[0] += 1; cell[1] += correctness(row, branch)
        result[branch] = [dict(count=n, correct=k, probability=(k+1)/(n+2)) for n, k in counts]
    return result


def selected(row, calibration):
    if row['detector_class'] == row['prototype_class']: return False
    # 无源样本的bin拒绝提供纠错监督，而非用先验0.5假装可信。
    a = calibration['detector'][bin_index(row['detector_margin'])]
    b = calibration['prototype'][bin_index(row['prototype_margin'])]
    return a['count'] > 0 and b['count'] > 0 and b['probability']-a['probability'] >= SPEC['probability_advantage']


def summarize(rows, calibration):
    eligible = [r for r in rows if r['acceptable_classes']]
    out = dict(candidates=len(rows), eligible=len(eligible), unmatched_or_low_overlap=len(rows)-len(eligible),
        detector_correct=0, prototype_correct=0, both_correct=0, both_wrong=0,
        oracle_corrections=0, reverse_harm=0, disagreements=0, selected=0, corrections=0, harms=0,
        selected_unmatched_or_low_overlap=sum(selected(r, calibration) for r in rows if not r['acceptable_classes']))
    for row in eligible:
        d, b = correctness(row, 'detector'), correctness(row, 'prototype')
        out['detector_correct'] += d; out['prototype_correct'] += b
        out['both_correct'] += d*b; out['both_wrong'] += (1-d)*(1-b)
        out['oracle_corrections'] += (1-d)*b; out['reverse_harm'] += d*(1-b)
        out['disagreements'] += row['detector_class'] != row['prototype_class']
        if selected(row, calibration):
            out['selected'] += 1; out['corrections'] += (1-d)*b; out['harms'] += d*(1-b)
    n = max(1, out['eligible'])
    out.update(net_correction_pp=100*(out['corrections']-out['harms'])/n,
        coverage=out['selected']/n, oracle_headroom_pp=100*out['oracle_corrections']/n,
        selected_precision=sum(correctness(r, 'prototype') for r in rows if selected(r, calibration))/max(1, out['selected']+out['selected_unmatched_or_low_overlap']))
    return out


def source_decision(results):
    if sorted(r['shift'] for r in results) != ['S'+str(i) for i in range(1, 7)]:
        raise ValueError('必须精确六方向')
    mean = math.fsum(r['metrics']['net_correction_pp'] for r in results)/6
    coverage = math.fsum(r['metrics']['coverage'] for r in results)/6
    precision = math.fsum(r['metrics']['selected_precision'] for r in results)/6
    # 视频为聚类单位，六方向等权；这是源侧探索区间，不是目标收益证明。
    rng = np.random.default_rng(SPEC['seed']); draws = []
    for _ in range(SPEC['bootstrap_replicates']):
        values = []
        for result in results:
            groups = result['per_video']; sampled = rng.integers(len(groups), size=len(groups))
            n = sum(groups[i]['eligible'] for i in sampled)
            gain = sum(groups[i]['corrections']-groups[i]['harms'] for i in sampled)
            values.append(100*gain/max(1, n))
        draws.append(math.fsum(values)/6)
    low, high = np.quantile(draws, [.025, .975]).tolist()
    valid = all(r['metrics']['eligible'] > 0 and r['train_eligible'] > 0 for r in results)
    passed = valid and mean >= GATE['minimum_mean_net_correction_pp'] and coverage >= GATE['minimum_mean_eligible_coverage'] and precision >= GATE['minimum_mean_selected_precision'] and low > GATE['minimum_clustered_bootstrap_lower_net_pp']
    return dict(status='passed' if passed else 'rejected', count=6, mean_net_correction_pp=mean,
        mean_coverage=coverage, mean_selected_precision=precision, clustered_bootstrap_net_pp=[low, high], denominators_positive=valid,
        frozen_gate=GATE, target_data_read=False, not_target_mAP=True)


@torch.no_grad()
def observe(model, video):
    from tools.instance_task_source import rows_from_dense, fully_visible
    x, mask = model.preprocessing([video], mode='test'); x = model.dropout(x)
    feats, masks, _ = model.backbone_source(x, mask); fpn, masks = model.neck_source(feats, masks)
    points = model.point_generator(fpn); logits, _ = model.cls_head(fpn, masks); offsets, _ = model.reg_head(fpn, masks)
    rows = rows_from_dense(points, masks, logits, offsets, video['metadata'], model.cfg['model']['pl_threshold'])
    rows = fully_visible(rows, video['metadata'], video['feats'].shape[-1])
    branch = model.rq3_capacity.branch
    score_arrays = [v[0].T.sigmoid().cpu().numpy() for v in logits]
    point_indices = [{float(c):i for i,c in enumerate(grid[:,0].cpu().tolist())} for grid in points]
    for begin in range(0, len(rows), 128):
        group = rows[begin:begin+128]
        proposal = dict(segments=fpn[0].new_tensor([r['feature_interval'] for r in group]),
            levels=torch.tensor([r['level'] for r in group], device=fpn[0].device))
        encoded = branch.encode(fpn, [proposal], model.rq3_capacity.strides)
        probabilities = encoded['logits'].softmax(-1).cpu().numpy()
        classes, margins = prediction_margin(probabilities)
        for j, row in enumerate(group):
            level = row['level']; idx = point_indices[level].get(float(row['absolute_feature_index']))
            if idx is None: raise ValueError('候选生成坐标无法精确回溯')
            scores = score_arrays[level][idx]
            dc, dm = prediction_margin(foreground_distribution(scores[None]))
            if int(dc[0]) != row['class_id']: raise ValueError('原候选类别与检测头不一致')
            row.update(detector_class=int(dc[0]), detector_margin=float(dm[0]), prototype_class=int(classes[j]),
                prototype_margin=float(margins[j]), detector_scores=scores.tolist(), prototype_probabilities=probabilities[j].tolist())
    return rows, (points, masks, logits, offsets)


def worker(args):
    import os
    from libs.models.SADA import make_meta_arch
    from libs.utils import fix_random_seed
    from tools.baseline_engine import dataset, tensor_digest
    from tools.baseline_gpu import first_cuda_guard
    from tools.calibrate_prototype_geometry import prepare_source_video
    entry = args.output_root/'source'/args.child; entry.mkdir(parents=True, exist_ok=False)
    identity = p.code_identity(); proof = p.read(args.output_root/'inputs.json'); reference = args.control/'formal/control'/args.child
    if proof['code'] != identity or proof['specification'] != SPEC: raise ValueError('诊断身份漂移')
    source_proof = proof['sources'][args.child]; cfg = p.read(reference/'config.json')
    for name, digest in source_proof['reference_artifacts'].items():
        if p.sha(reference/name) != digest: raise ValueError('冻结参考改变')
    saved = torch.load(reference/'best.pt', map_location='cpu')
    if saved['config'] != cfg or saved['epoch'] != p.termination(p.read(reference/'termination.json')['history'])['best_epoch']:
        raise ValueError('不是原源选best EMA')
    os.chdir(args.data_root); guard = first_cuda_guard(); fix_random_seed(cfg['seed'], include_cuda=True); torch.cuda.manual_seed(cfg['seed'])
    model = make_meta_arch('sada', deepcopy(cfg), device='cuda:0').to('cuda:0')
    model.load_state_dict(saved['ema_state_dict'], strict=True); del saved
    model.eval().requires_grad_(False); before = tensor_digest(model.state_dict())
    rng_cpu = torch.get_rng_state().clone(); rng_cuda = torch.cuda.get_rng_state(0).clone()
    all_rows = {}; members = {}; features = {}; truths = {}; null_checked = False
    for split, subsets in [('train', cfg['src_train_split']), ('validation', cfg['src_val_split'])]:
        ds = dataset(cfg, cfg['dataset']['json_file'], subsets, False); members[split] = []; all_rows[split] = []
        if [v['id'] for v in ds.data_list] != source_proof['membership'][split]: raise ValueError('源成员改变')
        for i, raw in enumerate(ds):
            video, _ = prepare_source_video(raw, 'cuda:0'); p.assert_no_target_gt([video]); vid = video['metadata']['video_id']
            rows, dense = observe(model, video)
            truths[vid] = dict(segments=raw['segments'].tolist(), labels=raw['labels'].tolist())
            if not null_checked:
                points, masks, logits, offsets = dense
                with torch.no_grad():
                    actual = model.inference([video], points, [m.squeeze(1) for m in masks], [v.permute(0,2,1) for v in logits], [v.permute(0,2,1) for v in offsets], None, None)[0]
                    full = model(video_list_src=[video], video_list_tgt=[video], mode='test')[0][0]
                for key in ('segments', 'scores', 'labels'): torch.testing.assert_close(actual[key], full[key], rtol=1e-6, atol=1e-7)
                null_checked = True
            for row in rows:
                labels, quality = matching_labels(raw['segments'].numpy(), raw['labels'].numpy(), row['feature_interval'])
                row.update(acceptable_classes=labels, maximum_tIoU=quality)
            all_rows[split].extend(rows); members[split].append(vid)
            feature = Path(source_proof['physical_features'][split][vid]); features[str(feature)] = p.sha(feature)
            print('SOURCE_COMPLEMENTARITY', args.child, split, i+1, len(ds), len(rows), flush=True)
    calibration = calibrate(all_rows['train']); validation = all_rows['validation']
    per_video = [dict(video_id=vid, **summarize([r for r in validation if r['video_id'] == vid], calibration)) for vid in members['validation']]
    if before != tensor_digest(model.state_dict()): raise ValueError('诊断修改冻结模型')
    if not torch.equal(rng_cpu, torch.get_rng_state()) or not torch.equal(rng_cuda, torch.cuda.get_rng_state(0)):
        raise ValueError('只读诊断消耗了模型随机流')
    for split, rows in all_rows.items(): p.save(entry/(split+'_rows.json'), rows)
    p.save(entry/'calibration.json', calibration); p.save(entry/'membership.json', members)
    p.save(entry/'source_truth.json', truths)
    p.save(entry/'result.json', dict(status='completed_source_complementarity', schema=SCHEMA, shift=args.child,
        code=identity, specification=SPEC, metrics=summarize(validation, calibration), per_video=per_video,
        train_eligible=sum(bool(r['acceptable_classes']) for r in all_rows['train']),
        model_sha256_before=before, model_sha256_after=before, model_rng_unchanged=True, full_backend_null_check=null_checked,
        target_data_read=False, source_gt_input_to_model=False, formal_checkpoint_written=False,
        input_proof_sha256=p.sha(args.output_root/'inputs.json'), reference_artifacts=source_proof['reference_artifacts'],
        source_manifest_sha256=p.sha(cfg['dataset']['json_file']), feature_inputs=features,
        gpu_guard=guard, gpu_guard_sha256=p.sha(guard), peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        artifacts={str(entry/name):p.sha(entry/name) for name in ('train_rows.json','validation_rows.json','calibration.json','membership.json','source_truth.json')}))


def audit_source(entry):
    """禁CUDA重算可部署选择与纠错，而非仅相信worker的通过字段。"""
    entry=Path(entry);result=p.read(entry/'result.json');root=entry.parents[1];proof=p.read(root/'inputs.json')
    source=proof['sources'][result['shift']]
    if result['code']!=proof['code'] or result['specification']!=SPEC or result['input_proof_sha256']!=p.sha(root/'inputs.json'):
        raise ValueError('源结果签名改变')
    for path,digest in result['artifacts'].items():
        if p.sha(path)!=digest:raise ValueError('源产物改变')
    for name,digest in result['reference_artifacts'].items():
        if p.sha(Path(source['reference'])/name)!=digest:raise ValueError('冻结原模型改变')
    if p.sha(source['source_manifest'])!=result['source_manifest_sha256']:raise ValueError('源清单改变')
    for path,digest in result['feature_inputs'].items():
        if p.sha(path)!=digest:raise ValueError('输入特征改变')
    members=p.read(entry/'membership.json')
    if members!=source['membership'] or set(members['train'])&set(members['validation']):raise ValueError('源成员不符')
    rows={split:p.read(entry/(split+'_rows.json')) for split in ('train','validation')}
    truths=p.read(entry/'source_truth.json')
    if set(truths)!=set(members['train'])|set(members['validation']):raise ValueError('源GT成员不符')
    for split,values in rows.items():
        for row in values:
            if row['video_id'] not in members[split]:raise ValueError('跨split候选')
            truth=truths[row['video_id']];labels,quality=matching_labels(truth['segments'],truth['labels'],row['feature_interval'])
            if labels!=row['acceptable_classes'] or abs(quality-row['maximum_tIoU'])>1e-9:raise ValueError('离线GT匹配未复现')
            for name,vector in [('detector',foreground_distribution([row['detector_scores']])),('prototype',np.asarray([row['prototype_probabilities']]))]:
                cls,margin=prediction_margin(vector)
                if row[name+'_class']!=int(cls[0]) or abs(row[name+'_margin']-float(margin[0]))>1e-6:raise ValueError('分支预测摘要与原始分数不符')
    calibration=calibrate(rows['train'])
    if calibration!=p.read(entry/'calibration.json'):raise ValueError('校准未由source_train独立复现')
    if summarize(rows['validation'],calibration)!=result['metrics']:raise ValueError('源纠错未复现')
    per_video=[dict(video_id=vid,**summarize([r for r in rows['validation'] if r['video_id']==vid],calibration)) for vid in members['validation']]
    if per_video!=result['per_video']:raise ValueError('视频聚类统计未复现')
    if (result['target_data_read'] or result['source_gt_input_to_model'] or result['formal_checkpoint_written'] or
        not result['model_rng_unchanged'] or not result['full_backend_null_check'] or result['model_sha256_before']!=result['model_sha256_after']):
        raise ValueError('只读源数据或模型隔离失败')
    guard=p.read(result['gpu_guard']);lease=p.read(guard['lease'])
    if (p.sha(result['gpu_guard'])!=result['gpu_guard_sha256'] or p.sha(guard['lease'])!=guard['lease_sha256'] or
        lease['commit']!=proof['code']['commit'] or guard['observed']['uuid']!=lease['observed']['uuid'] or
        guard['observed']['free_mib']<lease['minimum_free_mib'] or lease['minimum_free_mib']<12288):raise ValueError('GPU准入证明不符')
    return dict(status='passed',shift=result['shift'],result=str(entry/'result.json'),result_sha256=p.sha(entry/'result.json'),metrics=result['metrics'])
