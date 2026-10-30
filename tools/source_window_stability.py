"""源域跨真实窗口定位核验；窗口/匹配先冻结，GT只评价，绝不训练。"""
import math
import numpy as np


SCHEMA = 'source-cross-window-localization-v1'
POLICY = dict(grid_alignment=32, minimum_window=128, context_duration_ratio=4.,
    maximum_window=2304, shift_fraction=.125, minimum_separation=32,
    views=3, uncertainty_views=[0, 2], quality_view=1,
    anchor='original_proposal_anchor_nearest_real_original_fpn_grid',
    interpolation=False, gt_for_window_or_prediction_selection=False,
    pair_matching=dict(same_video=True, same_level=True,
                       duration_ratio_max=2., confidence_gap_max=.1),
    primary='source_video_equal_matched_pair_concordance',
    bootstrap_unit='physical_video_across_shifts', bootstrap_repeats=2000,
    thresholds_fitted=False, training=False, target_data_used=False)


def window_plan(total, segment, level, anchor, maximum=2304):
    """起点对齐最粗stride，使每层同一绝对时间点无需插值即可对应。"""
    start, end = map(float, segment)
    if not (math.isfinite(start) and math.isfinite(end) and end > start):
        raise ValueError('非法proposal')
    alignment = POLICY['grid_alignment']
    stride = 2 ** int(level)
    if stride > alignment:
        raise ValueError('原FPN层超出冻结的六层结构')
    # 给三个真实窗口留下至少两个32步位移；短视频不补零伪造上下文。
    desired = max(POLICY['minimum_window'],
                  math.ceil(POLICY['context_duration_ratio']*(end-start)/alignment)*alignment)
    length = min(desired, maximum, ((int(total)-2*alignment)//alignment)*alignment)
    margin = max(2*stride, 8)
    if length < 64:
        return dict(valid=False, reason='video_too_short_for_three_real_windows')
    low = max(0, math.ceil((end+margin-length)/alignment)*alignment)
    high = min(((int(total)-length)//alignment)*alignment,
               math.floor((start-margin)/alignment)*alignment)
    if high-low < 2*alignment:
        return dict(valid=False, reason='insufficient_untruncated_context')
    center = round(((start+end)/2-length/2)/alignment)*alignment
    center = min(max(center, low+alignment), high-alignment)
    delta = max(alignment, round(length*POLICY['shift_fraction']/alignment)*alignment)
    starts = [max(low, center-delta), center, min(high, center+delta)]
    point = round(float(anchor)/stride)*stride
    if not start <= point <= end:
        return dict(valid=False, reason='original_anchor_outside_proposal')
    assert len(set(starts)) == 3 and all(x % alignment == 0 for x in starts)
    assert all(x+margin <= start and end <= x+length-margin for x in starts)
    return dict(valid=True, starts=starts, length=length, anchor=float(point),
                level=int(level), margin=margin, common=[starts[-1], starts[0]+length])


def interval_iou(a, b):
    intersection = max(0., min(a[1], b[1])-max(a[0], b[0]))
    return intersection/max(a[1]-a[0]+b[1]-b[0]-intersection, 1e-12)


def assess_prediction(segment, predictions, ground_truth, gt_labels, plan):
    """先有冻结预测，再与原proposal匹配的同一个GT比较，禁止逐视图挑最佳GT。"""
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (3, 2) or not np.isfinite(predictions).all():
        raise ValueError('必须是三个有限绝对时间边界')
    duration = float(segment[1]-segment[0])
    u = np.abs(predictions[0]-predictions[2])/duration
    result = dict(uncertainty=float(u.mean()), start_uncertainty=float(u[0]),
                  end_uncertainty=float(u[1]), predictions=predictions.tolist(),
                  prediction_fully_observed=bool((predictions[:, 0] >= plan['common'][0]).all()
                    and (predictions[:, 1] <= plan['common'][1]).all()),
                  nondegenerate=bool((predictions[:, 1] > predictions[:, 0]).all()))
    if not len(ground_truth):
        return dict(result, gt_matched=False)
    overlaps = [interval_iou(segment, gt) for gt in ground_truth]
    index = int(np.argmax(overlaps))  # 固定原proposal决定GT身份，而非任一新预测。
    gt = np.asarray(ground_truth[index], dtype=np.float64)
    error = np.abs(predictions[1]-gt)/max(float(gt[1]-gt[0]), 1e-12)
    return dict(result, gt_matched=bool(overlaps[index] > 0),
        reference_gt_index=index, reference_gt_label=int(gt_labels[index]),
        reference_gt=gt.tolist(), original_proposal_iou=float(overlaps[index]),
        iou=interval_iou(predictions[1], gt), start_error=float(error[0]),
        end_error=float(error[1]), boundary_error=float(error.mean()),
        # 仅诊断两个不确定性视图是否可能切换到别的实例，不参与筛选。
        offline_gt_identity_consistent=bool(all(int(np.argmax(
            [interval_iou(pred, g) for g in ground_truth])) == index for pred in predictions)))


def pair_statistics(rows, u_key='uncertainty', quality_key='iou', class_consistent=False):
    """候选对在同视频/层、相近时长/置信度内比较；先每视频等权。"""
    groups = {}
    direction = 1 if quality_key.endswith('error') else -1
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a['video_id'] != b['video_id'] or a['level'] != b['level']:
                continue
            if class_consistent and a['predicted_class'] != b['predicted_class']:
                continue
            if max(a['duration'], b['duration'])/min(a['duration'], b['duration']) > 2:
                continue
            if abs(a['confidence']-b['confidence']) > .1:
                continue
            du = a[u_key]-b[u_key]
            dq = a[quality_key]-b[quality_key]
            if du == 0 or dq == 0:
                continue
            concordance = float(du*dq*direction > 0)
            confidence = a['confidence']-b['confidence']
            confidence_concordance = (.5 if confidence == 0 else
                float(confidence*dq*(-direction) > 0))
            groups.setdefault(a['video_id'], []).append((concordance, confidence_concordance))
    per_video = {v:dict(pairs=len(x), concordance=float(np.mean([a for a,b in x])),
        confidence_concordance=float(np.mean([b for a,b in x]))) for v,x in groups.items()}
    return dict(videos=len(groups), pairs=sum(len(x) for x in groups.values()), per_video=per_video)


def retention_statistics(rows, key='uncertainty'):
    """同视频/层/时长箱/置信箱内精确保留一半，和同数量高置信候选比较。"""
    groups = {}
    for row in rows:
        group = (row['video_id'], row['level'], math.floor(math.log2(row['duration'])),
                 int(row['confidence']/.1))
        groups.setdefault(group, []).append(row)
    stable, confidence, random_expected = [], [], []
    for values in groups.values():
        if len(values) < 2:
            continue
        keep = max(1, len(values)//2)
        stable.extend(sorted(values, key=lambda x:(x[key], x['candidate_id']))[:keep])
        confidence.extend(sorted(values, key=lambda x:(-x['confidence'], x['candidate_id']))[:keep])
        # 同箱随机抽相同数量时，每个样本入选概率相同，均值期望无需蒙特卡洛。
        random_expected.append(dict(video=values[0]['video_id'], n=keep,
                                    mean=float(np.mean([x['iou'] for x in values]))))
    def video_means(values):
        buckets = {}
        for row in values:buckets.setdefault(row['video_id'], []).append(row['iou'])
        return {k:float(np.mean(v)) for k,v in buckets.items()}
    random_video = {}
    for row in random_expected:random_video.setdefault(row['video'], []).append(row)
    return dict(selected=len(stable), stable_iou_by_video=video_means(stable),
        confidence_iou_by_video=video_means(confidence),
        random_expected_iou_by_video={v:sum(x['n']*x['mean'] for x in xs)/sum(x['n'] for x in xs)
                                     for v,xs in random_video.items()})


def summarize(rows):
    from collections import Counter
    valid = [x for x in rows if x.get('valid')]
    matched = [x for x in valid if x['gt_matched'] and x['nondegenerate']]
    eligible = [x for x in matched if x['semantic_consistent'] and x['prediction_fully_observed']]
    return dict(total=len(rows), valid_three_windows=len(valid), gt_overlapping=len(matched),
        unavailable_reasons=dict(Counter(x['plan']['reason'] for x in rows if not x.get('valid'))),
        semantic_consistent=sum(x['semantic_consistent'] for x in matched),
        fully_observed=sum(x['prediction_fully_observed'] for x in matched), eligible=len(eligible),
        class_correct=sum(x['predicted_class']==x['reference_gt_label'] for x in eligible),
        same_instance_offline=sum(x['offline_gt_identity_consistent'] for x in eligible),
        all_overlap=pair_statistics(matched), primary=pair_statistics(eligible),
        same_predicted_class=pair_statistics(eligible,class_consistent=True),
        start=pair_statistics(eligible,'start_uncertainty','start_error'),
        end=pair_statistics(eligible,'end_uncertainty','end_error'),
        boundary_error=pair_statistics(eligible,'uncertainty','boundary_error'),
        retention=retention_statistics(eligible))


def combine(summaries):
    """同一真实视频可跨shift复用，不能把shift-video当独立bootstrap样本。"""
    physical = {}
    for summary in summaries.values():
        for video, row in summary['primary']['per_video'].items():
            physical.setdefault(video, []).append((row['concordance'],row['confidence_concordance']))
    values = np.asarray([np.mean(x,axis=0) for x in physical.values()])
    if len(values) < 2:
        return dict(status='insufficient_matched_source_evidence', physical_videos=len(values))
    rng=np.random.RandomState(123)
    boots=np.stack([values[rng.randint(0,len(values),len(values))].mean(0)
                    for _ in range(POLICY['bootstrap_repeats'])])
    mean=values.mean(0);interval=np.quantile(boots[:,0],[.025,.975])
    advantage=np.quantile(boots[:,0]-boots[:,1],[.025,.975])
    promising=bool(interval[0]>.5 and advantage[0]>0)
    return dict(status='source_signal_supported' if promising else 'source_signal_not_established',
        physical_videos=len(values), physical_video_mean_concordance=float(mean[0]),
        confidence_concordance=float(mean[1]), concordance_ci95=interval.tolist(),
        advantage_over_confidence=float(mean[0]-mean[1]),advantage_ci95=advantage.tolist(),
        rule='clustered_ci_lower_gt_half_and_advantage_ci_lower_gt_zero',
        target_risk_guarantee=False, training_authorized_by_this_report=False)
