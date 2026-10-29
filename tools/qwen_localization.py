"""源验证定位诊断：推理输入无GT，先冻结全部评分，再按置信度选20%。"""
from bisect import bisect_left
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess

from tools.baseline_protocol import read, save, sha
from tools.qwen_source_interface import LABELS, ACTION_PHRASES, build_prompt as class_prompt

SCHEMA = 'qwen-source-localization-v1'
OPTIONS = tuple('ABCDEFGHIJKLMNOPQRS')
POLICY = dict(model='Qwen3-VL-8B-Instruct', frames=16, input='chronological_timestamped_images',
    context='existing_source_validation_context2_clips', min_pixels=224**2, max_pixels=448**2,
    class_source='frozen_existing_8b_predictions_not_ground_truth',
    boundaries='independent_START_END_single_token_choice', temperature=1.,
    selection='per_shift_floor_20pct_rank_min_start_end_full_vocabulary_token_probability',
    failure_score=-1., tie_break='candidate_id', target_data_used=False,
    evaluation_reference='unique_best_original_proposal_overlap_gt_instance_positive_overlap',
    interval_point='midpoint_of_selected_adjacent_observed_frames',
    no_student_training=True, existing_validation_diagnostic_not_untouched_test=True)


def temporal_iou(a, b):
    return max(0., min(a[1], b[1])-max(a[0], b[0])) / max(1e-12, max(a[1], b[1])-min(a[0], b[0]))


def frozen_class(probabilities):
    if len(probabilities) != 10 or not all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities):
        raise ValueError('原8B类别概率非法')
    if abs(sum(probabilities)-1.) > 1e-5:
        raise ValueError('原8B类别概率未归一化')
    winners = [i for i,x in enumerate(probabilities) if x == max(probabilities)]
    return winners[0] if len(winners) == 1 else None


def sample_indices(pts):
    if not pts or any(not math.isfinite(x) for x in pts) or any(b <= a for a,b in zip(pts,pts[1:])):
        raise ValueError('源视频真实帧时间戳非法')
    targets = [pts[0]+(pts[-1]-pts[0])*(i+.5)/16 for i in range(16)]
    indices = []
    for t in targets:
        right = min(bisect_left(pts,t),len(pts)-1)
        indices.append(min({right,max(0,right-1)}, key=lambda j:(abs(pts[j]-t),j)))
    return indices if len(set(indices)) == 16 else None


def build_prompt(row, boundary):
    if boundary not in ('START','END') or len(row['frame_times_sec']) != 16 or row['class_id'] is None:
        raise ValueError('定位提示输入非法')
    times = row['frame_times_sec']
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('帧必须严格按真实时间排序')
    options = [f'{OPTIONS[i]}: between frame {i} ({times[i]:.6f} s) and frame {i+1} ({times[i+1]:.6f} s)'
               for i in range(15)]
    options += ['P: the boundary is before the first sampled frame',
                'Q: the boundary is after the last sampled frame',
                'R: no identifiable occurrence of the target action is visible',
                'S: UNCERTAIN - insufficient evidence or ambiguous occurrence']
    return (
        'You are locating a hand-object action in an egocentric video.\n'
        f"TARGET ACTION: {LABELS[row['class_id']]} ({ACTION_PHRASES[row['class_id']]})\n"
        'The images are chronological video frames with actual timestamps. They include context '
        'before and after an imperfect detector proposal.\n'
        f"REFERENCE PROPOSAL: {row['proposal'][0]:.6f} to {row['proposal'][1]:.6f} seconds.\n"
        'Locate the occurrence of the target action associated with this proposal. The proposal '
        'boundaries may be wrong. Do not copy them. If multiple occurrences cannot be distinguished, '
        'choose UNCERTAIN. Use visible changes in hand motion, object interaction, and object state. '
        'Seeing a relevant object alone does not prove the action is occurring. A persistent result, '
        'such as an already-open door, does not mean the opening action is still being performed.\n'
        f'BOUNDARY TO LOCATE: {boundary}\n'
        'For START, identify the transition into execution of the target action. For END, identify '
        'the transition out of execution of the target action. Choose the interval between consecutive '
        'sampled frames containing this boundary. Do not invent events between frames.\nOPTIONS:\n'
        + '\n'.join(options)
        + '\nAnswer with exactly one option letter. Do not provide an explanation.')


def option_scores(logits, log_total):
    if len(logits) != len(OPTIONS) or not all(math.isfinite(x) for x in logits) or not math.isfinite(log_total):
        raise ValueError('定位选项logits非有限或数量不符')
    peak = max(logits);weights = [math.exp(x-peak) for x in logits];total = sum(weights)
    absolute = [math.exp(x-log_total) for x in logits]
    if any(x > 1+1e-5 for x in absolute) or sum(absolute) > 1+1e-5:
        raise ValueError('全词表概率归一化非法')
    best = max(range(len(logits)), key=lambda i:logits[i])
    tied = sum(x == logits[best] for x in logits) > 1
    return dict(option=OPTIONS[best], option_index=best, tied=tied,
        full_vocab_logsumexp=log_total,
        probability=absolute[best], conditional_probability=weights[best]/total,
        candidate_mass=sum(absolute), option_logits=logits,
        option_probabilities=absolute, conditional_probabilities=[x/total for x in weights])


def combine(row, start, end):
    result = dict(candidate_id=row['candidate_id'], shift=row['shift'], video_id=row['video_id'],
        class_id=row['class_id'], start=start, end=end, status='unresolved', confidence=-1., interval=None)
    a,b = start['option_index'],end['option_index']
    if start['tied'] or end['tied']:
        result['reason'] = 'tied_boundary_options'
    elif a >= 15 or b >= 15:
        result['reason'] = 'abstain_or_boundary_outside_observed_frames'
    elif a >= b:
        result['reason'] = 'reversed_or_unresolved_same_gap'
    else:
        ts = row['frame_times_sec']
        result.update(status='localized',reason='ordered_internal_gaps',
            confidence=min(start['probability'],end['probability']),
            interval=[(ts[a]+ts[a+1])/2,(ts[b]+ts[b+1])/2])
    return result


def top_twenty(rows, predictions):
    ids = [r['candidate_id'] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(predictions):
        raise ValueError('评分身份不完整或重复')
    selected = []
    for shift in sorted({r['shift'] for r in rows}):
        subset = [r for r in rows if r['shift'] == shift]
        for r in subset:
            score=predictions[r['candidate_id']]['confidence']
            if not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError('定位置信度非法')
        ranked = sorted(subset, key=lambda r:(-predictions[r['candidate_id']]['confidence'],r['candidate_id']))
        selected += [r['candidate_id'] for r in ranked[:len(subset)//5]]
    return selected


def reference_instance(proposal, annotations):
    instances = sorted({(float(a['segment'][0]),float(a['segment'][1]),int(a['label_id']))
                        for a in annotations if a['segment'][1] > a['segment'][0]})
    scores = [temporal_iou(proposal,x[:2]) for x in instances]
    best = max(scores,default=0.)
    winners = [x for x,s in zip(instances,scores) if abs(s-best) < 1e-9]
    if best <= 0 or len(winners) != 1 or not 0 <= winners[0][2] < 10:
        return dict(status='no_unique_known_reference',original_tiou=best)
    x = winners[0]
    return dict(status='matched',interval=list(x[:2]),class_id=x[2],original_tiou=best)


def extract_frames(item, root, cls):
    import cv2
    from PIL import Image
    clip = Path(item['clip_path'])
    if sha(clip) != item['clip_sha256']:
        raise ValueError('源clip SHA改变')
    frames = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_entries','frame=best_effort_timestamp_time','-of','json',str(clip)],text=True))['frames']
    pts = [float(x['best_effort_timestamp_time']) for x in frames]
    indices = sample_indices(pts)
    common = {k:item[k] for k in ('candidate_id','shift','video_id','proposal','clip_path','clip_sha256','clip_start_sec')}
    common.update(class_id=cls,frame_paths=[],frame_sha256=[],frame_times_sec=[],frame_indices=[],
                  frame_count_available=len(pts))
    if indices is None or cls is None:
        return dict(common,input_status='insufficient_distinct_temporal_frames' if indices is None else 'cached_class_tie')
    folder = root/'frames'/item['candidate_id'];folder.mkdir(parents=True,exist_ok=False)
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError('真实源视频解码器无法打开视频')
    try:
        for k,index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame = cap.read()
            if not ok:
                raise ValueError('真实源视频解码失败，不能伪装成拒答')
            if abs(cap.get(cv2.CAP_PROP_POS_FRAMES)-(index+1))>.1:
                raise ValueError('解码帧序号与真实时间戳不符')
            path = folder/f'{k:02d}.png'
            Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(path)
            common['frame_paths'].append(str(path));common['frame_sha256'].append(sha(path))
        common.update(frame_indices=indices,frame_times_sec=[item['clip_start_sec']+pts[i] for i in indices],
                      input_status='ready')
    finally:
        cap.release()
    common['prompt_sha256'] = {b:hashlib.sha256(build_prompt(common,b).encode()).hexdigest() for b in ('START','END')}
    return common


def prepare(root, previous, baseline):
    frozen = read(previous/'frozen_candidates.json')
    old_manifest = read(previous/'manifest.json');receipt = read(previous/'8b_inference_complete.json')
    if receipt['status'] != 'passed' or sha(previous/'8b_predictions.jsonl') != receipt['predictions_sha256']:
        raise ValueError('旧8B类别评分凭证失败')
    if sha(previous/'manifest.json') != receipt['manifest_sha256']:
        raise ValueError('旧8B推理输入改变')
    if sha(previous/'8b_model_identity.json') != receipt['model_identity_sha256']:
        raise ValueError('旧8B模型身份改变')
    pred_rows = [json.loads(x) for x in (previous/'8b_predictions.jsonl').read_text().splitlines()]
    predictions = {x['candidate_id']:x for x in pred_rows}
    items = frozen['inputs'];ids = {x['candidate_id'] for x in items}
    if len(ids) != len(items) or len(predictions) != len(pred_rows) or ids != set(predictions):
        raise ValueError('旧候选或类别评分身份不完整')
    if {x['candidate_id'] for x in old_manifest['candidates']} != ids:
        raise ValueError('旧类别评分不是冻结候选')
    for row in old_manifest['candidates']:
        prediction=predictions[row['candidate_id']]
        expected=hashlib.sha256(class_prompt(row['candidate_id'],8,'baseline','native_video').encode()).hexdigest()
        if prediction['frame_sha256']!=row['frame_sha256'] or prediction['prompt_sha256']!=expected:
            raise ValueError('旧类别缓存与其真实帧/提示不符')
    classes = {k:frozen_class(v['class_probabilities']) for k,v in predictions.items()}
    labels = {};sources = {}
    for shift in sorted({x['shift'] for x in items}):
        cfg = read(baseline/'formal'/shift/'uda/config.json')
        path=Path(cfg['dataset']['json_file'])
        if sha(path) != frozen['sources'][shift]['source_json_sha256']:
            raise ValueError('源GT版本与冻结诊断候选不符')
        db=read(path)['database']
        train={v for v,d in db.items() if d['subset'].lower() in cfg['src_train_split']}
        val={v for v,d in db.items() if d['subset'].lower() in cfg['src_val_split']}
        if (train & val or set(frozen['membership'][shift]['validation_videos']) != val
                or set(frozen['membership'][shift]['train_videos']) != train):
            raise ValueError('源训练/验证隔离失败')
        for item in [x for x in items if x['shift']==shift]:
            if item['video_id'] not in val:
                raise ValueError('定位候选不在源验证集')
            labels[item['candidate_id']] = reference_instance(item['proposal'],db[item['video_id']].get('annotations',[]))
        sources[shift]=dict(source_json=str(path),sha256=sha(path),validation_videos=sorted(val))
    save(root/'evaluation_labels.json',labels)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=[]
        for i,row in enumerate(pool.map(lambda x:extract_frames(x,root,classes[x['candidate_id']]),items)):
            rows.append(row)
            if (i+1)%50==0:
                print('LOCALIZATION_FRAMES',i+1,len(items),flush=True)
    manifest=dict(schema=SCHEMA,policy=POLICY,candidates=rows,target_data_used=False,
        ground_truth_in_inference_input=False,candidate_counts=dict(Counter(x['shift'] for x in rows)),
        input_status_counts=dict(Counter(x['input_status'] for x in rows)))
    save(root/'manifest.json',manifest)
    save(root/'inputs.json',dict(previous_root=str(previous),baseline_root=str(baseline),source_manifests=sources,
        previous_sha256={n:sha(previous/n) for n in ['frozen_candidates.json','manifest.json','8b_predictions.jsonl','8b_inference_complete.json','8b_model_identity.json']},
        manifest_sha256=sha(root/'manifest.json'),evaluation_labels_sha256=sha(root/'evaluation_labels.json')))
    return manifest


def reuse_preparation(root, prepared, identity):
    if read(prepared/'state.json')['phase']!='prepared_only':
        raise ValueError('只能复用完整CPU准备，不恢复部分推理')
    plan=read(prepared/'plan.json');inputs=read(prepared/'inputs.json')
    if plan['code']!=identity or plan['policy']!=POLICY:
        raise ValueError('CPU准备不属于当前冻结代码/定位协议')
    if (sha(prepared/'manifest.json')!=inputs['manifest_sha256']
            or sha(prepared/'evaluation_labels.json')!=inputs['evaluation_labels_sha256']):
        raise ValueError('CPU准备工件改变')
    for name,h in inputs['previous_sha256'].items():
        if sha(Path(inputs['previous_root'])/name)!=h:
            raise ValueError('CPU准备依赖的原类别缓存改变')
    for name in ('manifest.json','inputs.json','evaluation_labels.json'):
        shutil.copyfile(prepared/name,root/name)
    save(root/'preparation_reuse.json',dict(prepared_root=str(prepared),
        copied_sha256={n:sha(root/n) for n in ('manifest.json','inputs.json','evaluation_labels.json')}))
    return read(root/'manifest.json')


def score_subset(rows, labels, predictions):
    matched=[r for r in rows if labels[r['candidate_id']]['status']=='matched']
    correct=[r for r in matched if r['class_id']==labels[r['candidate_id']]['class_id']]
    def measure(subset):
        values=[]
        for row in subset:
            cid=row['candidate_id'];gt=labels[cid];pred=predictions[cid]
            same=row['class_id']==gt['class_id']
            estimated=pred['interval'] if pred['status']=='localized' else None
            raw=temporal_iou(estimated,gt['interval']) if estimated else 0.
            before=gt['original_tiou'] if same else 0.;after=raw if same else 0.
            ts=row['frame_times_sec'];grid=[(a+b)/2 for a,b in zip(ts,ts[1:])]
            ceiling=max((temporal_iou([a,b],gt['interval']) for i,a in enumerate(grid) for b in grid[i+1:]),default=0.)
            e=dict(candidate_id=cid,video_id=row['video_id'],class_correct=same,
                original_tiou=before,predicted_tiou=after,gain=after-before,grid_best_tiou=ceiling,
                start_abs_error_sec=abs(estimated[0]-gt['interval'][0]) if estimated and same else None,
                end_abs_error_sec=abs(estimated[1]-gt['interval'][1]) if estimated and same else None,
                both_gt_boundaries_in_observed_span=bool(ts and ts[0]<=gt['interval'][0]<gt['interval'][1]<=ts[-1]))
            if estimated and same:
                a,b=pred['start']['option_index'],pred['end']['option_index']
                e.update(start_gt_inside_selected_gap=ts[a]<=gt['interval'][0]<=ts[a+1],
                         end_gt_inside_selected_gap=ts[b]<=gt['interval'][1]<=ts[b+1])
            values.append(e)
        n=len(values)
        avg=lambda key:sum(x[key] for x in values)/n if n else None
        finite_errors=[x for x in values if x['start_abs_error_sec'] is not None]
        return dict(count=n,original_mean_tiou=avg('original_tiou'),vlm_mean_tiou=avg('predicted_tiou'),
            mean_gain=avg('gain'),grid_best_mean_tiou=avg('grid_best_tiou'),
            improved=sum(x['gain']>1e-9 for x in values),degraded=sum(x['gain'] < -1e-9 for x in values),
            success_at={str(t):sum(x['predicted_tiou']>=t for x in values)/n if n else None for t in (.3,.5,.7)},
            original_success_at={str(t):sum(x['original_tiou']>=t for x in values)/n if n else None for t in (.3,.5,.7)},
            valid_localization_count=len(finite_errors),
            start_mae_on_localized_sec=sum(x['start_abs_error_sec'] for x in finite_errors)/len(finite_errors) if finite_errors else None,
            end_mae_on_localized_sec=sum(x['end_abs_error_sec'] for x in finite_errors)/len(finite_errors) if finite_errors else None,
            per_candidate=values)
    return dict(total_selected_candidates=len(rows),matched_reference_count=len(matched),
        prediction_status_counts=dict(Counter(predictions[r['candidate_id']]['status'] for r in rows)),
        selected_class_counts=dict(Counter(LABELS[r['class_id']] if r['class_id'] is not None else 'TIE' for r in rows)),
        matched_class_aware=measure(matched),correct_class_only_localization=measure(correct))


def summarize(root):
    manifest=read(root/'manifest.json');rows=manifest['candidates']
    lines=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()]
    predictions={x['candidate_id']:x for x in lines}
    if len(lines)!=len(predictions):
        raise ValueError('重复定位评分')
    # 必须先只依赖模型评分冻结名单，随后才读评估GT。
    chosen=top_twenty(rows,predictions)
    save(root/'top20_selection.json',dict(candidate_ids=chosen,policy=POLICY,
        predictions_sha256=sha(root/'predictions.jsonl'),ground_truth_used_for_selection=False))
    labels=read(root/'evaluation_labels.json');selected=set(chosen);result={}
    for shift in sorted({r['shift'] for r in rows})+['pooled_candidates_not_independent']:
        subset=rows if shift.startswith('pooled') else [r for r in rows if r['shift']==shift]
        result[shift]=dict(all=score_subset(subset,labels,predictions),
            top20=score_subset([r for r in subset if r['candidate_id'] in selected],labels,predictions))
    save(root/'comparison.json',dict(status='passed',schema=SCHEMA,policy=POLICY,results=result,
        manifest_sha256=sha(root/'manifest.json'),selection_sha256=sha(root/'top20_selection.json'),
        labels_sha256=sha(root/'evaluation_labels.json'),predictions_sha256=sha(root/'predictions.jsonl')))
    return result


def audit_predictions(root):
    """不读GT：复算提示SHA、选项概率、有效区间和排序输入。"""
    rows=read(root/'manifest.json')['candidates'];lines=[json.loads(x) for x in (root/'predictions.jsonl').read_text().splitlines()]
    predictions={x['candidate_id']:x for x in lines}
    if len(lines)!=len(predictions) or set(predictions)!={x['candidate_id'] for x in rows}:
        raise ValueError('定位评分身份不完整')
    for row in rows:
        p=predictions[row['candidate_id']]
        if row['input_status']!='ready':
            if p['status']!='input_unavailable' or p['confidence']!=-1. or p['interval'] is not None:
                raise ValueError('不可用输入不能伪造有效预测')
            continue
        for name,key in [('START','start'),('END','end')]:
            q=p[key];expected=option_scores(q['option_logits'],q['full_vocab_logsumexp'])
            prompt_hash=hashlib.sha256(build_prompt(row,name).encode()).hexdigest()
            if q!={**expected,'prompt_sha256':prompt_hash} or prompt_hash!=row['prompt_sha256'][name]:
                raise ValueError('定位概率或实际提示不可复算')
        if p!=combine(row,p['start'],p['end']):
            raise ValueError('时间换算或置信度与原始选项不一致')
    top_twenty(rows,predictions)
    return dict(status='passed',count=len(rows),manifest_sha256=sha(root/'manifest.json'),
                predictions_sha256=sha(root/'predictions.jsonl'))
