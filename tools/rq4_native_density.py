"""固定源域原生定位诊断；GT只评分，不参与抽样、匹配预测或选择提示。"""
import math
import numpy as np
from tools.rq4_protocol import digest, decode_response, interval_iou, match_response

POLICY=dict(seed=6,gpu=3,videos=20,per_video=2,shifts=['S2','S3'],
    conditions={'fps8':dict(fps=8.,max_frames=200),'fps16':dict(fps=16.,max_frames=400)},
    max_new_tokens=256,minimum_delta_iou=.02,bootstrap_samples=10000,
    primary='fps16',per_teacher_interval=.975,automatic_student_training=False)


def select_requests(manifests,excluded_videos):
    groups={}
    for manifest in manifests:
        if manifest['split']!='source_validation' or manifest['shift'] not in POLICY['shifts'] or manifest['seed']!=6:
            raise ValueError('wrong source pool')
        for r in manifest['requests']:
            c=r['candidate']
            if c['video_id'] in excluded_videos or not c['candidate_id'].endswith('-context3'):continue
            groups.setdefault(c['video_id'],{})[c['candidate_id']]=r
    eligible=[v for v,g in groups.items() if len(g)>=2]
    videos=sorted(eligible,key=lambda v:digest([6,'native-density-video',v]))[:POLICY['videos']]
    if len(videos)!=POLICY['videos']:raise ValueError('insufficient fixed video budget')
    return [r for v in videos for r in sorted(groups[v].values(),
        key=lambda r:digest([6,'native-density-candidate',r['candidate']['candidate_id']]))[:2]]


def prediction(candidate,response,teacher):
    """预先固定最大proposal重叠匹配；格式无效/无匹配均保留proposal。"""
    if response.get('error'):raise RuntimeError(response['error'])
    if teacher=='timeple_8b' and '<|TIMESPAN|>' in response['raw_text'] and response['metadata'].get('codec_predictions') is None:
        raise RuntimeError('TIMESPAN codec technical failure')
    try:
        spans=decode_response(teacher,response['raw_text'],response['metadata'],candidate['window'][1]-candidate['window'][0])
    except (ValueError,TypeError,KeyError):
        return list(candidate['proposal']),'invalid_output'
    box,status=match_response(candidate,spans,0.,0.)
    return box if box is not None else list(candidate['proposal']),status


def actual_times(metadata):
    if 'frame_timestamps' in metadata:
        values=metadata['frame_timestamps']
    else:
        m=metadata['video_metadata']
        if isinstance(m,list):m=m[0]
        values=[i/m['fps'] for i in m['frames_indices']]
    times=[float(t) for t in values]
    if not times or not all(math.isfinite(t) for t in times) or times!=sorted(times):
        raise ValueError('invalid actual frame timestamps')
    return times


def temporal_audit(candidate,times,gt=None):
    absolute=[t+candidate['window'][0] for t in times]
    def box_audit(box):
        return dict(frames_inside=sum(box[0]<=t<=box[1] for t in absolute),
            nearest_start_seconds=min(abs(t-box[0]) for t in absolute),
            nearest_end_seconds=min(abs(t-box[1]) for t in absolute),
            start_neighbors=sum(abs(t-box[0])<=.125 for t in absolute),
            end_neighbors=sum(abs(t-box[1])<=.125 for t in absolute))
    gaps=np.diff(times)
    return dict(sampled_frames=len(times),unique_frames=len(set(times)),
        maximum_gap_seconds=float(max(gaps)) if len(gaps) else None,
        proposal=box_audit(candidate['proposal']),gt=box_audit(gt) if gt is not None else None,
        note='Sampling coverage is not proof that the visual boundary is identifiable.')


def mean_by_video(rows,key):
    return {v:float(np.mean([r[key] for r in rows if r['video_id']==v])) for v in sorted({r['video_id'] for r in rows})}


def paired_summary(rows):
    if not rows:return dict(count=0,video_count=0,mean_delta_iou=None,interval=None)
    per_video=mean_by_video(rows,'delta_iou');values=np.array(list(per_video.values()))
    rng=np.random.default_rng(6)
    boot=values[rng.integers(0,len(values),size=(POLICY['bootstrap_samples'],len(values)))].mean(1)
    tail=(1-POLICY['per_teacher_interval'])/2
    return dict(count=len(rows),video_count=len(values),mean_delta_iou=float(values.mean()),
        interval=[float(x) for x in np.quantile(boot,[tail,1-tail])],per_video_delta_iou=per_video,
        baseline_iou=float(np.mean(list(mean_by_video(rows,'baseline_iou').values()))),
        teacher_iou=float(np.mean(list(mean_by_video(rows,'teacher_iou').values()))),
        start_error=float(np.mean(list(mean_by_video(rows,'start_error').values()))),
        end_error=float(np.mean(list(mean_by_video(rows,'end_error').values()))),
        baseline_start_error=float(np.mean(list(mean_by_video(rows,'baseline_start_error').values()))),
        baseline_end_error=float(np.mean(list(mean_by_video(rows,'baseline_end_error').values()))),
        improved=sum(r['delta_iou']>1e-8 for r in rows),harmed=sum(r['delta_iou'] < -1e-8 for r in rows),
        unchanged=sum(abs(r['delta_iou'])<=1e-8 for r in rows))


def evaluate(manifest,responses,truth,teacher):
    requests=manifest['requests']
    if len(responses)!=len(requests):raise ValueError('missing response rows')
    rows=[];formats={}
    for req,response in zip(requests,responses):
        if response['request_sha256']!=digest(req) or response['request_id']!=req['request_id'] or response['teacher']!=teacher:
            raise ValueError('response identity mismatch')
        c=req['request']['candidate'];label=truth[c['candidate_id']]
        box,status=prediction(c,response,teacher)
        times=actual_times(response['metadata'])
        audit=temporal_audit(c,times,label['gt'])
        formats.setdefault(req['condition'],{}).setdefault(status,0);formats[req['condition']][status]+=1
        row=dict(candidate_id=c['candidate_id'],video_id=c['video_id'],condition=req['condition'],
            prediction=box,status=status,scorable=label['scorable'],sampling=audit)
        if label['scorable']:
            gt=label['gt'];p=c['proposal'];length=gt[1]-gt[0]
            old,new=interval_iou(p,gt),interval_iou(box,gt)
            row.update(baseline_iou=old,teacher_iou=new,delta_iou=new-old,
                start_error=abs(box[0]-gt[0])/length,end_error=abs(box[1]-gt[1])/length,
                baseline_start_error=abs(p[0]-gt[0])/length,baseline_end_error=abs(p[1]-gt[1])/length)
        rows.append(row)
    summaries={k:paired_summary([r for r in rows if r['condition']==k and r['scorable']]) for k in POLICY['conditions']}
    pairs={}
    for r in rows:pairs.setdefault(r['candidate_id'],{})[r['condition']]=r
    dense_deltas=[]
    for p in pairs.values():
        if set(p)!=set(POLICY['conditions']):raise ValueError('missing paired density')
        if p['fps16']['sampling']['sampled_frames']<p['fps8']['sampling']['sampled_frames']:
            raise ValueError('dense condition sampled fewer frames')
        if p['fps8']['scorable']:
            dense_deltas.append(dict(video_id=p['fps8']['video_id'],delta=p['fps16']['teacher_iou']-p['fps8']['teacher_iou']))
    return dict(summary=summaries,output_status=formats,rows=rows,
        high_minus_low_video_mean=float(np.mean(list(mean_by_video(dense_deltas,'delta').values()))) if dense_deltas else None)


def decision(reports):
    # 两教师的16fps是预定主比较，97.5%逐教师区间作两比较的保守控制。
    primary=[v['summary']['fps16'] for v in reports.values()]
    if any(p['video_count']<10 for p in primary):return 'insufficient_scorable_videos'
    if any(p['mean_delta_iou']>=.02 and p['interval'][0]>0 for p in primary):return 'signal_for_separate_mechanism_validation'
    if all(p['interval'][1]<.02 for p in primary):return 'do_not_expand_current_direct_transfer_route'
    return 'inconclusive_no_automatic_expansion'
