"""只复核原高分候选的任务背景；三个固定假设，不搜索提示或阈值。"""
import hashlib
import json
import math
import random

LABELS = ('TAKE','PUT','WASH','OPEN','CLOSE','INSERT','TURN_ON','CUT','TURN_OFF','POUR')
ACTIONS = ('taking/picking up an object','putting/placing an object down',
           'washing/cleaning an object','opening an object','closing an object',
           'inserting an object into a container','turning on a device/light/flame/flow',
           'cutting/slicing an object','turning off a device/light/flame/flow',
           'pouring contents from a container')
MODES = ('presence','verify','contrast')
STATES = ('present','absent','uncertain')
SPEC = dict(schema='prototype-high-score-VLM-background-three-v1',cutoff=.3,
    target='pure task background: any-task-class max tIoU<.01',
    goal='delete AP-damaging high-score BG while protecting useful foreground',
    perfect_filter_reference_pp=4.557834209554855,
    selection='one complete source_val and one complete target_train video per S; SHA metadata only; at least one high-score prediction',
    frames='12 interval midpoints plus 2 before/2 after; nearest actual PTS; chronological timestamped RGB images',
    context_seconds='max(.5,.25*duration)',minimum_unique_inside=4,
    no_GT_to_VLM=True,detector_frozen=True,model='Qwen3-VL-32B-Instruct',
    dtype='bfloat16',attention='sdpa',quantization=False,CPU_or_disk_offload=False,
    max_pixels=448**2,min_pixels=224**2,contrast_alpha=.5,
    presence='blind any-of-ten ternary presence judgement',
    verify='predicted-class ternary query; only if absent query remaining nine; delete only if both absent',
    contrast='1.5*original option logits-.5*static repeated-frame option logits; not exact VCD/SEASON reproduction',
    primary_rule='unique absent argmax; uncertain/ties/insufficient frame visibility preserve original',
    original_scores_labels_endpoints_unchanged=True,prompt_or_parameter_search=False,
    scope='fixed selected complete-video capacity pilot, not full six-direction UDA training')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def option_order(view_id,question):
    order=list(range(3));random.Random(int(digest([view_id,question])[:16],16)).shuffle(order)
    return order


def prompt(row,question):
    inside=row['visible_interval'];catalog='; '.join(f'{k}: {v}' for k,v in zip(LABELS,ACTIONS))
    if question=='presence':
        query='Is there visible execution of ANY of the ten listed task actions inside the marked interval?'
    elif question=='predicted':
        query=f'Is {LABELS[row["label"]]} ({ACTIONS[row["label"]]}) visibly being executed inside the marked interval?'
    elif question=='others':
        rest='; '.join(f'{k}: {v}' for i,(k,v) in enumerate(zip(LABELS,ACTIONS)) if i!=row['label'])
        query='Is ANY of these other nine task actions visibly being executed inside the marked interval? '+rest
    else:raise ValueError(question)
    descriptions=('YES: at least part of the requested action execution is visible inside.',
        'NO: the observed interval clearly does not contain execution of the requested action(s).',
        'INSUFFICIENT: short/sparse/occluded or ambiguous evidence prevents deciding.')
    order=option_order(row['view_id'],question)
    choices='\n'.join(f'{c}: {descriptions[i]}' for c,i in zip('ABC',order))
    return (f'Inspect an egocentric video. Task action definitions: {catalog}.\n'
        f'The marked interval is [{inside[0]:.6f}, {inside[1]:.6f}) seconds. Context outside it is not an occurrence inside.\n'
        'Judge visible execution, not object presence or a persistent opened/closed/on/off state. '
        'Partial execution counts YES; an action need not start and finish in the interval. '
        'Do not infer absence merely because a brief action falls between frames.\n'+query+'\n'+choices+
        '\nAnswer with exactly one letter A, B, or C.')


def score(logits,order,log_total=None):
    if len(logits)!=3 or sorted(order)!=[0,1,2] or not all(math.isfinite(v) for v in logits):
        raise ValueError('invalid ternary logits')
    x=max(logits);weights=[math.exp(v-x) for v in logits];total=sum(weights)
    probs={STATES[state]:w/total for state,w in zip(order,weights)}
    tied=logits.count(x)>1;state='uncertain' if tied else STATES[order[logits.index(x)]]
    out=dict(state=state,tied=tied,probabilities=probs,BG_score=probs['absent'])
    if log_total is not None:
        if not math.isfinite(log_total) or x>log_total+1e-5:raise ValueError('bad full vocabulary normalization')
        out['option_mass']=sum(math.exp(v-log_total) for v in logits)
    return out


def decisions(row,responses):
    if not row['observable']:
        return {m:dict(delete=False,BG_score=0.,state='unobservable_preserved') for m in MODES}
    order=option_order(row['view_id'],'presence');p=score(responses['presence']['logits'],order)
    v=score(responses['predicted']['logits'],option_order(row['view_id'],'predicted'))
    if v['state']=='absent':
        other=score(responses['others']['logits'],option_order(row['view_id'],'others'))
        verified=dict(delete=other['state']=='absent',BG_score=v['BG_score']*other['BG_score'],
                      state='both_absent' if other['state']=='absent' else 'other_or_uncertain_preserved')
    else:verified=dict(delete=False,BG_score=0.,state='predicted_or_uncertain_preserved')
    alpha=SPEC['contrast_alpha']
    contrast=score([(1+alpha)*a-alpha*b for a,b in zip(responses['presence']['logits'],responses['static']['logits'])],order)
    return dict(presence=dict(p,delete=p['state']=='absent'),verify=verified,
                contrast=dict(contrast,delete=contrast['state']=='absent'))


def frame_indices(times,start,end):
    """选取真实显示时间；保留原预测区间，越界/不可见候选不作背景删除。"""
    from bisect import bisect_left
    if not times or end<=start or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('invalid physical time grid')
    a=max(float(start),times[0]);b=min(float(end),times[-1]);d=end-start
    if b<=a:return dict(observable=False,indices=[],times=[],visible_interval=[a,b],inside_unique=0)
    pad=max(.5,.25*d);before=max(times[0],a-pad);after=min(times[-1],b+pad)
    desired=[before+(a-before)*(i+.5)/2 for i in range(2)]
    desired += [a+(b-a)*(i+.5)/12 for i in range(12)]
    desired += [b+(after-b)*(i+.5)/2 for i in range(2)]
    selected=[]
    for t in desired:
        k=min(bisect_left(times,t),len(times)-1)
        selected.append(min({k,max(0,k-1)},key=lambda j:(abs(times[j]-t),j)))
    inside={i for i in selected[2:14] if a<=times[i]<b}
    observable=start>=times[0] and end<=times[-1] and len(inside)>=SPEC['minimum_unique_inside']
    return dict(observable=observable,indices=selected,times=[times[i] for i in selected],
                visible_interval=[a,b],inside_unique=len(inside))


def verify_response(request,response):
    if response['request']!=request or response['request_id']!=digest(request):raise ValueError('response identity changed')
    score(response['logits'],request['option_order'],response['logsumexp'])
    return response
