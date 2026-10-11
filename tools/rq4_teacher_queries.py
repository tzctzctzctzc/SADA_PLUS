"""直接视频查询诊断；请求生成不接收GT，评分仅在全部响应冻结后运行。"""
from copy import deepcopy
import re

from tools.rq4_protocol import digest, interval_iou

POLICY = dict(seed=6,shift='S1',per_video=2,fps=8.,max_frames=200,max_new_tokens=64,
              point_offsets=[-.25,.5,1.25],perturbation=.25,tolerance_fraction=.1,
              tolerance_floor_seconds=.125,ranking_tie_iou=.02,gpu=0)
OPTIONS = dict(state=['BEFORE','DURING','AFTER','ABSENT','UNCERTAIN'],
               direction=['EARLIER','LATER','KEEP','ABSENT','UNCERTAIN'],
               ranking=['A','B','TIE','ABSENT','UNCERTAIN'])


def choose(requests):
    groups={}
    for request in requests:
        c=request['candidate']
        if c['candidate_id'].endswith('-context3'):
            groups.setdefault(c['video_id'],[]).append(request)
    return [r for vid in sorted(groups) for r in sorted(groups[vid],
            key=lambda x:digest([POLICY['seed'],x['candidate']['candidate_id']]))[:POLICY['per_video']]]


def tasks(candidate,query):
    """绝对时间仅作内部记账；给模型的时间全部转换到实际clip起点。"""
    p=candidate['proposal']; origin,finish=candidate['window']; length=p[1]-p[0]
    base=candidate['candidate_id'].rsplit('-context',1)[0]
    reference=[x-origin for x in p]
    header=(f'Watch this video to judge the event "{query}". All times below are seconds from '
        f'the start of this clip. The reference interval [{reference[0]:.3f}, {reference[1]:.3f}] '
        'is an uncertain detector proposal, NOT a correct answer. Judge the occurrence nearest '
        'this reference interval. Use visible evidence; if that occurrence is absent choose ABSENT, '
        'and if you cannot judge choose UNCERTAIN. ')
    outputs=[]
    def add(kind,name,body,**fields):
        outputs.append(dict(task_id=base+'-'+name,candidate_id=base,video_id=candidate['video_id'],
            kind=kind,prompt=header+body+' Reply with exactly one of: '+', '.join(OPTIONS[kind])+'.',
            proposal=list(p),window=list(candidate['window']),class_id=candidate['class_id'],**fields))
    for index,offset in enumerate(POLICY['point_offsets']):
        time=min(max(p[0]+offset*length,origin),finish)
        add('state','state'+str(index),f'At time {time-origin:.3f}, is this occurrence not yet started '
            '(BEFORE), ongoing (DURING), or already finished (AFTER)?',time=time)
    for endpoint,name in enumerate(('start','end')):
        tolerance=max(POLICY['tolerance_floor_seconds'],POLICY['tolerance_fraction']*length)
        add('direction',name,f'The proposed {name} boundary is {p[endpoint]-origin:.3f}. '
            f'Should this boundary move EARLIER, move LATER, or KEEP its current time '
            f'(within {tolerance:.3f} seconds of the correct boundary)?',endpoint=endpoint,tolerance=tolerance)
    # 仅根据候选ID生成边界扰动，不根据GT选择容易的比较。
    code=int(digest([6,base,'comparison'])[:8],16); endpoint=code%2; sign=1 if (code//2)%2 else -1
    alternative=list(p);alternative[endpoint]+=sign*POLICY['perturbation']*length
    alternative[0]=max(origin,alternative[0]);alternative[1]=min(finish,alternative[1])
    if alternative[0]>=alternative[1] or alternative==p:raise ValueError('degenerate fixed perturbation')
    boxes=[list(p),alternative]
    if (code//4)%2:boxes.reverse()
    for swap in (False,True):
        order=list(reversed(boxes)) if swap else boxes
        a,b=[[v-origin for v in box] for box in order]
        add('ranking','rank_swap' if swap else 'rank',
            f'Which interval better covers the full occurrence while excluding unrelated time: '
            f'A [{a[0]:.3f}, {a[1]:.3f}] or B [{b[0]:.3f}, {b[1]:.3f}]? '
            'Choose TIE only when neither is clearly better.',boxes=deepcopy(order),swapped=swap)
    return outputs


def parse_answer(kind,raw):
    text=raw.strip()
    if text.endswith('<|im_end|>'):text=text[:-10].strip()
    match=re.fullmatch('('+'|'.join(OPTIONS[kind])+r')[.!]?',text)
    return match.group(1) if match else None


def match_gt(candidate,annotations):
    eligible=[a['segment'] for a in annotations if int(a['label_id'])==candidate['class_id']]
    ranked=sorted([(interval_iou(candidate['proposal'],g),g) for g in eligible],reverse=True)
    if not ranked or ranked[0][0]<=0:
        return dict(gt=None,status='no_matching_class_instance',scorable=False)
    if len(ranked)>1 and abs(ranked[0][0]-ranked[1][0])<1e-8:
        return dict(gt=None,status='ambiguous_instance',scorable=False)
    gt=ranked[0][1]
    # 避免将另一重复动作解释为本实例的BEFORE/AFTER。响应仍保留，不过滤查询。
    repeats=sum(interval_iou(candidate['window'],g)>0 for g in eligible)>1
    return dict(gt=gt,status='matched',scorable=True,repeated_same_class_in_clip=repeats)


def answer_for(task,gt):
    if gt is None:return None
    if task['kind']=='state':
        return 'BEFORE' if task['time']<gt[0] else 'AFTER' if task['time']>gt[1] else 'DURING'
    if task['kind']=='direction':
        delta=gt[task['endpoint']]-task['proposal'][task['endpoint']]
        return 'KEEP' if abs(delta)<=task['tolerance'] else 'EARLIER' if delta<0 else 'LATER'
    delta=interval_iou(task['boxes'][0],gt)-interval_iou(task['boxes'][1],gt)
    return 'TIE' if abs(delta)<=POLICY['ranking_tie_iou'] else 'A' if delta>0 else 'B'


def geometry_answer(task):
    return answer_for(task,task['proposal'])


def accuracy(rows):
    if not rows:return dict(count=0,accuracy=None,valid_rate=None)
    return dict(count=len(rows),accuracy=sum(r['answer']==r['truth'] for r in rows)/len(rows),
        valid_rate=sum(r['answer'] is not None for r in rows)/len(rows),
        abstention_rate=sum(r['answer'] in ('ABSENT','UNCERTAIN') for r in rows)/len(rows))


def evaluate(requests,responses,truth):
    lookup={r['request_id']:r for r in responses}
    if len(lookup)!=len(responses):raise ValueError('duplicate response identity')
    if set(lookup)-{r['request_id'] for r in requests}:raise ValueError('unknown response identity')
    formats={}
    for condition in ('video','black'):
        observed=[r for r in requests if r['condition']==condition and r['request_id'] in lookup]
        formats[condition]=dict(count=len(observed),valid=sum(
            parse_answer(r['task']['kind'],lookup[r['request_id']]['raw_text']) is not None for r in observed))
    rows=[]
    for request in requests:
        if request['request_id'] not in lookup:continue
        response=lookup[request['request_id']]
        if response['request_sha256']!=digest(request):raise ValueError('response/input mismatch')
        task=request['task'];label=truth[task['task_id']]
        if not label['scorable']:continue
        rows.append(dict(task_id=task['task_id'],candidate_id=task['candidate_id'],video_id=task['video_id'],
            kind=task['kind'],condition=request['condition'],swapped=task.get('swapped',False),
            truth=label['answer'],answer=parse_answer(task['kind'],response['raw_text']),
            geometry=geometry_answer(task),repeated=label.get('repeated_same_class_in_clip',False)))
    summary={}
    for kind in OPTIONS:
        for condition in ('video','black'):
            subset=[r for r in rows if r['kind']==kind and r['condition']==condition]
            canonical=[r for r in subset if not r['swapped']]
            score=accuracy(canonical)
            score['geometry_accuracy']=sum(r['geometry']==r['truth'] for r in canonical)/len(canonical) if canonical else None
            vids=sorted({r['video_id'] for r in canonical})
            score['per_video']={v:accuracy([r for r in canonical if r['video_id']==v]) for v in vids}
            score['video_balanced_accuracy']=sum(v['accuracy'] for v in score['per_video'].values())/len(vids) if vids else None
            score['truth_counts']={a:sum(r['truth']==a for r in canonical) for a in OPTIONS[kind]}
            present=[a for a,n in score['truth_counts'].items() if n]
            score['macro_recall']=sum(sum(r['answer']==a for r in canonical if r['truth']==a)/score['truth_counts'][a] for a in present)/len(present) if present else None
            score['nonrepeated']=accuracy([r for r in canonical if not r['repeated']])
            if kind=='ranking':
                pairs={}
                for r in subset:pairs.setdefault(r['candidate_id'],{})[r['swapped']]=r['answer']
                flip={'A':'B','B':'A','TIE':'TIE','ABSENT':'ABSENT','UNCERTAIN':'UNCERTAIN'}
                valid=[p for p in pairs.values() if len(p)==2 and p[False] in flip and p[True] in flip]
                score['swap_valid_pairs']=len(valid)
                score['swap_consistency']=sum(flip[p[False]]==p[True] for p in valid)/len(valid) if valid else None
            summary[kind+'_'+condition]=score
    return dict(summary=summary,rows=rows,all_response_format=formats,scoring='unmatched/ambiguous instances unscored, not falsely labelled ABSENT; invalid/abstain counted as incorrect on matched cases')


def validate_completion(requests,responses,completion):
    expected=len(requests) if completion['status']=='complete' else 14
    if completion['status'] not in ('complete','unsupported_answer_interface'):
        raise ValueError('unknown completion status')
    if completion['count']!=len(responses) or len(responses)!=expected:
        raise ValueError('incomplete response set')
    for request,response in zip(requests,responses):
        if response['request_id']!=request['request_id'] or response['request_sha256']!=digest(request):
            raise ValueError('response order/input mismatch')
    if completion['status']=='unsupported_answer_interface' and any(
            parse_answer(r['task']['kind'],v['raw_text']) is not None for r,v in zip(requests,responses)):
        raise ValueError('unsupported interface has valid answers')
