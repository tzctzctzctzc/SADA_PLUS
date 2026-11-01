"""CoTT Labeler 前置验证：完整三阶段；不把生成置信度称为校准概率。"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = 'cott-source-holdout-pseudolabel-quality-v1'
CLASSES = ['TAKE', 'PUT', 'WASH', 'OPEN', 'CLOSE', 'INSERT', 'TURN_ON', 'CUT', 'TURN_OFF', 'POUR']
DEFINITIONS = [
    'grasp an object and move it away from its resting place',
    'place a held object down and release it',
    'rinse, rub or wash an object, food or hands',
    'change a container, door or lid from closed to open',
    'change a container, door or lid from open to closed',
    'move an object into another object, container or slot',
    'switch a device, flame or water flow from inactive to active',
    'divide or slice material using a cutting tool',
    'switch a device, flame or water flow from active to inactive',
    'tilt a container so its contents flow out',
]
SPEC = dict(schema=SCHEMA, paper='https://aclanthology.org/2026.emnlp-main.1163/',
    scope='complete_three_stage_Labeler_quality_pilot_not_Lite_student_training',
    seed=123, source_videos_per_shift=2, duration_range_sec=[20, 180],
    selection='SHA256(123:shift:video_id), source_val metadata only; unique video inference cache',
    fps=2, frames_per_window=12, window_stride_frames=10,
    refine_frames=12, refine_padding='max(1 second, 0.25 * primitive duration), clipped to video',
    model='Qwen3-VL-32B-Instruct', dtype='bfloat16', attn='sdpa', quantization=False,
    decode='greedy', min_pixels=224**2, max_pixels=448**2,
    max_new_tokens=dict(transition=2048, grouping=8192, refine=1536, direct=4096),
    max_input_tokens=65536, schema_retries=1, score_threshold=.5, dedup_tiou=.5,
    transition_contract_version=2,
    grouping_contract_version=3,
    grouping_serialization='ordered_transition_index_spans_constrained_greedy_v1',
    retry_contract='same visual evidence plus exact schema error; do not include invalid assistant answer as a completion template',
    metrics_tiou=[.1,.2,.3,.4,.5], bootstrap_video_replicates=1000,
    adaptations=['10-class vocabulary plus OTHER', 'typed JSON output with explicit validation',
        'overlap stride and refinement padding fixed here because exact values are unspecified',
        'one frame-based global-time coordinate system', 'no streaming distillation in this pilot'],
    baselines=['same Qwen direct full-video sliding-window detection',
        'audited DyFADet SO source-best EMA full detection predictions'],
    baseline_note='SO is a fixed teacher reference, not exact crop-time UDA raw_pseudo replay',
    confidence_note='VLM self-reported confidence is not calibrated; report PR/counts as well as AP',
    training=False, target_gt_used=False, target_videos_used=False,
    limitations=['small short-video source selection subset, not independent target evidence',
        'paper non-overlapping primitive grouping may underrepresent simultaneous benchmark actions',
        'same base frame stream for direct and CoTT, but CoTT has additional refinement views and cost'])

def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))

def save(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()

def no_annotations(obj):
    if isinstance(obj,dict):
        assert not {'annotations','ground_truth','gt','segments_gt'} & set(obj)
        for v in obj.values(): no_annotations(v)
    elif isinstance(obj,list):
        for v in obj: no_annotations(v)

def select_videos(database,shift):
    pool=[v for v,x in database.items() if 20<=x['duration']<=180]
    pool.sort(key=lambda v:hashlib.sha256(('123:'+shift+':'+v).encode()).hexdigest())
    if len(pool)<2: raise ValueError('insufficient eligible videos: '+shift)
    return pool[:2]

def windows(count,k=12,stride=10):
    if count<2: raise ValueError('video needs at least two distinct frames')
    if count<=k: return [list(range(count))]
    starts=list(range(0,count-k+1,stride))
    if starts[-1]!=count-k: starts.append(count-k)
    out=[list(range(s,s+k)) for s in starts]
    assert set(i for w in out for i in w[:-1])==set(range(count-1))
    return out

SYSTEM = ('You annotate visible physical hand-object interactions in egocentric kitchen videos. '
    'Use only provided visual evidence and absolute timestamps in seconds. Describe one concrete '
    'low-level action, not intentions, camera motion or a chain of verbs. Boundaries are first and '
    'last visible execution/contact, not the whole context window. The hypothesis may be wrong. '
    'Return only the requested JSON object; no markdown. OTHER is allowed for actions outside '
    'the vocabulary or unclear identity. Never invent an action to fill a gap.\nVocabulary:\n' +
    '\n'.join(f'{c}: {d}.' for c,d in zip(CLASSES,DEFINITIONS)))

def transition_prompt(frames):
    pairs=[dict(pair=i,start=a['time_sec'],end=b['time_sec']) for i,(a,b) in enumerate(zip(frames,frames[1:]))]
    count=len(frames)-1
    slots=dict(transitions=[dict(pair=i,description=f'REPLACE with change from FRAME {i} to FRAME {i+1}',
        event='REPLACE with start, continuing, ending or none') for i in range(count)])
    return (f'There are {len(frames)} input frames and exactly {count} ADJACENT FRAME PAIRS. '
        'FRAME 0 is context for the first comparison, NOT an extra observation to caption. '
        'For pair i compare FRAME i with FRAME i+1: describe the physical CHANGE between '
        'them, including object state and hand contact. Do not independently caption each '
        'frame. Do not infer what happens after the final frame. Mark the action phase '
        'as start, continuing, ending or none. Keep one row even for no physical change. '
        f'Return exactly {count} rows with pair IDs 0 through {count-1}, in order; '
        f'pair {count} does not exist and must not appear. Do not add or remove slots. '
        'Fill the following complete output template, replacing every description and event '
        'placeholder with an observation from the specified pair.\nOUTPUT TEMPLATE:\n'+
        json.dumps(slots)+'\nEXACT PAIR TIMESTAMPS (seconds):\n'+json.dumps(pairs))

def grouping_prompt(transitions,duration):
    return ('Group the COMPLETE chronological transition evidence below into atomic physical '
        'interactions. Merge the same uninterrupted interaction across sliding-window boundaries; '
        'do not merge distinct repeated actions. Output non-overlapping intervals, one concrete '
        'verb per interval. Gaps are permitted. Multiple descriptions at one pair are overlapping '
        'window observations, not repeated actions. Assign exactly one vocabulary label or OTHER '
        'to each interval. The video duration is '+str(duration)+' seconds. '
        'Serialize each group using the given global transition pair IDs: start_pair is the '
        'first included pair, end_pair is the EXCLUSIVE end (last included pair plus one). '
        'The real start time is the start of start_pair; the real end time is the end of '
        'pair end_pair-1. Group consecutive evidence; do not generate float timestamps here. '
        'For EVERY adjacent pair of output rows i and i+1 enforce end_pair[i] <= start_pair[i+1]. '
        'An action may touch the next action at a shared boundary, but no instant can belong '
        'to two primitive rows. Where two descriptions span a shared transition, use the '
        'evidence to locate their handoff boundary. Before returning, check every adjacent numeric '
        'end/start pair. Keep all supported distinct interactions; do not discard an action '
        'just to satisfy the constraint. '
        'Use concise English descriptions. The decoder enforces only this JSON format and '
        'legal ordered indices; YOU must choose all actions, boundaries, labels and descriptions '
        'from the complete evidence. Stage 3 will revisit each group visually for finer timing. '
        'Schema: {"intervals":[{"start_pair":0,"end_pair":1,"label":"TAKE",'
        '"description":"take the cup"}]}; use an empty list if no action is evidenced.\n'+
        json.dumps(transitions,ensure_ascii=False))

def refine_prompt(primitive,lo,hi):
    return ('Revisit these chronological frames and refine ONE hypothesized interaction. The '
        'initial description, class and boundaries can be wrong. Select the most prominent '
        'single evidenced interaction corresponding to it; correct class and boundaries if '
        'needed, or reject if unsupported. Do not simply copy the full padded interval. Return '
        'absolute seconds in ['+str(lo)+','+str(hi)+']. Give a short visible-evidence description, '
        'not a reasoning transcript. Schema if supported: {"valid":true,"start":0.0,"end":1.0,'
        '"label":"TAKE","description":"...","confidence":0.8,"evidence":"..."}; '
        'otherwise {"valid":false,"evidence":"..."}. Hypothesis: '+json.dumps(primitive))

def direct_prompt(lo,hi):
    return ('Identify ALL visible atomic physical interactions in these chronological frames. '
        'Use a single concrete action per interval and exact vocabulary labels or OTHER. '
        'Do not label the whole context just because an action occurs inside it. Return absolute '
        'seconds in ['+str(lo)+','+str(hi)+']. Schema: {"intervals":[{"start":0.0,"end":1.0,'
        '"label":"TAKE","description":"...","confidence":0.8,"evidence":"..."}]}. '
        'Return an empty list when no interaction is visible.')

def parse_object(raw):
    raw=raw.strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw=raw.split('\n',1)[1].rsplit('```',1)[0].strip()
    x=json.loads(raw)
    if not isinstance(x,dict): raise ValueError('top-level JSON must be an object')
    return x

def number(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

def interval(x,lo,hi,confidence=False):
    if not isinstance(x,dict): raise ValueError('interval is not an object')
    if not all(number(x.get(k)) for k in ['start','end']): raise ValueError('invalid time')
    if not (lo-1e-5<=x['start']<x['end']<=hi+1e-5): raise ValueError('out-of-range/empty interval')
    if x.get('label') not in CLASSES+['OTHER']: raise ValueError('invalid vocabulary label')
    if not isinstance(x.get('description'),str) or not x['description'].strip(): raise ValueError('missing description')
    if confidence and (not number(x.get('confidence')) or not 0<=x['confidence']<=1):
        raise ValueError('invalid confidence')

def validate(x,stage,lo,hi,pair_count=0):
    if stage=='transition':
        rows=x.get('transitions')
        if not isinstance(rows,list) or [v.get('pair') for v in rows]!=list(range(pair_count)):
            got=[v.get('pair') for v in rows] if isinstance(rows,list) else None
            raise ValueError(f'Expected exactly {pair_count} transition rows with pair IDs '
                f'{list(range(pair_count))}, got {got}. There are {pair_count+1} frames '
                f'but only {pair_count} transitions: no initial-frame caption and no pair {pair_count}.')
        for r in rows:
            if r.get('event') not in ['start','continuing','ending','none']: raise ValueError('invalid phase')
            if not isinstance(r.get('description'),str) or not r['description'].strip(): raise ValueError('empty transition')
    elif stage=='refine':
        if type(x.get('valid')) is not bool: raise ValueError('missing boolean validity')
        if not isinstance(x.get('evidence'),str) or not x['evidence'].strip(): raise ValueError('missing visual evidence')
        if x['valid']: interval(x,lo,hi,True)
    else:
        rows=x.get('intervals')
        if not isinstance(rows,list): raise ValueError('intervals must be a list')
        for r in rows: interval(r,lo,hi,stage=='direct')
        if stage=='grouping':
            if rows!=sorted(rows,key=lambda r:r['start']): raise ValueError('grouping not chronological')
            conflicts=[dict(previous_index=i,previous_label=a['label'],previous_end=a['end'],
                next_index=i+1,next_label=b['label'],next_start=b['start'],overlap_seconds=a['end']-b['start'])
                for i,(a,b) in enumerate(zip(rows,rows[1:])) if a['end']>b['start']+1e-5]
            if conflicts:
                raise ValueError('paper primitive grouping must be non-overlapping: '+json.dumps(conflicts)+
                    '. Regenerate the complete chronological interval list from the SAME evidence. '
                    'Choose the physical handoff boundary so every end[i] <= start[i+1]; '
                    'do not repeat the overlapping numbers, drop an interaction, or invent evidence.')
    return x

def iou(a,b):
    intersection=max(0.,min(a['end'],b['end'])-max(a['start'],b['start']))
    union=max(a['end'],b['end'])-min(a['start'],b['start'])
    return intersection/union if union>0 else 0.

def deduplicate(rows,threshold=.5):
    # Fixed class-aware hard NMS, applied to BOTH generated detection sets.
    kept=[]; removed=[]
    for row in sorted(rows,key=lambda r:(-r['confidence'],r['start'],r['end'],r['label'])):
        match=next((k for k in kept if row['label']==k['label'] and iou(row,k)>=threshold),None)
        if match is None: kept.append(row)
        else: removed.append(dict(prediction=row,suppressed_by=match))
    return kept,removed

def match_predictions(predictions,groundtruth,threshold,class_aware=True):
    # Score-ranked one-to-one greedy matching, as in temporal detection AP.
    used=set(); rows=[]
    for p in sorted(predictions,key=lambda x:-x['confidence']):
        candidates=[(iou(p,g),i) for i,g in enumerate(groundtruth)
            if i not in used and p['video_id']==g['video_id'] and
            (not class_aware or p['label']==g['label'])]
        q,j=max(candidates,default=(0.,-1))
        if j>=0 and q>=threshold: used.add(j); rows.append((p,j,q))
        else: rows.append((p,None,q))
    return rows,used

def average_precision(matches,ngt):
    if ngt==0: return None
    tp=0; precision=[]; recall=[]
    for n,(_,j,_) in enumerate(matches,1):
        tp+=j is not None; precision.append(tp/n); recall.append(tp/ngt)
    p=[0.]+precision+[0.]; r=[0.]+recall+[1.]
    for i in range(len(p)-2,-1,-1): p[i]=max(p[i],p[i+1])
    return sum((r[i]-r[i-1])*p[i] for i in range(1,len(r)) if r[i]!=r[i-1])

def metrics(predictions,groundtruth):
    report={}
    for t in SPEC['metrics_tiou']:
        classes={}
        for c in CLASSES:
            gt=[g for g in groundtruth if g['label']==c]; pp=[p for p in predictions if p['label']==c]
            matched,_=match_predictions(pp,gt,t)
            classes[c]=dict(gt=len(gt),predictions=len(pp),ap=average_precision(matched,len(gt)))
        aps=[v['ap'] for v in classes.values() if v['ap'] is not None]
        filtered=[p for p in predictions if p['confidence']>=SPEC['score_threshold']]
        matched,used=match_predictions(filtered,groundtruth,t); tp=len(used)
        ag,agu=match_predictions(filtered,groundtruth,t,False)
        n=len(filtered); g=len(groundtruth)
        errors=[(abs(p['start']-groundtruth[j]['start']),abs(p['end']-groundtruth[j]['end']))
            for p,j,_ in matched if j is not None]
        report[str(t)]=dict(map=sum(aps)/len(aps) if aps else None,per_class=classes,
            tp=tp,fp=n-tp,fn=g-tp,precision=tp/n if n else 0.,recall=tp/g if g else 0.,
            f1=2*tp/(n+g) if n+g else 0.,predictions=n,groundtruth=g,
            class_agnostic_tp=len(agu),class_agnostic_f1=2*len(agu)/(n+g) if n+g else 0.,
            matched_start_mae_sec=sum(a for a,b in errors)/len(errors) if errors else None,
            matched_end_mae_sec=sum(b for a,b in errors)/len(errors) if errors else None)
    return report
