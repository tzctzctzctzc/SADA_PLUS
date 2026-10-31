"""B/C共用动作定义与仅源训练集的可审核视频示例池。"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace

from tools.baseline_protocol import read,save,sha,code_identity
from tools.run_qwen_gt_oracle import BASE,LABELS,OPTIONS,LETTERS,option_order,cid,prepare_video,safe_row
from tools.qwen_source_interface import build_native_video_messages,mixed_frame_grid
from tools.refresh_frames import video_paths_by_id

DEFINITIONS=dict(zip(LABELS,(
    'take, pick up, retrieve or grasp an object from its resting place; track the object becoming held or moved away',
    'put, place, lay down, set down or return an object onto a surface or location; track placement or release, rather than retrieval',
    'wash, rinse, clean or wipe an object or surface; cleaning can be brief and need not involve running water',
    'open a door, drawer, lid, container or closure; the closure itself moves toward an open state',
    'close or shut a door, drawer, lid or container, including screwing a closure on; the closure moves toward a closed state',
    'put in, place into, load, pack, fit, slide or insert an object into an interior, opening, rack or receptacle; not limited to a narrow slot or forceful insertion',
    'turn on, activate, start or ignite a device, light, flame or water flow; use the activation direction, not the mere presence of a hand near a control',
    'cut, slice, chop, dice, trim or divide an object with a tool; a single cutting motion also qualifies',
    'turn off, deactivate, stop or extinguish a device, light, flame or water flow; distinguish stopping from starting',
    'pour, tip, drizzle, shake or sieve contents out of a vessel into or onto a destination; follow the transferred contents rather than the vessel placement',
)))
DEFINITIONS.update(OTHER_ACTION='a visible hand-object action outside these ten classes',
    MIXED_ACTIONS='multiple different actions with no clearly dominant supported action',
    NO_ACTION='no visible hand-object action; a persistent object state alone is insufficient',
    UNCERTAIN='insufficient visible evidence to determine the action or its direction')


def prompt(key):
    return ('Classify the action in the QUERY video, using its eight frames in chronological order. '
        'Use the dataset meanings below rather than only a literal reading of the class name. '
        'Track the manipulated object and its change over time; distinguish the action from nearby incidental hand motions. '
        'PUT describes placement onto a surface or location; INSERT also includes ordinary putting/loading INTO an interior or receptacle. '
        'OPEN/CLOSE concern the closure itself; TAKE/PUT/INSERT concern the moved object. '
        'Use visible temporal evidence for opposite directions, but do not require an ideal complete action sequence. '
        'If the action or direction is unsupported, choose an appropriate rejection option.\n'+
        '\n'.join('%s: %s (%s)'%(letter,label,DEFINITIONS[label]) for letter,label in zip(LETTERS,option_order(key)))+
        '\nOnly classify the QUERY, never the reference videos. Reply immediately with exactly one letter A to N, without explanation.')


def source_candidates(database,source_splits,limit=3):
    """先按可见时长和非重叠筛选，再按固定SHA选不同视频；不使用VLM/目标结果选例。"""
    groups=defaultdict(list)
    for video,record in database.items():
        if record['subset'].lower() not in source_splits:continue
        annotations=record.get('annotations',[])
        for index,a in enumerate(annotations):
            label=int(a['label_id']);start,end=map(float,a['segment'])
            if not 0<=label<10 or not .8<=end-start<=12:continue
            if any(j!=index and int(b['label_id'])!=label and min(end,b['segment'][1])>max(start,b['segment'][0]) for j,b in enumerate(annotations)):continue
            key=cid(video,start,end)
            groups[label].append(dict(candidate_id=key,video_id=video,start=start,end=end,label=label,
                rank_sha256=hashlib.sha256(('source-demo-seed123:'+key).encode()).hexdigest()))
    result=[]
    for label in range(10):
        used=set();chosen=[]
        for row in sorted(groups[label],key=lambda r:(abs((r['end']-r['start'])-2.5),r['rank_sha256'])):
            if row['video_id'] in used:continue
            used.add(row['video_id']);chosen.append(row)
            if len(chosen)==limit:break
        if len(chosen)!=limit:raise ValueError('没有足够的独立源视频示例: '+LABELS[label])
        result+=chosen
    return result


def prepare(root,video_root,recovery_root):
    root.mkdir(parents=True,exist_ok=False);selected={};membership={};unique={}
    for shift in ('S1','S2','S3','S4','S5','S6'):
        cfg=read(BASE/shift/'uda/config.json');annotation=cfg['baseline']['annotation_for_offline_evaluation'];db=read(annotation)['database']
        rows=source_candidates(db,cfg['src_train_split']);selected[shift]=rows
        membership[shift]=dict(source_splits=cfg['src_train_split'],target_splits=cfg['tgt_train_split'],
            annotation=annotation,annotation_sha256=sha(annotation),config_sha256=sha(BASE/shift/'uda/config.json'))
        for row in rows:unique[row['candidate_id']]={k:row[k] for k in ('candidate_id','video_id','start','end')}
    save(root/'selection.json',dict(code=code_identity(),source_training_only=True,selection_rule='no-other-annotated-class-overlap; duration-near-2.5s; distinct-videos; SHA-tiebreak',
        candidates=selected,membership=membership))
    grouped=defaultdict(list)
    for row in unique.values():grouped[row['video_id']].append(row)
    args=SimpleNamespace(output=root,video_paths=video_paths_by_id(video_root),recovery_root=recovery_root)
    candidates=[];videos=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(prepare_video,v,rows,args) for v,rows in sorted(grouped.items())]
        for task in tasks:
            good,bad,video=task.result()
            if bad or any(row['unique_frames']!=8 for row in good):raise ValueError('示例缺少8个真实独立帧')
            candidates+=good;videos.append(video);print('SOURCE_VIDEOS_READY',len(videos),len(tasks),flush=True)
    save(root/'manifest.json',dict(candidates=candidates,videos=videos,selection_sha256=sha(root/'selection.json')))
    from PIL import Image,ImageDraw
    lookup={x['candidate_id']:x for x in candidates}
    for shift,rows in selected.items():
        canvas=Image.new('RGB',(1280,len(rows)*116),'white');draw=ImageDraw.Draw(canvas)
        for n,row in enumerate(rows):
            draw.text((2,n*116),'%s %s option%d %s %.2f-%.2f %s'%(shift,LABELS[row['label']],n%3,row['video_id'],row['start'],row['end'],row['candidate_id'][:10]),fill='black')
            for k,path in enumerate(lookup[row['candidate_id']]['frame_paths']):
                im=Image.open(path).convert('RGB');im.thumbnail((160,90));canvas.paste(im,(160*k,n*116+20))
        canvas.save(root/(shift+'_review.jpg'))
    save(root/'prepared.json',dict(status='passed',manifest_sha256=sha(root/'manifest.json'),selection_sha256=sha(root/'selection.json')))


def freeze(root,review):
    selection=read(root/'selection.json');lookup={r['candidate_id']:r for r in read(root/'manifest.json')['candidates']};banks={}
    if set(review['choices'])!=set(selection['candidates']):raise ValueError('必须审阅六方向')
    for shift,choices in review['choices'].items():
        if set(choices)!=set(LABELS):raise ValueError('每类须一个源示例')
        pool={x['candidate_id']:x for x in selection['candidates'][shift]};bank=[];meta=selection['membership'][shift]
        db=read(meta['annotation'])['database']
        if sha(meta['annotation'])!=meta['annotation_sha256']:raise ValueError('源GT版本改变')
        for label,name in enumerate(LABELS):
            choice=choices[name];key=choice['candidate_id'];original=pool[key];row=lookup[key]
            if not choice['reason'] or original['label']!=label or db[row['video_id']]['subset'].lower() not in meta['source_splits']:
                raise ValueError('示例审核/源成员/类别不符')
            if db[row['video_id']]['subset'].lower() in meta['target_splits']:raise ValueError('目标视频不能成为源示例')
            for p,h in zip(row['frame_paths'],row['frame_sha256']):
                if sha(p)!=h:raise ValueError('源示例帧改变')
            bank.append(dict(label=name,reason=choice['reason'],**row))
        banks[shift]=bank
    output=root/'frozen_banks.json'
    if output.exists():raise ValueError('禁止覆盖已冻结示例')
    save(output,dict(banks=banks,review=review,selection_sha256=sha(root/'selection.json'),manifest_sha256=sha(root/'manifest.json'),
        source_frame_min_pixels=224*224,source_frame_max_pixels=224*224,source_frame_count=8))


def messages(row,bank):
    safe_row(row);content=[]
    for i,example in enumerate(bank):
        safe_row({k:v for k,v in example.items() if k not in ('label','reason')})
        if example['label'] not in LABELS:raise ValueError('未知源示例类别')
        content.append(dict(type='text',text='REFERENCE %d: source-domain labeled example, class %s.'%(i+1,example['label'])))
        video=build_native_video_messages(example['frame_paths'],'',224*224,224*224,8/(example['end']-example['start']))[0]['content'][0]
        video.update(mixed_frame_grid(example['frame_paths'],224*224,224*224));content.append(video)
        content.append(dict(type='text',text='REFERENCE %d label: %s. %s'%(i+1,example['label'],DEFINITIONS[example['label']])))
    content.append(dict(type='text',text='QUERY: classify this video; its label is unknown.'))
    video=build_native_video_messages(row['frame_paths'],'',224*224,448*448,8/(row['end']-row['start']))[0]['content'][0]
    video.update(mixed_frame_grid(row['frame_paths'],224*224,448*448));content+=[video,dict(type='text',text=prompt(row['candidate_id']))]
    return [dict(role='user',content=content)]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path)
    p.add_argument('--video-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/videos'))
    p.add_argument('--recovery-root',type=Path,default=Path('/vepfs/vlms/data/EPIC_KITCHENS/rq2_frame_recovery_qwen_0924'))
    args=p.parse_args()
    if args.review:freeze(args.output,read(args.review))
    else:prepare(args.output,args.video_root,args.recovery_root)
